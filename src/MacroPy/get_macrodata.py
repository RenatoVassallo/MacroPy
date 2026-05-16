import datetime
import io
import logging
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import fredapi as fa
import numpy as np
import pandas as pd
import requests

# Set up basic logging
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# BCRP date / value parsing helpers
# ---------------------------------------------------------------------------

# Spanish 3-letter month abbreviations as used in BCRP responses (English `ing`
# lang switches to English for most periods but daily HTML always uses Spanish).
_BCRP_MONTH_MAP: Dict[str, int] = {
    # Spanish
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
    # English
    "jan": 1, "apr": 4, "aug": 8, "dec": 12,
    # Identical in both: feb, mar, may, jun, jul, sep, oct, nov already above
}

# Match `Q1.96`, `Q1.1996`, etc.
_RE_QUARTER = re.compile(r"^Q([1-4])\.(\d{2,4})$", re.IGNORECASE)
# Match `Jan.2018`, `Ene.2018`, `Set.1999` (3-letter month + . + 2/4-digit year).
_RE_MONTHLY = re.compile(r"^([A-Za-zñÑ]{3})\.(\d{2,4})$")
# Match `02Ene97`, `15Jan2025` (day + 3-letter month + 2/4-digit year).
_RE_DAILY = re.compile(r"^(\d{1,2})([A-Za-zñÑ]{3})(\d{2,4})$")
# Year-only.
_RE_ANNUAL = re.compile(r"^(\d{4})$")

# Quarter end-of-period months used by BCRP convention (matches legacy behavior).
_QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


def _two_digit_year_to_four(yy: int, pivot: int = 50) -> int:
    """Heuristically expand a 2-digit year. `49` → 2049, `50` → 1950."""
    return 1900 + yy if yy >= pivot else 2000 + yy


def _parse_bcrp_date(date_str: object) -> pd.Timestamp:
    """
    Parse a date label returned by the BCRP API into a pandas Timestamp.

    Recognises annual (`YYYY`), quarterly (`QX.YY` / `QX.YYYY`), monthly
    (English or Spanish abbreviation + year), and daily (`ddMMMyy` /
    `ddMMMyyyy`) formats. Returns `pd.NaT` if no pattern matches.
    """
    if not isinstance(date_str, str):
        return pd.NaT
    s = date_str.strip()
    if not s:
        return pd.NaT

    try:
        m = _RE_ANNUAL.match(s)
        if m:
            return pd.Timestamp(int(m.group(1)), 1, 1)

        m = _RE_QUARTER.match(s)
        if m:
            quarter = int(m.group(1))
            year = int(m.group(2))
            if year < 100:
                year = _two_digit_year_to_four(year)
            return pd.Timestamp(year, _QUARTER_END_MONTH[quarter], 1)

        m = _RE_MONTHLY.match(s)
        if m:
            month_key = m.group(1).lower()
            year = int(m.group(2))
            month = _BCRP_MONTH_MAP.get(month_key)
            if month is None:
                return pd.NaT
            if year < 100:
                year = _two_digit_year_to_four(year)
            return pd.Timestamp(year, month, 1)

        m = _RE_DAILY.match(s)
        if m:
            day = int(m.group(1))
            month_key = m.group(2).lower()
            year = int(m.group(3))
            month = _BCRP_MONTH_MAP.get(month_key)
            if month is None:
                return pd.NaT
            if year < 100:
                year = _two_digit_year_to_four(year)
            return pd.Timestamp(year, month, day)
    except (ValueError, KeyError):
        return pd.NaT

    return pd.NaT


def convert_date(date_str):
    """
    Backward-compatible wrapper around :func:`_parse_bcrp_date`.

    Kept so that downstream code importing ``convert_date`` keeps working.
    """
    return _parse_bcrp_date(date_str)


# Strings that BCRP uses to denote missing observations.
_BCRP_MISSING = {"n.d.", "N.D.", "n.d", "N.D", "nan", "NaN", "", "—", "-"}


def _to_float(value: object) -> float:
    """Coerce a BCRP value (str/None/number) to a float, mapping sentinels to NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s in _BCRP_MISSING:
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


# ---------------------------------------------------------------------------
# BCRP fetcher
# ---------------------------------------------------------------------------

_BCRP_API_ROOT = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
_BCRP_DAILY_ROOT = "https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/"
_BCRP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

# Accept either: a list of codes; a list of (code, name) tuples; or a dict
# {code: name}. We always normalise to two parallel lists.
SeriesCodes = Union[
    Sequence[str],
    Sequence[Tuple[str, str]],
    Mapping[str, str],
]


def _resolve_series_codes(
    series_codes: SeriesCodes,
    names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Normalise the user's series specification to ``(codes, names)`` lists."""
    if isinstance(series_codes, Mapping):
        codes = list(series_codes.keys())
        resolved_names = [str(series_codes[c]) for c in codes]
    else:
        items = list(series_codes)
        if not items:
            raise ValueError("`series_codes` cannot be empty.")
        if isinstance(items[0], (tuple, list)):
            codes = [str(c) for c, _ in items]
            resolved_names = [str(n) for _, n in items]
        else:
            codes = [str(c) for c in items]
            resolved_names = codes.copy()

    if names is not None:
        names = list(names)
        if len(names) != len(codes):
            raise ValueError(
                f"`names` has length {len(names)} but `series_codes` has length {len(codes)}."
            )
        resolved_names = [str(n) for n in names]

    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError(f"Column names must be unique. Got: {resolved_names}")

    return codes, resolved_names


def _format_end_period(end_period: Optional[str], frequency: str) -> str:
    """Build a sensible default `end_period` for each BCRP frequency."""
    if end_period is not None:
        return end_period
    today = datetime.datetime.now()
    if frequency == "Q":
        return f"{today.year}-{(today.month - 1) // 3 + 1}"
    if frequency in ("M", "D"):
        return today.strftime("%Y-%m") if frequency == "M" else today.strftime("%Y-%m-%d")
    if frequency == "A":
        return str(today.year)
    return today.strftime("%Y-%m")


def _fetch_bcrp_json_single(code: str, start: str, end: str, lang: str, timeout: float) -> dict:
    """
    Hit the BCRP JSON endpoint for a single code and return the parsed body.

    The BCRP API supports batched calls (`A-B-C`) but reorders the response
    by series family rather than by URL order, with no code field to map
    columns back. Fetching one code at a time eliminates the alignment
    ambiguity entirely.
    """
    url = f"{_BCRP_API_ROOT}{code}/json/{start}/{end}/{lang}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    # BCRP returns HTTP 200 with an HTML error page for invalid codes — detect it.
    content_type = response.headers.get("Content-Type", "")
    body_bytes = response.content
    if "json" not in content_type.lower() and not body_bytes.lstrip().startswith(b"{"):
        raise ValueError(
            f"BCRP returned a non-JSON response (likely invalid series code '{code}'). URL: {url}"
        )
    # BCRP sometimes prepends a UTF-8 BOM; decode explicitly to handle it.
    import json as _json
    return _json.loads(body_bytes.decode("utf-8-sig"))


def _periods_to_series(payload: dict, code: str) -> pd.Series:
    """Turn a single-code JSON payload into a date-indexed pd.Series."""
    periods = payload.get("periods", []) or []
    if not periods:
        return pd.Series(dtype=float, name=code)

    dates: List[pd.Timestamp] = []
    values: List[float] = []
    for period in periods:
        raw_date = period.get("name")
        vs = period.get("values", []) or []
        v = vs[0] if vs else None
        d = _parse_bcrp_date(raw_date)
        if pd.isna(d):
            continue
        dates.append(d)
        values.append(_to_float(v))

    s = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=code, dtype=float)

    duplicates = s.index.duplicated().sum()
    if duplicates:
        logging.warning(
            "BCRP returned %d duplicate date(s) for '%s'; keeping the last occurrence.",
            duplicates, code,
        )
        s = s[~s.index.duplicated(keep="last")]

    return s.sort_index()


def _build_bcrp_dataframe(
    codes: List[str], names: List[str], start: str, end: str, lang: str, timeout: float,
) -> pd.DataFrame:
    """Fetch each code separately and merge into a single date-keyed DataFrame."""
    series_list: List[pd.Series] = []
    for code, name in zip(codes, names):
        payload = _fetch_bcrp_json_single(code, start, end, lang, timeout)
        s = _periods_to_series(payload, code)
        if s.empty:
            logging.warning(
                "BCRP returned 0 periods for '%s'. Check the code, frequency, and date range.",
                code,
            )
        s.name = name
        series_list.append(s)

    df = pd.concat(series_list, axis=1).sort_index()
    df.index.name = "date"
    return df.reset_index()


def _fetch_bcrp_daily_html(code: str, timeout: float) -> pd.DataFrame:
    """Scrape the BCRP daily HTML viewer for a single series, returning a 2-column frame."""
    url = f"{_BCRP_DAILY_ROOT}{code}/html/"
    response = requests.get(url, headers={"User-Agent": _BCRP_USER_AGENT}, timeout=timeout)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    # Pick the first table that has a 'Fecha' column — robust to layout changes.
    data_table = None
    for t in tables:
        if any(str(c).strip().lower() == "fecha" for c in t.columns):
            data_table = t
            break
    if data_table is None:
        raise ValueError(
            f"Could not locate the data table for BCRP daily series '{code}'. "
            "The page layout may have changed."
        )

    df = data_table.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # The non-date column holds the value; rename for caller's convenience.
    value_cols = [c for c in df.columns if c.lower() != "fecha"]
    if len(value_cols) != 1:
        raise ValueError(
            f"Expected exactly one value column for daily series '{code}', got {value_cols}."
        )
    df = df.rename(columns={"Fecha": "date", value_cols[0]: code})
    return df


def get_bcrp_data(
    series_codes: SeriesCodes,
    frequency: str = "M",
    start_period: str = "2003-1",
    end_period: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    lang: str = "ing",
    timeout: float = 30.0,
    date_index: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Retrieve time series from the BCRP API and return a tidy DataFrame.

    Parameters
    ----------
    series_codes : list[str] | list[tuple[str, str]] | dict[str, str]
        BCRP series codes. Accepts three input shapes:

        - ``["PN02538AQ", "PN02533AQ"]`` — columns are labelled with the codes.
        - ``[("PN02538AQ", "pbi"), ("PN02533AQ", "invpriv")]`` — pairs of code
          and the desired column name (URL-order preserved).
        - ``{"PN02538AQ": "pbi", "PN02533AQ": "invpriv"}`` — same as above using
          a mapping (insertion order preserved on Python 3.7+).

    frequency : {'M', 'Q', 'D', 'A'}, default 'M'
        Sampling frequency. 'A' falls back to the JSON endpoint with `start_period`
        formatted as a 4-digit year.
    start_period, end_period : str, optional
        Date window in BCRP-native format (``YYYY-M``, ``YYYY-Q``, ``YYYY-MM-DD``,
        or ``YYYY``). ``end_period`` defaults to "today" in the appropriate format.
    names : list[str], optional
        Override the resolved column names. Must have the same length as
        ``series_codes``. If omitted, names are taken from the input shape above.
    lang : {'ing', 'esp'}, default 'ing'
        Response language. Only affects month-name labels; date parsing handles
        both transparently.
    timeout : float, default 30
        HTTP timeout in seconds.
    date_index : bool, default False
        If True, return a DataFrame indexed by date instead of a `date` column.

    Returns
    -------
    pandas.DataFrame
        Sorted ascending by date, with NaN for missing observations and one
        numeric column per requested series.
    """
    codes, resolved_names = _resolve_series_codes(series_codes, names=names)

    freq = frequency.upper()
    if freq not in {"M", "Q", "D", "A"}:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Use 'D', 'M', 'Q', or 'A'."
        )

    end = _format_end_period(end_period, freq)

    try:
        if freq in {"M", "Q", "A"}:
            df = _build_bcrp_dataframe(codes, resolved_names, start_period, end, lang, timeout)
        else:  # daily
            frames = [_fetch_bcrp_daily_html(c, timeout) for c in codes]
            # Parse date column and clean each frame independently, then outer-merge.
            cleaned = []
            for code, frame in zip(codes, frames):
                frame = frame.copy()
                frame["date"] = frame["date"].apply(_parse_bcrp_date)
                frame[code] = frame[code].map(_to_float)
                frame = frame.dropna(subset=["date"]).sort_values("date")
                cleaned.append(frame[["date", code]])
            df = cleaned[0]
            for extra in cleaned[1:]:
                df = df.merge(extra, on="date", how="outer")
            # Apply requested names + date window
            df = df.rename(columns=dict(zip(codes, resolved_names)))
            df = df.sort_values("date").reset_index(drop=True)
            start_ts = pd.to_datetime(start_period, errors="coerce")
            end_ts = pd.to_datetime(end, errors="coerce")
            if pd.notna(start_ts):
                df = df[df["date"] >= start_ts]
            if pd.notna(end_ts):
                df = df[df["date"] <= end_ts]
            df = df.reset_index(drop=True)

    except requests.RequestException as e:
        logging.error("BCRP request failed: %s", e)
        return None

    if date_index:
        df = df.set_index("date")
    return df


def get_fred_data(series_codes, series_names, frequency, api_key, start_period='2003-1', end_period=None):
    """
    Retrieve data from FRED using given series_codes.

    Parameters:
    series_codes (list of str): List of FRED series IDs.
    series_names (list of str): List of names to assign to the retrieved data columns.
    frequency (str): Frequency of the data ('a' for annual, 'q' for quarterly, 'm' for monthly).
    api_key (str): API key for accessing FRED.
    start_period (str, optional): The starting period for the data retrieval in 'yyyy-m' format. Defaults to '2003-1'.
    end_period (str, optional): The ending period for the data retrieval in 'yyyy-m' format. Defaults to current month.

    Returns:
    pandas.DataFrame: DataFrame with a datetime 'date' column and other columns for the retrieved data.
    """
    if end_period is None:
        end_period = datetime.datetime.now().strftime("%Y-%m")

    fred = fa.Fred(api_key=api_key)
    df_fred = pd.DataFrame()

    for i, indicator in enumerate(series_codes):
        data = fred.get_series(indicator, observation_start=start_period, 
                               observation_end=end_period, frequency=frequency)
        data.name = series_names[i]
        data = pd.DataFrame(data)
        df_fred = pd.concat([df_fred, data], axis=1)

    # Adjusting the index to a 'date' column
    df_fred.reset_index(inplace=True)
    df_fred.rename(columns={'index': 'date'}, inplace=True)

    # Converting 'date' to datetime format
    df_fred['date'] = pd.to_datetime(df_fred['date'])

    return df_fred


import requests
import pandas as pd
import io
import time
from functools import reduce
import pycountry

BASE_URL = "https://stats.bis.org/api/v1/data"

# ----------------------------------------------------------
# Low-level helper: fetch one BIS series as CSV
# ----------------------------------------------------------
def _fetch_bis_series_raw(series_code, label, start_str, end_str):
    """
    Fetch a single BIS SDMX series (v1) as CSV.
    Keeps TIME_PERIOD, country (ISO2), and value column (renamed to label).
    """
    flow, key = series_code.split("/", 1)
    url = f"{BASE_URL}/{flow}/{key}/all"

    headers = {
        "Accept": "application/vnd.sdmx.data+csv;version=1.0.0;labels=id;timeFormat=original;keys=none"
    }
    params = {
        "startPeriod": start_str,
        "endPeriod": end_str,
        "detail": "dataonly",
    }

    r = requests.get(url, headers=headers, params=params)
    try:
        r.raise_for_status()
    except Exception:
        print("Status:", r.status_code)
        print("Response snippet:\n", r.text[:1500])
        raise

    df = pd.read_csv(io.StringIO(r.text))

    # Filter to monthly frequency
    if "FREQ" in df.columns:
        df = df[df["FREQ"] == "M"]

    # Detect country column (BIS uses REF_AREA normally)
    country_col = None
    for cand in ["REF_AREA", "LOC", "LOCATION", "LOCATION_CODE"]:
        if cand in df.columns:
            country_col = cand
            break

    if country_col is None:
        print("Columns:", df.columns)
        raise ValueError("Could not find country column in BIS response.")

    # Rename value + country
    if "OBS_VALUE" in df.columns:
        df = df.rename(columns={"OBS_VALUE": label})
    elif "value" in df.columns:
        df = df.rename(columns={"value": label})
    else:
        print("Columns:", df.columns)
        raise ValueError("Could not find OBS_VALUE/value in BIS response.")

    df = df.rename(columns={country_col: "country"})

    # Keep essentials
    return df[["TIME_PERIOD", "country", label]]

# ----------------------------------------------------------
# Single-country BIS extraction using ISO-3
# ----------------------------------------------------------
def get_bis_data_single(iso3, series_map, date_range):
    """
    Fetch BIS data for a single ISO-3 country over a given date range.
    Returns columns: date, isocode (ISO-3), and one column per series label.
    """
    # Convert ISO-3 → ISO-2 (to match BIS REF_AREA)
    country_obj = pycountry.countries.get(alpha_3=iso3.upper())
    if country_obj is None:
        raise ValueError(f"Invalid ISO-3 code: {iso3}")
    iso2 = country_obj.alpha_2

    # Extract YYYY-MM strings from date_range
    start_str = date_range.min().strftime("%Y-%m")
    end_str   = date_range.max().strftime("%Y-%m")

    dfs = []
    for bis_code, label in series_map.items():
        df_raw = _fetch_bis_series_raw(bis_code, label, start_str, end_str)

        # Filter to selected country (ISO2)
        df_cty = df_raw[df_raw["country"] == iso2].copy()

        dfs.append(df_cty)
        time.sleep(0.25)

    # Merge all series
    df_merged = reduce(
        lambda l, r: pd.merge(l, r, on=["TIME_PERIOD", "country"], how="outer"),
        dfs
    )

    # Clean up
    df_merged = df_merged.sort_values("TIME_PERIOD")
    df_merged["TIME_PERIOD"] = pd.to_datetime(df_merged["TIME_PERIOD"])

    # Restrict to requested period (just in case)
    df_merged = df_merged[
        (df_merged["TIME_PERIOD"] >= date_range.min()) &
        (df_merged["TIME_PERIOD"] <= date_range.max())
    ]

    # Rename & format
    df_merged = df_merged.rename(columns={"TIME_PERIOD": "date"})
    df_merged["isocode"] = iso3.upper()

    # Reorder columns: date, isocode, series...
    series_cols = list(series_map.values())
    df_merged = df_merged[["date", "isocode"] + series_cols]

    return df_merged

# ----------------------------------------------------------
# Multi-country wrapper
# ----------------------------------------------------------
def get_bis_data(countries_iso3, series_map, date_range):
    """
    Fetch BIS data for a list of ISO-3 countries.
    Returns a panel: date × isocode × series.
    """
    all_dfs = []
    for iso3 in countries_iso3:
        print(f"Fetching BIS data for {iso3}...")
        df_cty = get_bis_data_single(iso3, series_map, date_range)
        all_dfs.append(df_cty)

    df_panel = pd.concat(all_dfs, ignore_index=True)
    df_panel = df_panel.sort_values(["isocode", "date"]).reset_index(drop=True)
    print("Data fetching complete.")
    return df_panel