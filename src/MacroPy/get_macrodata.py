import pandas as pd
import numpy as np
import datetime
import requests
import logging
import fredapi as fa

# Set up basic logging
logging.basicConfig(level=logging.INFO)


def convert_date(date_str):
    """
    Convert date string to datetime object. Handles annual (yyyy), quarterly (Qx.yy), 
    monthly (MMM.yyyy), and daily (ddMMMyy) formats.
    """
    try:
        # If the date_str is of length 4 and numeric, it's likely an annual format
        if len(date_str) == 4 and date_str.isdigit():
            return pd.to_datetime(f'{date_str}-01-01')
        
        # Try parsing as quarterly data in the format 'Qx.yy'
        if date_str.startswith(('Q1.', 'Q2.', 'Q3.', 'Q4.')):
            quarter, year = date_str.split('.')
            year = '19' + year if int(year) >= 50 else '20' + year  # Convert 2-digit year to 4-digit
            quarter_to_month = {'Q1': '03', 'Q2': '06', 'Q3': '09', 'Q4': '12'}
            month = quarter_to_month[quarter]
            return pd.to_datetime(f'{year}-{month}-01')

        # Try parsing as monthly data in the format 'MMM.yyyy'
        monthly_date = pd.to_datetime(date_str, format='%b.%Y', errors='coerce')
        if not pd.isna(monthly_date):
            return monthly_date

        # Try parsing as daily data in the format 'ddMMMyy' (e.g., 02Ene97)
        month_map = {
            'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Ago': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
        }

        day = date_str[:2]  # Extract day
        month_abbr = date_str[2:5]  # Extract month abbreviation
        year_suffix = date_str[5:]  # Extract year

        if month_abbr not in month_map or not year_suffix.isdigit():
            return pd.NaT  # Return NaT for invalid dates

        month = month_map[month_abbr]

        # Convert 2-digit year to 4-digit year (assume 19XX for years >= 50, else 20XX)
        year = f"19{year_suffix}" if int(year_suffix) >= 50 else f"20{year_suffix}"

        return pd.to_datetime(f"{year}-{month}-{day}", format='%Y-%m-%d')

    except Exception as e:
        return pd.NaT  # Return NaT for errors

def get_bcrp_data(series_codes, frequency='M', start_period='2003-1', end_period=None):
    """
    Retrieves time series data from the BCRP API for specified series codes.

    Parameters:
    -----------
    series_codes : list of str
        List of series codes to retrieve data for.
    frequency : str, optional
        Frequency of the data ('D' for daily, 'M' for monthly, 'Q' for quarterly).
        Default is 'M' (monthly).
    start_period : str, optional
        Start period for data in 'yyyy-m' format for monthly data or 'yyyy' for annual data.
        Default is '2003-1'.
    end_period : str, optional
        End period for data retrieval. Defaults to the current date if not provided.

    Returns:
    --------
    pandas.DataFrame
        DataFrame where each column is a series, rows represent time periods, and a 'date' column contains
        the periods in datetime format.

    Raises:
    -------
    requests.RequestException
        If there is a network or API issue.
    Exception
        For any other errors during data retrieval and processing.
    """

    try:
        concat_code = '-'.join(series_codes) + '/'
        
        if frequency in ['Q', 'M']:
            
            if end_period is None:
                end_period = datetime.datetime.now().strftime("%Y-%m")
            root = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
            format = "json/"
            date = f'{start_period}/{end_period}/'
            lang = 'ing'
            url = root + concat_code + format + date + lang
            response = requests.get(url)

            if response.status_code != 200:
                logging.error(f"API request failed with status code {response.status_code}")
                return None

            response_json = response.json()
            series_info = response_json['config']['series']
            periods_info = response_json['periods']
            series_names = [item['name'] for item in series_info]

            transformed_data = []
            for period in periods_info:
                date = period['name']
                values = period['values']
                period_dict = {'date': date}
                for i, name in enumerate(series_names):
                    value = values[i] if i < len(values) else 'n.d.'
                    period_dict[name] = float(value) if value != 'n.d.' else np.nan
                transformed_data.append(period_dict)

            df = pd.DataFrame(transformed_data)
            df['date'] = df['date'].apply(convert_date)
            return df

        elif frequency == 'D':
            if end_period is None:
                end_period = datetime.datetime.now().strftime("%Y-%m-%d")
            root = "https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/"
            format = "html/"
            url = root + concat_code + format
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                logging.error(f"API request failed with status code {response.status_code}")
                return None

            # Extract tables from HTML
            tables = pd.read_html(response.text)
            df = tables[1].copy()
            df.rename(columns={'Fecha': 'date'}, inplace=True)
            df['date'] = df['date'].apply(convert_date)
            df = df.dropna(subset=['date']).reset_index(drop=True)
            df = df[(df['date'] >= pd.to_datetime(start_period)) & (df['date'] <= pd.to_datetime(end_period))]

            return df

        else:
            raise ValueError("Invalid frequency. Use 'D' for daily, 'M' for monthly, or 'Q' for quarterly.")

    except requests.RequestException as e:
        logging.error(f"An error occurred while making the API request: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
    

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