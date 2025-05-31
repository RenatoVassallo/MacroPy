import numpy as np
import pandas as pd
import math
from plotnine import *
from mizani.breaks import breaks_date
import warnings
warnings.filterwarnings("ignore")

def generate_series_plot(yy, yy_dates=None, yy_names=None, series_titles=None, title=None, 
                         color_scheme=1, n_breaks: int = 10, zero_line: bool = False):
    """
    Generate time series plots using plotnine (ggplot-style).
    
    Parameters:
        yy (array-like or DataFrame): Time series data (variables in columns).
        yy_dates (list or pd.DatetimeIndex): Dates for the time series.
        yy_names (list): List of variable names (column names) if `yy` is array-like.
        series_titles (list): Optional custom names for each variable (replacing yy_names).
        title (str): Plot title.
        color_scheme (int): Color scheme option (1–4).
        n_breaks (int): Number of x-axis breaks (year ticks).
        zero_line (bool): Whether to include a horizontal zero line.
    
    Returns:
        plotnine.ggplot object
    """
    # Define conservative color schemes
    color_schemes = {
        1: "#1f77b4",  # Dark Blue
        2: "#c44e52",  # Soft Red
        3: "#55a868",  # Muted Green
        4: "#8172b3"   # Soft Purple
    }
    line_color = color_schemes.get(color_scheme, "#1f77b4")

    # Handle input types
    if isinstance(yy, pd.DataFrame):
        df = yy.copy()
        if yy_dates is not None:
            df.index = pd.to_datetime(yy_dates)
        yy_names = list(df.columns)
    else:
        df = pd.DataFrame(yy, index=pd.to_datetime(yy_dates), columns=yy_names)

    # Reshape to long format
    df = df.rename_axis("index").reset_index().melt(id_vars='index', var_name='Variable', value_name='Value')

    # Preserve variable order
    df['Variable'] = pd.Categorical(df['Variable'], categories=yy_names, ordered=True)

    # Replace with custom titles if provided
    if series_titles:
        title_dict = dict(zip(yy_names, series_titles))
        df['Variable'] = df['Variable'].map(title_dict)
        df['Variable'] = pd.Categorical(df['Variable'], categories=series_titles, ordered=True)

    num_vars = len(df['Variable'].unique())
    num_rows = int(np.ceil(num_vars / 2))

    # Build base plot
    plot = (
        ggplot(df, aes(x='index', y='Value', group='Variable')) +
        geom_line(color=line_color, size=1.2) +
        facet_wrap('~Variable', ncol=2, scales='free') +
        scale_x_datetime(date_labels="%y", breaks=breaks_date(n=n_breaks)) +
        labs(title=title, x="", y="") +
        theme(
            figure_size=(10, num_rows * 3),
            plot_title=element_text(size=14, face="bold") if title else element_blank(),
            panel_background=element_rect(fill="white", color="white"),
            plot_background=element_rect(fill="white", color="white"),
            strip_background=element_rect(fill="white", color="white"),
            panel_grid_major=element_line(color="grey", linetype="dashed", size=0.8, alpha=0.2),
            panel_grid_minor=element_blank(),
            strip_text=element_text(size=12, weight='bold'),
            axis_text_x=element_text(hjust=1, margin={'t': 5}),
            axis_line_x=element_line(color="black", size=1),
            axis_line_y=element_line(color="black", size=1),
            legend_position="none"
        )
    )

    # Optionally add zero line
    if zero_line:
        plot += geom_hline(yintercept=0, linetype='dashed', color='black')

    return plot


def generate_coeff_plot(self):
    """
    Plot posterior distributions for constants and VAR coefficients.
    - Constants: Single grid plot
    - VAR Coefficients: One plot per lag
    """
    if not hasattr(self, 'beta_draws') or len(self.beta_draws) == 0:
        raise ValueError("Posterior draws for beta coefficients are not available.")

    beta_array = np.array(self.beta_draws)  # shape: (n_draws, n_total_coeffs)
    num_draws, num_coeffs = beta_array.shape

    labels = []
    equations = []
    lags_list = []
    
    self.ncoeff_eq = self.n_endo * self.lags + (1 if self.constant else 0)

    for i in range(self.n_endo):  # Equation i
        for lag in range(1, self.lags + 1):
            for j in range(self.n_endo):
                labels.append(f"b_{i+1},{j+1} (lag {lag})")
                lags_list.append(lag)
        if self.constant:
            labels.append(f"Constant {i+1}")
            lags_list.append(0)
        
        equations.extend([self.names[i]] * self.ncoeff_eq)

    if len(labels) != num_coeffs:
        raise ValueError("Mismatch between beta_draws and generated labels.")

    df = pd.DataFrame(beta_array, columns=labels)
    df = df.melt(var_name="Coefficient", value_name="Value")
    df["Equation"] = np.repeat(equations, num_draws)
    df["Lag"] = np.repeat(lags_list, num_draws)

    # Split constants and VAR coefficients
    df_const = df[df["Lag"] == 0]
    df_var = df[df["Lag"] > 0]

    # ------------------------------
    # Constants Plot
    # ------------------------------
    medians_const = df_const.groupby("Coefficient")["Value"].median().reset_index()
    medians_const["label"] = medians_const["Value"].round(2).astype(str)

    n_const = df_const["Coefficient"].nunique()
    ncol_const = min(4, n_const)
    nrow_const = math.ceil(n_const / ncol_const)

    const_plot = (
        ggplot(df_const, aes(x='Value')) +
        geom_density(fill='green', alpha=0.4) +
        geom_vline(data=medians_const, mapping=aes(xintercept='Value'), linetype='dashed', color='red') +
        geom_text(data=medians_const, mapping=aes(x='Value', y=0, label='label'),
                  color='red', va='bottom', ha='left', nudge_y=0.01, size=10) +
        facet_wrap('~Coefficient', nrow=nrow_const, ncol=ncol_const, scales='free') +
        labs(title="Posterior Distributions of Constants") +
        theme(
            figure_size=(ncol_const * 4, nrow_const * 3),
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
            strip_background=element_rect(fill="white"),
            panel_grid_major=element_line(color="grey", linetype="dashed", size=0.5),
            panel_grid_minor=element_blank(),
            strip_text=element_text(size=10),
            axis_line_x=element_line(color="black", size=1),
            axis_line_y=element_line(color="black", size=1),
            legend_position="none"
        )
    )

    # ------------------------------
    # VAR Coefficient Plots by Lag
    # ------------------------------
    var_plots = []
    for lag in range(1, self.lags + 1):
        df_lag = df_var[df_var["Lag"] == lag].copy()

        # Create unique facet label per coefficient → equation
        df_lag["facet_id"] = df_lag["Coefficient"] + " → " + df_lag["Equation"]

        medians_lag = df_lag.groupby("facet_id")["Value"].median().reset_index()
        medians_lag["label"] = medians_lag["Value"].round(2).astype(str)

        var_plot = (
            ggplot(df_lag, aes(x='Value')) +
            geom_density(fill='blue', alpha=0.4) +
            geom_vline(data=medians_lag, mapping=aes(xintercept='Value'), linetype='dashed', color='red') +
            geom_text(data=medians_lag, mapping=aes(x='Value', y=0, label='label'),
                      color='red', va='bottom', ha='left', nudge_y=0.01, size=10) +
            facet_wrap('~facet_id', nrow=self.n_endo, ncol=self.n_endo, scales='free') +
            labs(title=f"Posterior Distributions of VAR Coefficients (Lag {lag})") +
            theme(
                figure_size=(self.n_endo * 4, self.n_endo * 2.5),
                panel_background=element_rect(fill="white"),
                plot_background=element_rect(fill="white"),
                strip_background=element_rect(fill="white"),
                panel_grid_major=element_line(color="grey", linetype="dashed", size=0.5),
                panel_grid_minor=element_blank(),
                strip_text=element_text(size=9),
                axis_line_x=element_line(color="black", size=1),
                axis_line_y=element_line(color="black", size=1),
                legend_position="none"
            )
        )

        var_plots.append(var_plot)

    return const_plot, var_plots



def generate_irf_plots(self, cred_interval: list = [0.68, 0.95]):
    """
    Generate one IRF plot per shock. Each plot shows responses of all variables.
    Supports customizable credible intervals and correct subplot order.
    """
    # Validate and convert credible interval(s)
    if isinstance(cred_interval, (float, int)):
        cred_intervals = [cred_interval]
    else:
        cred_intervals = list(cred_interval)

    irf_array = np.array(self.ir_draws)  # shape: (draws, hor, var, shock)
    H, N = self.hor, self.n_endo
    ir_plots = []

    # Label for type of shock
    shock_type_label = "unit" if self.irf_1std == 0 else "1 s.d."

    for shock in range(N):
        # Collect responses to this shock
        data_list = []

        for var_idx, var_name in enumerate(self.names):
            responses = irf_array[:, :, var_idx, shock]  # shape: (draws, hor)
            for t in range(H):
                row = {
                    "Horizon": t,
                    "Variable": var_name,
                    "Median": np.percentile(responses[:, t], 50),
                }
                for ci in cred_intervals:
                    lower_p = 50 - ci * 50
                    upper_p = 50 + ci * 50
                    lower_val, upper_val = np.percentile(responses[:, t], [lower_p, upper_p])
                    row[f"Lower{int(ci*100)}"] = lower_val
                    row[f"Upper{int(ci*100)}"] = upper_val

                data_list.append(row)

        df = pd.DataFrame(data_list)
        df["Variable"] = pd.Categorical(df["Variable"], categories=self.names, ordered=True)

        # Layout
        ncols = min(2, N)
        nrows = math.ceil(N / ncols)

        # Build plot
        p = (
            ggplot(df, aes(x="Horizon", y="Median")) +
            geom_hline(yintercept=0, linetype="dashed", color="black") +
            facet_wrap("~Variable", ncol=ncols, scales="free") +
            labs(
                title=f"Impulse Responses to a {shock_type_label} shock in {self.names[shock]}",
                y="Response",
                x="Horizon"
            ) +
            theme(
                figure_size=(ncols * 5, nrows * 3),
                strip_text=element_text(size=12, weight='bold'),
                axis_text=element_text(size=10),
                axis_title=element_text(size=11),
                panel_background=element_rect(fill="white"),
                plot_background=element_rect(fill="white"),
                panel_grid_major=element_line(color="grey", linetype="dashed", size=0.3),
                panel_grid_minor=element_line(color="lightgrey", size=0.1),
            )
        )

        # Add credible intervals
        hist_color = "#8AB2D4"
        for ci in sorted(cred_intervals, reverse=True):  # Draw wider intervals first
            p += geom_ribbon(
                aes(
                    ymin=f"Lower{int(ci*100)}",
                    ymax=f"Upper{int(ci*100)}"
                ),
                alpha=0.4 if ci < 0.9 else 0.2,
                fill=hist_color
            )

        # Add median line
        p += geom_line(color="black", size=1.2)

        ir_plots.append(p)

    return ir_plots


def generate_fevd_plot(self, series_titles=None, shock_titles=None, title=None, color_palette=None):
    """
    Plot FEVD as stacked area plot using cumulative bands (like MATLAB's AreaPlot).
    
    - self.fevd: [steps, shocks, variables]
    """
    steps, shocks, variables = self.fevd.shape
    
    # Replace with custom titles if provided
    var_names = self.names

    if shock_titles is None:
        shock_titles = [f"Shock {i+1}" for i in range(shocks)]
    if color_palette is None:
        color_palette = ["#fbb4ae", "#b3cde3", "#ccebc5"][:shocks]  # soft pastel tones

    # Build cumulative contributions (from last shock to first)
    records = []
    for v in range(variables):
        for h in range(steps):
            cum_upper = 100.0
            for s in reversed(range(shocks)):
                contrib = self.fevd[h, s, v]
                cum_lower = cum_upper - contrib
                # Ensure the bottom layer starts at 0
                if s == 0:
                    cum_lower = 0
                records.append({
                    "Horizon": h + 1,
                    "Variable": var_names[v],
                    "Shock": shock_titles[s],
                    "ymin": cum_lower,
                    "ymax": cum_upper
                })
                cum_upper = cum_lower

    df = pd.DataFrame(records)
    df["Shock"] = pd.Categorical(df["Shock"], categories=shock_titles, ordered=True)
    df["Variable"] = pd.Categorical(df["Variable"], categories=var_names, ordered=True)
    if series_titles:
        title_dict = dict(zip(var_names, series_titles))
        df['Variable'] = df['Variable'].map(title_dict)
        df['Variable'] = pd.Categorical(df['Variable'], categories=series_titles, ordered=True)

    # Layout
    ncols = min(2, variables)
    nrows = math.ceil(variables / ncols)
        
    plot = (
        ggplot(df, aes(x="Horizon", ymin="ymin", ymax="ymax", fill="Shock")) +
        geom_ribbon(alpha=0.9) +
        facet_wrap("~Variable", ncol=2, scales="free") +
        scale_fill_manual(values=color_palette) +
        labs(
            title=title or "Forecast Error Variance Decomposition",
            x="Horizon (steps)",
            y="Contribution to FEVD (%)"
        ) +
        scale_y_continuous(limits=[0, 100]) +
        theme(
            figure_size=(ncols * 5, nrows * 3),
            plot_title=element_text(size=14, face="bold") if title else element_blank(),
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
            strip_background=element_rect(fill="white"),
            strip_text=element_text(size=12, weight='bold'),
            panel_grid_major=element_line(color="grey", linetype="dashed", alpha=0.2),
            panel_grid_minor=element_blank(),
            legend_position='bottom',
            legend_direction='horizontal',
            legend_title=element_blank(),
            axis_line_x=element_line(color="black"),
            axis_line_y=element_line(color="black")
        )
    )

    return plot



def generate_hd_plot(self, series_titles=None, shock_titles=None, title=None, color_palette=None):
    """
    Plot Historical Decomposition (HD) as stacked bar chart for shocks + total line (black).
    
    Parameters:
    - self.HD['shock']: array [T x shocks x variables]
    - series_titles: list of variable names
    - shock_titles: list of shock names
    - title: plot title
    - color_palette: list of HEX colors
    
    Returns:
    - plotnine.ggplot object
    """
    hd_array = self.HD['shock']  # shape: [T x S x V]
    T, S, V = hd_array.shape
    dates = pd.to_datetime(self.dates)  # Use last T dates (after lags)

    if series_titles is None:
        series_titles = self.names
    if shock_titles is None:
        shock_titles = [f"Shock {i+1}" for i in range(S)]
    if color_palette is None:
        color_palette = ["#fbb4ae", "#b3cde3", "#ccebc5"][:S]

    # Long-format dataframe
    records = []
    for t in range(T):
        for s in range(S):
            for v in range(V):
                records.append({
                    "Date": dates[t],
                    "Shock": shock_titles[s],
                    "Variable": series_titles[v],
                    "Contribution": hd_array[t, s, v]
                })
    df = pd.DataFrame(records)

    # Cumulative line
    df_sum = df.groupby(['Date', 'Variable'])['Contribution'].sum().reset_index()
    df_sum['Shock'] = 'Total'
    df_sum.rename(columns={'Contribution': 'TotalShockSum'}, inplace=True)

    # Format categories
    df["Variable"] = pd.Categorical(df["Variable"], categories=series_titles, ordered=True)
    df["Shock"] = pd.Categorical(df["Shock"], categories=shock_titles, ordered=True)
    df_sum["Variable"] = pd.Categorical(df_sum["Variable"], categories=series_titles, ordered=True)

    # Layout
    ncols = min(2, V)
    nrows = math.ceil(V / ncols)

    plot = (
        ggplot(df, aes(x="Date", y="Contribution", fill="Shock")) +
        geom_bar(stat="identity", position="stack") +
        geom_line(df_sum, aes(x="Date", y="TotalShockSum", group="Variable"),
                  color="black", size=1.1) +
        facet_wrap("~Variable", ncol=ncols, scales="free") +
        scale_fill_manual(values=color_palette) +
        scale_x_datetime(date_labels="%y", date_breaks="5 years") +
        labs(
            title=title or "Historical Decomposition",
            x="Date",
            y="Shock Contribution"
        ) +
        theme(
            figure_size=(ncols * 5, nrows * 3),
            plot_title=element_text(size=14, face="bold"),
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
            strip_background=element_rect(fill="white"),
            strip_text=element_text(size=12, weight='bold'),
            panel_grid_major=element_line(color="grey", linetype="dashed", alpha=0.2),
            panel_grid_minor=element_blank(),
            axis_line_x=element_line(color="black"),
            axis_line_y=element_line(color="black"),
            legend_position='bottom',
            legend_direction='horizontal',
            legend_title=element_blank(),
            axis_text_x=element_text(size=9)
        )
    )

    return plot



def generate_forecast_plots(self, forecasts: np.ndarray, cred_interval: list = [0.68, 0.95], 
                            last_k: int = None, n_breaks: int = 10, zero_line: bool = False,
                            forecast_type: str = "Unconditional"):
    """
    Plot actual time series and forecast distributions for each variable.

    Parameters:
        forecasts (np.ndarray): Array of shape (n_draws, steps, n_endo)
        cred_interval (list): List of credible intervals to display (e.g., [0.68, 0.95])
        last_k (int): Show only last_k historical observations + forecast. If None, show all history.
        n_breaks (int): Number of x-axis breaks (year ticks).
        zero_line (bool): Whether to include a horizontal zero line.
        forecast_type (str): "Unconditional" or "Conditional"

    Returns:
        ggplot object with forecast + actual data
    """
    n_draws, steps, n_endo = forecasts.shape
    cred_intervals = [cred_interval] if isinstance(cred_interval, (float, int)) else list(cred_interval)

    # === Historical Data ===
    hist_dates = self.yy_dates
    yy = self.yy
    if last_k is not None:
        yy = yy[-last_k:]
        hist_dates = hist_dates[-last_k:]

    hist_df = pd.DataFrame(yy, columns=self.names)
    hist_df["date"] = hist_dates
    hist_df = hist_df.melt(id_vars="date", var_name="Variable", value_name="Value")
    hist_df["Type"] = "Actual"

    # === Forecast Data ===
    forecast_start = hist_dates[-1]
    freq = pd.infer_freq(hist_dates[:5]) or 'Q'
    forecast_dates = pd.date_range(start=forecast_start, periods=steps + 1, freq=freq)[1:]

    data_list = []
    for var_idx, var_name in enumerate(self.names):
        dist = forecasts[:, :, var_idx]
        for t in range(steps):
            row = {
                "date": forecast_dates[t],
                "Variable": var_name,
                "Median": np.percentile(dist[:, t], 50),
            }
            for ci in cred_intervals:
                lower, upper = 50 - ci * 50, 50 + ci * 50
                row[f"Lower{int(ci*100)}"] = np.percentile(dist[:, t], lower)
                row[f"Upper{int(ci*100)}"] = np.percentile(dist[:, t], upper)
            data_list.append(row)

    forecast_df = pd.DataFrame(data_list)
    forecast_df["Type"] = forecast_type

    # === Combine ===
    combined_df = pd.concat([
        hist_df.rename(columns={"Value": "Median"}),
        forecast_df
    ], ignore_index=True)
    combined_df["Variable"] = pd.Categorical(combined_df["Variable"], categories=self.names, ordered=True)
    combined_df = combined_df.sort_values(["Variable", "date"])
    forecast_df["Variable"] = pd.Categorical(forecast_df["Variable"], categories=self.names, ordered=True)

    # === Plot Layout ===
    ncols = min(2, n_endo)
    nrows = math.ceil(n_endo / ncols)

    # === Plot ===
    forecast_color = "#1f77b4" if forecast_type == "Unconditional" else "#d62728"

    p = (
        ggplot(combined_df, aes(x="date", y="Median")) +
        facet_wrap("~Variable", ncol=ncols, scales="free_y") +
        scale_x_datetime(date_labels="%y", breaks=breaks_date(n=n_breaks)) +
        labs(title=f"{forecast_type} Forecast", x="", y="") +
        theme(
            figure_size=(ncols * 5, nrows * 3),
            strip_text=element_text(size=12, weight='bold'),
            axis_text=element_text(size=10),
            axis_title=element_text(size=11),
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
            panel_grid_major=element_line(color="grey", linetype="dashed", size=0.3),
            panel_grid_minor=element_line(color="lightgrey", size=0.1),
        )
    )

    for ci in sorted(cred_intervals, reverse=True):
        p += geom_ribbon(
            data=forecast_df,
            mapping=aes(ymin=f"Lower{int(ci*100)}", ymax=f"Upper{int(ci*100)}"),
            alpha=0.4 if ci < 0.9 else 0.2,
            fill=forecast_color
        )

    p += geom_line(data=combined_df[combined_df["Type"] == "Actual"], color="black", size=1)
    p += geom_line(data=combined_df[combined_df["Type"] == forecast_type], color=forecast_color, size=1.1)

    if zero_line:
        p += geom_hline(yintercept=0, linetype='dashed', color='black')

    return p