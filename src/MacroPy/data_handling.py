import numpy as np

def prepare_data(YY, lags, constant=True, timetrend=False):
    """
    Organizes the data in the form of Y = XB + E.
    
    Parameters:
        YY (numpy.ndarray): The input time series data.
        lags (int): Number of lags.
        constant (bool): Whether to include a constant column.
        timetrend (bool): Whether to include a time trend column.
    
    Returns:
        YYact (numpy.ndarray): The dependent variable matrix.
        XXact (numpy.ndarray): The independent variable matrix including lagged values and optional constants/trends.
    """
    
    T0 = lags  # Pre-sample size
    nv = YY.shape[1]  # Number of variables
    nobs = YY.shape[0] - T0  # Number of observations
    
    # Actual observations
    YYact = YY[T0:T0 + nobs, :]
    XXact = np.zeros((nobs, nv * lags))
    
    for i in range(1, lags + 1):
        XXact[:, (i - 1) * nv:i * nv] = YY[T0 - i:T0 + nobs - i, :]
    
    if timetrend:
        XXact = np.hstack((np.arange(1, nobs + 1).reshape(-1, 1), XXact))
        
    if constant:
        XXact = np.hstack((np.ones((nobs, 1)), XXact))
    
    return YYact, XXact


def estimate_ols(yy, XX):
    """Estimate OLS coefficients."""
    
    # Parameters
    T, _ = yy.shape
    K = XX.shape[1]
    
    # OLS Estimation 
    XtX = XX.T @ XX
    XtY = XX.T @ yy
    B_OLS = np.linalg.inv(XtX) @ XtY  
    b_ols = B_OLS.flatten(order='F')
    residuals = yy - XX @ B_OLS 
    Sigma_ols = (residuals.T @ residuals) / (T - K)
    
    return b_ols, Sigma_ols
