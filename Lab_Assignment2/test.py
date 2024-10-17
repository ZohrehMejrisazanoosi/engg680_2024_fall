import numpy as np

def fit_polynomial_curve(a: np.ndarray, l: np.ndarray, P: np.ndarray, deg: int) -> tuple:
    '''
    Args:
        a:              function input (constants, independent variable values)
        l:              noisy observations (dependent variable values)
        P:              observation weight matrix (weights for observations)
        deg:            degree of polynomial to fit to data
    Returns:
        x:              estimated polynomial coefficients
        cov_x:          uncertainty in estimated coefficients
        apv:            a posteriori variance factor (goodness of fit)
        v:              observation residuals
    '''

    # Step 1: Construct the design matrix A (Vandermonde matrix)
    A = np.vander(a, deg+1, increasing=True)  # Construct Vandermonde matrix for polynomial fitting
    print('A :\n',A)
    # Step 2: Apply the weighted least squares estimation formula
    # Normal equation: x = (A^T P A)^{-1} A^T P l
    AT = A.T                           # Transpose of the design matrix A
    AT_P = np.dot(AT, P)               # A^T * P
    N = np.dot(AT_P, A)                # N = A^T * P * A (normal matrix)
    x = np.linalg.inv(N) @ AT_P @ l    # Estimated coefficients using the normal equation

    # Step 3: Calculate the observation residuals
    v = A @ x - l  # Residuals (difference between observed and fitted values)

    # Step 4: Calculate the covariance matrix of the estimated coefficients
    cov_x = np.linalg.inv(N)  # Covariance of the coefficients

    # Step 5: Compute the a posteriori variance factor (apv)
    # apv = (v^T P v) / (n - m), where n is the number of observations and m is the number of parameters
    n = len(l)                   # Number of observations
    m = deg + 1                  # Number of parameters (degree of polynomial + 1)
    apv = (v.T @ P @ v) / (n - m)  # a posteriori variance factor

    return (x, cov_x, apv, v)

# Example usage
a = np.array([1, 2, 3, 4, 5])      # Independent variable values
l = np.array([1.1, 2.0, 2.9, 4.1, 5.0])  # Noisy observations (dependent variable values)
P = np.eye(len(a))                 # Weight matrix (assuming equal weights here)
deg = 2                            # Degree of polynomial to fit

x, cov_x, apv, v = fit_polynomial_curve(a, l, P, deg)
