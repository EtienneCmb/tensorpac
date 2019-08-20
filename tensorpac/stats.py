"""Statistics for cross frequency coupling."""
import logging

import numpy as np

from tensorpac.io import is_pandas_installed, is_statsmodels_installed

logger = logging.getLogger('tensorpac')


def test_stationarity(x, p=.05):
    """Test the stationarity of an electrophysiological dataset.

    This function performs a Augmented Dickey-Fuller test and returns a table
    (dataframe) with statistical properties for each epoch.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs, n_times)
    p : float | 0.05
        P-value to use as a threshold in order to infer if the time-series
        are significantly stationary

    Returns
    -------
    df : pandas.DataFrame
        Dataframe that contains the statistical properties for each
        epoch. The table contains the followong columns :

            * Epochs : epoch number
            * P-values
            * Stationary : boolean indicating if the time-serie is
              significantly stationary at the critical level p
            * Statistics : statistical test
            * CV (5%) and CV (1%) : critical value respectively at 5% and 1%

    Notes
    -----
    This function requires both pandas and statsmodels Python packages. See :

        * Pandas : https://pandas.pydata.org/pandas-docs/stable/install.html
        * Statsmodels : http://www.statsmodels.org/stable/install.html
    """
    is_pandas_installed()
    is_statsmodels_installed()
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
    x = np.atleast_2d(x)
    assert x.ndim == 2, ("x should be a two dimentional array of shape "
                         "(n_epochs, n_times)")
    n_epochs, n_times = x.shape
    # test time series
    logger.info(f"Performing a Augmented Dickey-Fuller test on {n_epochs} "
                f"epochs with p={p}")
    epochs = ['epoch %i' % k for k in range(n_epochs)]
    pvalues = np.zeros((n_epochs,), dtype=float)
    adf_stat = np.zeros_like(pvalues)
    stationary = np.zeros_like(pvalues, dtype=bool)
    cv_5, cv_1 = np.zeros_like(pvalues), np.zeros_like(pvalues)
    for k in range(n_epochs):
        result = adfuller(x[k, :])
        adf_stat[k], pvalues[k] = result[0], result[1]
        cv_5[k], cv_1[k] = result[4]['5%'], result[4]['1%']
        stationary[k] = result[1] <= p
    # pandas formating
    cols = ['Epochs', 'P-values', f'Stationary', 'Statistics',
            'CV (5%)', 'CV (1%)']
    df = pd.DataFrame({cols[0]: epochs, cols[1]: pvalues, cols[2]: stationary,
                       cols[3]: adf_stat, cols[4]: cv_5, cols[5]: cv_1},
                      columns=cols)
    n_signi = stationary.sum()
    # n_not_signi = (~stationary.sum())
    logger.info(f"    {n_signi}/{n_epochs} epochs were found as significantly "
                f"stationary at p={p}")
    return df
