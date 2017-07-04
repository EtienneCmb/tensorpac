"""Normalize PAC by surrogates methods.

This file include the following methods :
- No normalization
- Substraction : substract the mean of surrogates
- Divide : divide by the mean of surrogates
- Substract then divide : substract then divide by the mean of surrogates
- Z-score : substract the mean and divide by the deviation of the
            surrogates
"""

__all__ = ('normalize')


def normalize(pac, s_mean, s_std, idn):
    """List of the normalization methods.

    Use a normalization to normalize the true cfc value by the surrogates.
    Here's the list of the normalization methods :
    - No normalization
    - Substraction : substract the mean of surrogates
    - Divide : divide by the mean of surrogates
    - Substract then divide : substract then divide by the mean of surrogates
    - Z-score : substract the mean and divide by the deviation of the
                surrogates

    The normalized method only return the normalized cfc.
    """
    if idn == 0:  # No normalisation
        return pac

    elif idn == 1:  # Substraction
        pac -= s_mean

    elif idn == 2:  # Divide
        pac /= s_mean

    elif idn == 3:  # Substract then divide
        pac -= s_mean
        pac /= s_mean

    elif idn == 4:  # Z-score
        pac -= s_mean
        pac /= s_std
