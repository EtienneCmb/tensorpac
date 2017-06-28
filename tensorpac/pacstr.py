"""Simply get the name of defined methods."""

__all__ = ['pacstr']


def pacstr(idpac):
    """Return correspond methods string."""
    # Pac methods :
    if idpac[0] == 1:
        method = 'Mean Vector Length (MVL, Canolty, 2006)'
    elif idpac[0] == 2:
        method = 'Kullback-Leiber Distance (KLD, Tort, 2010)'
    elif idpac[0] == 3:
        method = 'Heights ratio (HR, Lakatos, 2005)'
    elif idpac[0] == 4:
        method = 'ndPac (Ozkürt, 2012)'
    elif idpac[0] == 5:
        method = 'Phase-Synchrony (Cohen, 2008; Penny, 2008)'
    else:
        raise ValueError("No corresponding pac method.")

    # Surrogate method :
    if idpac[1] == 0:
        suro = 'No surrogates'
    elif idpac[1] == 1:
        suro = 'Swap phase/amplitude across trials'
    elif idpac[1] == 2:
        suro = 'Swap amplitude blocks across time'
    elif idpac[1] == 3:
        suro = 'Shuffle amplitude time-series'
    elif idpac[1] == 4:
        suro = 'Time lag'
    else:
        raise ValueError("No corresponding surrogate method.")

    # Normalization methods :
    if idpac[2] == 0:
        norm = 'No normalization'
    elif idpac[2] == 1:
        norm = 'Substract the mean of surrogates'
    elif idpac[2] == 2:
        norm = 'Divide by the mean of surrogates'
    elif idpac[2] == 3:
        norm = 'Substract then divide by the mean of surrogates'
    elif idpac[2] == 4:
        norm = "Substract the mean and divide by the deviation of the " + \
               "surrogates"
    else:
        raise ValueError("No corresponding normalization method.")

    return method, suro, norm
