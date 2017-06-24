"""Main surrogates estimation methods.

This file include the following methods :
- No surrogates
- Swap phase/amplitude across trials
- Swap amplitude blocks across time
- Shuffle amplitude and phase time-series
- Shuffle phase time-series
- Shuffle amplitude time-series
- Time lag
"""

import numpy as np
from joblib import Parallel, delayed
from .methods import ComputePac

__all__ = ['ComputeSurogates']


def ComputeSurogates(pha, amp, surargs, pacargs, nperm, njobs):
    """Compute surrogates using tensors and parallel computing.

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        suragrs: tuple
            Tuple containing the arguments to pass to the suroSwitch function.

        pacargs: tuple
            Tuple containing the arguments to pass to the ComputePac function.

        nperm: int
            Number of permutations.

        njobs: int
            Number of jos for the parallel computing.

    Returns:
        suro: np.ndarray
            Array of pac surrogates of shape (nperm, npha, namp, ..., npts)
    """
    s = Parallel(n_jobs=njobs)(delayed(_computeSur)(
                            pha, amp, surargs, pacargs) for k in range(nperm))
    return np.array(s)


def _computeSur(pha, amp, surargs, pacargs):
    """Compute surrogates.

    This is clearly not the optimal implementation. Indeed, for each loop the
    suroSwicth and ComputePac have a several "if" that slow down the execution,
    at least a little bit. And, it's not esthetic but joblib doesn't accept
    to pickle functions.

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        suragrs: tuple
            Tuple containing the arguments to pass to the suroSwitch function.

        pacargs: tuple
            Tuple containing the arguments to pass to the ComputePac function.
    """
    # Get the surrogates :
    pha, amp = suroSwitch(pha, amp, *surargs)
    # Compute PAC on surrogates :
    return ComputePac(pha, amp, *pacargs)


def suroSwitch(pha, amp, idn, axis, traxis, nblocks):
    """List of methods to compute surrogates.

    The surrogates are used to normalized the cfc value. It help to determine
    if the cfc is reliable or not. Usually, the surrogates used the same cfc
    method on surrogates data.
    Here's the list of methods to compute surrogates:
    - No surrogates
    - Swap phase/amplitude across trials
    - Swap amplitude blocks across time.
    - Shuffle amplitude time-series
    - Time lag
    """
    # No surrogates
    if idn == 0:
        return None

    # Swap phase/amplitude across trials :
    elif idn == 1:
        return SwapPhaAmp(pha, amp, traxis)

    # Swap amplitude :
    elif idn == 2:
        return SwapBlocks(pha, amp, axis, nblocks)

    # Shuffle amplitude values
    elif idn == 3:
        return ShuffleAmp(pha, amp, axis)

    # Introduce a time lag
    elif idn == 4:
        return TimeLag(pha, amp, axis)

    else:
        raise ValueError(str(idn) + " is not recognized as a valid surrogates"
                         " evaluation method.")


###############################################################################
###############################################################################
#                            SWAPING
###############################################################################
###############################################################################


def SwapPhaAmp(pha, amp, axis):
    """Swap phase/amplitude trials (Tort, 2010).

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        axis: int
            Location of the trial axis.

    Return:
        pha: np.ndarray
            Swapped version of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Swapped version of amplitudes of shapes (namp, ..., npts)
    """
    return _dimswap(pha, axis), _dimswap(amp, axis)


def SwapBlocks(pha, amp, axis, nblocks):
    """Swap amplitudes time blocks.

    To reproduce (Bahramisharif, 2013), use a number of blocks of 2.

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        axis: int
            Location of the time axis.

        nblocks: int
            Number of blocks to in which the amplitude is splitted.

    Return:
        pha: np.ndarray
            Original version of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Swapped version of amplitudes of shapes (namp, ..., npts)
    """
    # Split amplitude across time into two parts :
    ampl = np.array_split(amp, nblocks, axis=axis)
    # Revered elements :
    ampl.reverse()
    return pha, np.concatenate(ampl, axis=axis)


###############################################################################
###############################################################################
#                            SHUFFLING
###############################################################################
###############################################################################


def ShuffleAmp(pha, amp, axis):
    """Randomly shuffle amplitudes across time.

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        axis: int
            Location of the time axis.

    Return:
        pha: np.ndarray
            Original version of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Shuffled version of amplitudes of shapes (namp, ..., npts)
    """
    return pha, _dimswap(amp, axis)


def TimeLag(pha, amp, axis):
    """Introduce a time lag on phase series..

    Args:
        pha: np.ndarray
            Array of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Array of amplitudes of shapes (namp, ..., npts)

        axis: int
            Location of the time axis.

    Return:
        pha: np.ndarray
            Shiffted version of phases of shapes (npha, ..., npts)

        amp: np.ndarra
            Original version of amplitudes of shapes (namp, ..., npts)
    """
    npts = pha.shape[-1]
    return np.roll(pha, np.random.randint(npts), axis=axis), amp


def _dimswap(x, axis=0):
    """Swap values into an array at a specific axis.

    Args:
        x: np.ndarray
            Array of data to swap

    Kargs:
        axis: int, optional, (def: 0)
            Axis along which to perform swapping.

    Returns:
        x: np.ndarray
            Swapped version of x.
    """
    # Dimension vector :
    dimvec = [slice(None)] * x.ndim
    # Random integer vector :
    rndvec = np.arange(x.shape[axis])
    np.random.shuffle(rndvec)
    dimvec[axis] = rndvec
    # Return a swapped version of x :
    return x[dimvec]
