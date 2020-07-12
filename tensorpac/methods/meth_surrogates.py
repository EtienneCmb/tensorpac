"""Individual methods for assessing surrogates."""
import numpy as np
from joblib import Parallel, delayed

from tensorpac.config import CONFIG


def compute_surrogates(pha, amp, ids, fcn, n_perm, n_jobs, random_state):
    """Compute surrogates using tensors and parallel computing."""
    if ids == 0:
        return None
    else:
        fcn_p = {1: swap_pha_amp, 2: swap_blocks, 3: time_lag}[ids]

    def para_surr(rnd):  # noqa
        return fcn(*fcn_p(pha, amp, random_state=random_state + rnd))
    s = Parallel(n_jobs=n_jobs, **CONFIG['JOBLIB_CFG'])(delayed(para_surr)(
        k) for k in range(n_perm))
    return np.array(s)


def swap_pha_amp(pha, amp, random_state=None):
    """Compute surrogates by swapping phase / amplitude trials.

    This function destroys the relation between the phase and the amplitude in
    order to correct for PAC that could be obtained by chance. To this end,
    this function reorganize the relation from trial-to-trial between the phase
    and the amplitude (swapping). In particular, in that case it is the trials
    of the phase that are randomly swapped and the amplitude is leaved
    unchanged. For a more detailed description, see :cite:`tort2010measuring`

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape
        (n_pha, n_trials, ..., n_times) and the array of amplitudes of shape
        (n_amp, n_trials, ..., n_times).
    random_state : int | None
        Fix the random state of the machine for reproducible results.

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Tort et al. 2010 :cite:`tort2010measuring`
    """
    if random_state is None:
        random_state = int(np.random.randint(0, 10000, size=1))
    rnd = np.random.RandomState(random_state)
    tr_ = rnd.permutation(pha.shape[1])
    return pha[:, tr_, ...], amp


def swap_blocks(pha, amp, random_state=None):
    """Compute surrogates by swapping amplitudes time blocks.

    This function destroys the relation between the phase and the amplitude in
    order to correct for PAC that could be obtained by chance. To this end,
    this function cut the amplitude at a random time point. Then, both time
    blocks are swapped. For a more detailed description, see
    :cite:`bahramisharif2013propagating`.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    random_state : int | None
        Fix the random state of the machine for reproducible results.

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Bahramisharif et al. 2013 :cite:`bahramisharif2013propagating`
    """
    if random_state is None:
        random_state = int(np.random.randint(0, 10000, size=1))
    rnd = np.random.RandomState(random_state)
    # random cutting point along time axis
    cut_at = rnd.randint(1, amp.shape[-1], (1,))
    # Split amplitude across time into two parts :
    ampl = np.array_split(amp, cut_at, axis=-1)
    # Revered elements :
    ampl.reverse()
    return pha, np.concatenate(ampl, axis=-1)


def time_lag(pha, amp, random_state=None):
    """Compute surrogates by introducing a time lag on phase series.

    This function destroys the relation between the phase and the amplitude in
    order to correct for PAC that could be obtained by chance. To this end, a
    random temporal lag is introduced at the beginning of the phase. For a more
    detailed description, see :cite:`canolty2006high`.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    random_state : int | None
        Fix the random state of the machine for reproducible results.

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Canolty et al. 2006 :cite:`canolty2006high`
    """
    if random_state is None:
        random_state = int(np.random.randint(0, 10000, size=1))
    rnd = np.random.RandomState(random_state)
    shift = rnd.randint(pha.shape[-1])
    return np.roll(pha, shift, axis=-1), amp


def normalize(idn, pac, surro):
    """Normalize the phase amplitude coupling.

    This function performs inplace normalization (i.e. without copy of array)

    Parameters
    ----------
    idn : int
        Normalization method to use :

            * 1 : substract the mean of surrogates
            * 2 : divide by the mean of surrogates
            * 3 : substract then divide by the mean of surrogates
            * 4 : substract the mean then divide by the deviation of surrogates
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)
    surro : array_like
        Array of surrogates of shape (n_perm, n_amp, n_pha, ...)
    """
    s_mean, s_std = np.mean(surro, axis=0), np.std(surro, axis=0)
    if idn == 1:  # Substraction
        pac -= s_mean
    elif idn == 2:  # Divide
        pac /= s_mean
    elif idn == 3:  # Substract then divide
        pac -= s_mean
        pac /= s_mean
    elif idn == 4:  # Z-score
        pac -= s_mean
        pac /= s_std
