"""Tensorpac configuration."""

# _____________________________ MUTUAL-INFORMATION ____________________________
# Gaussian-Copula configuration
MI_BIASCORRECT = False
MI_DEMEAN = False

# ___________________________________ JOBLIB __________________________________
# Joblib config
JOBLIB_CFG = dict()  # prefer='threads'

# ___________________________________ STATS ___________________________________
MIN_SHIFT = 1
MAX_SHIFT = None

# _____________________________________ MNE ___________________________________
# MNE config
try:
    import mne
    MNE_EPOCHS_TYPE = [mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs]
except:
    MNE_EPOCHS_TYPE = []

# centralize configurations
CONFIG = dict(MI_BIASCORRECT=MI_BIASCORRECT, MI_DEMEAN=MI_DEMEAN,
              JOBLIB_CFG=JOBLIB_CFG, MNE_EPOCHS_TYPE=MNE_EPOCHS_TYPE,
              MIN_SHIFT=MIN_SHIFT, MAX_SHIFT=MAX_SHIFT)
