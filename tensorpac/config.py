"""Tensorpac configuration."""

# Gaussian-Copula configuration
MI_BIASCORRECT = False
MI_DEMEAN = False

# Joblib config
JOBLIB_CFG = dict()  # prefer='threads'

# MNE config
try:
    import mne
    MNE_EPOCHS_TYPE = [mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs]
except:
    MNE_EPOCHS_TYPE = []

# centralize configurations
CONFIG = dict(MI_BIASCORRECT=MI_BIASCORRECT, MI_DEMEAN=MI_DEMEAN,
              JOBLIB_CFG=JOBLIB_CFG, MNE_EPOCHS_TYPE=MNE_EPOCHS_TYPE)
