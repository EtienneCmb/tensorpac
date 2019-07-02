"""Tensorpac configuration."""
# JOBLIB_CFG = dict(prefer='threads')
JOBLIB_CFG = dict()

try:
    import mne
    MNE_EPOCHS_TYPE = [mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs]
except:
    MNE_EPOCHS_TYPE = []
