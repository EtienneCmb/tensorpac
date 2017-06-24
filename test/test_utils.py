"""Test tensorpac utils."""
import numpy as np
from tensorpac.utils import PacVec, PacSignals, PacTriVec


def test_pacvec():
    """Definition of PAC vectors."""
    assert PacVec()
    assert PacVec(fpha=(1, 30, 2, 2), famp=(60, 200, 10, 5))
    assert PacVec(fpha=[1, 2], famp=np.arange(50))
    assert PacVec(fpha=np.array([[2, 4], [5, 7], [9, 10]]),
                  famp=np.array([[30, 60], [60, 90], [100, 200]]).T)
    assert PacVec(fpha=[[1, 2], [5, 7]], famp=[60, 150])


def test_pacsignals_dTrials():
    """Definition of artificially coupled signals using dPha/dAmp."""
    assert PacSignals(fpha=5, famp=130, sf=512, ndatasets=23, chi=0.9, noise=2,
                      dpha=35, damp=46)


def test_pacsignals_bandwidth():
    """Definition of artificially coupled signals using bandwidth."""
    assert PacSignals(fpha=[5, 7], famp=[30, 60], sf=200., ndatasets=100,
                      chi=0.5, noise=3., npts=1000)


def test_defaultArgs():
    """Test default aurguments for PacVec."""
    assert PacSignals(chi=2., noise=11., dpha=120., damp=200.)


def test_trivec():
    """Definition of triangular vectors."""
    assert PacTriVec(2, 200, 10)
