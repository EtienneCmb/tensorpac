import numpy as np

from tensorpac.utils import PacVec, PacSignals
from tensorpac import Pac

def test_pacvec():
    """Test the definition of PAC vectors."""
    assert PacVec()
    assert PacVec(pha=(1, 30, 2, 2), amp=(60, 200, 10, 5))
    assert PacVec(pha=[1, 2], amp=np.arange(50))
    assert PacVec(pha=[[1, 2], [5, 7]], amp=[60, 150])

def test_pacsignals():
    """Test the definitino of artificially coupled signals."""
    assert PacSignals(fpha=5, famp=130, sf=512, ndatasets=23, tmax=3, chi=0.9,
                      noise=2, dpha=35, damp=46)

def test_pacdefinition():
    """Test the definition of pac inputs."""
    pass