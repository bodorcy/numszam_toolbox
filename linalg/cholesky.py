import numpy as np
from numpy import ndarray, transpose
from numpy.linalg import eigh

def check_pos_def(A :ndarray):
    if not np.all(A == transpose(A)):
        raise ValueError("Az A mátrix nem szimmetrikus!")
    if not np.all(eigh(A) > 0): # eigh, mert szimmetrikus
        raise ValueError("Nem minden sajátréték > 0.")


def chol(A :ndarray, verbose=True):
    check_pos_def(A)

