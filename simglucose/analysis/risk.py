import numpy as np
from numba import njit


def risk_index(BG, horizon):
    # BG is in mg/dL
    BG_to_compute = BG[-horizon:]
    risks =[risk(r)[-1] for r in BG_to_compute]
    RI = np.mean([r for r in risks])

    return 0, 0, RI


@njit
def risk(BG):
    """
    Reference, in particular see appendix for the derivation of risk:
    https://diabetesjournals.org/care/article/20/11/1655/21162/Symmetrization-of-the-Blood-Glucose-Measurement

    """
    
    U = 1.509 * (np.log(BG)**1.084 - 5.381) ** 2

    return 0, 0, U
