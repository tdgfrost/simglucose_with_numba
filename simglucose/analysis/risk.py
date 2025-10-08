import numpy as np
from numba import njit


def risk_index(BG, horizon):
    # BG is in mg/dL
    BG_to_compute = BG[-horizon:]
    risks =[risk(r) for r in BG_to_compute]
    LBGI = np.mean([r[0] for r in risks])
    HBGI = np.mean([r[1] for r in risks])
    RI = np.mean([r[2] for r in risks])

    return (LBGI, HBGI, RI)


@njit
def risk(BG, w_high=2.0, p_high=2.0):
    """
    Risk is a percentage - ranging from 0 to 100%.
    The 20 and 600 mg/dl are just the values to which the risk formula was fit. 
    The aim is to make the risk maximum when it is either 20 or 600.
    The units in the paper below are different (mmol/l), but in our units (mg/dl) these limits are 20 and 600.

    Reference, in particular see appendix for the derivation of risk:
    https://diabetesjournals.org/care/article/20/11/1655/21162/Symmetrization-of-the-Blood-Glucose-Measurement

    """
    MIN_BG = 20.0
    MAX_BG = 600.0
    if BG <= MIN_BG: 
        return (100.0, 0.0, 100.0)
    if BG >= MAX_BG:
        return (0.0, 100.0, 100.0)
    
    U = 1.509 * (np.log(BG)**1.084 - 5.381)

    # Keep hypo side unchanged
    if U < 0:
        rl = 10.0 * (np.absolute(U) ** 2.0)
        rh = 0.0
    else:
        # Make hyper stricter: weight and/or increase exponent
        rh = 10.0 * (w_high * np.absolute(U))** p_high
        rl = 0.0

    ri = rl + rh

    # manual scalar clamp to [0, 100]
    if rl < 0.0: rl = 0.0
    elif rl > 100.0: rl = 100.0
    if rh < 0.0: rh = 0.0
    elif rh > 100.0: rh = 100.0
    if ri < 0.0: ri = 0.0
    elif ri > 100.0: ri = 100.0

    return rl, rh, ri

