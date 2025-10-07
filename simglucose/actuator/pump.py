import pandas as pd
import pkg_resources
import logging
import numpy as np
from numba import njit

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')
logger = logging.getLogger(__name__)


@njit
def jitted_basal(amount, U2PMOL, inc_basal, min_basal, max_basal):
    bas = amount * U2PMOL  # convert from U/min to pmol/min
    bas = np.round(bas / inc_basal
                   ) * inc_basal
    bas = bas / U2PMOL  # convert from pmol/min to U/min
    bas = min(bas, max_basal)
    bas = max(bas, min_basal)
    return bas


@njit
def jitted_bolus(amount, U2PMOL, inc_bolus, min_bolus, max_bolus):
    bol = amount * U2PMOL  # convert from U/min to pmol/min
    bol = np.round(bol / inc_bolus
                   ) * inc_bolus
    bol = bol / U2PMOL     # convert from pmol/min to U/min
    bol = min(bol, max_bolus)
    bol = max(bol, min_bolus)
    return bol


class InsulinPump(object):
    U2PMOL = 6000

    def __init__(self, params):
        self._params = params

    @classmethod
    def withName(cls, name):
        pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
        params = pump_params.loc[pump_params.Name == name].squeeze()
        return cls(params)

    def bolus(self, amount):
        if not isinstance(amount, (float, int)):
            amount = amount[0]
        inc_bolus = self._params['inc_bolus']
        min_bolus = self._params['min_bolus']
        max_bolus = self._params['max_bolus']
        U2PMOL = self.U2PMOL
        # JITTED
        return jitted_bolus(amount, U2PMOL, inc_bolus, min_bolus, max_bolus)

    def basal(self, amount):
        if not isinstance(amount, (float, int)):
            amount = amount[0]
        inc_basal = self._params['inc_basal']
        min_basal = self._params['min_basal']
        max_basal = self._params['max_basal']
        U2PMOL = self.U2PMOL
        # JITTED
        return jitted_basal(amount, U2PMOL, inc_basal, min_basal, max_basal)

    def reset(self):
        logger.info('Resetting insulin pump ...')
        pass
