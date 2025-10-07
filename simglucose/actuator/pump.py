import pandas as pd
import pkg_resources
import logging
import numpy as np
from numba import njit
import numba as nb
from numba.experimental import jitclass

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')
logger = logging.getLogger(__name__)

params_spec = [
    ('Name', nb.types.unicode_type),
    ('min_bolus', nb.float64),
    ('max_bolus', nb.float64),
    ('inc_bolus', nb.float64),
    ('min_basal', nb.float64),
    ('max_basal', nb.float64),
    ('inc_basal', nb.float64),
    ('sample_time', nb.float64),
]

# A complete jitclass using the spec above
@jitclass(params_spec)
class PumpParams:
    def __init__(self, Name, min_bolus, max_bolus, inc_bolus,
                 min_basal, max_basal, inc_basal, sample_time):
        self.Name = Name
        self.min_bolus = min_bolus
        self.max_bolus = max_bolus
        self.inc_bolus = inc_bolus
        self.min_basal = min_basal
        self.max_basal = max_basal
        self.inc_basal = inc_basal
        self.sample_time = sample_time


def params_to_numba(params):
    return PumpParams(params.Name, params.min_bolus, params.max_bolus,
                      params.inc_bolus, params.min_basal, params.max_basal,
                      params.inc_basal, params.sample_time)


class ParamManager:
    """
    A wrapper that holds both pandas and Numba parameter versions.

    It delegates attribute access to the pandas Series by default,
    making it compatible with libraries like simglucose. The optimized
    Numba version is available via the `.numba` attribute for use in
    performance-critical functions.
    """

    def __init__(self, pandas_params_series):
        # Store the original pandas series
        self._pandas_params = pandas_params_series

        # Create and store the Numba jitclass instance immediately
        self._numba_params = PumpParams(
            *[pandas_params_series[name] for name, _ in params_spec]
        )

    def __getattr__(self, name):
        """
        This is the magic part. If an attribute isn't found on this
        object, it automatically looks for it on the pandas Series.
        """
        return getattr(self._pandas_params, name)

    def __getitem__(self, key):
        """
        Delegates bracket access (e.g., params["BW"]).
        This is the new method that fixes the error.
        """
        return self._pandas_params[key]


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
        self._params = ParamManager(params)

    @classmethod
    def withName(cls, name):
        pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
        params = pump_params.loc[pump_params.Name == name].squeeze()
        return cls(params)

    def bolus(self, amount):
        if not isinstance(amount, (float, int)):
            amount = amount[0]
        # JITTED
        return jitted_bolus(amount, self.U2PMOL,
                            self._params._numba_params.inc_bolus,
                            self._params._numba_params.min_bolus,
                            self._params._numba_params.max_bolus)

    def basal(self, amount):
        if not isinstance(amount, (float, int)):
            amount = amount[0]
        # JITTED
        return jitted_basal(amount, self.U2PMOL,
                            self._params._numba_params.inc_basal,
                            self._params._numba_params.min_basal,
                            self._params._numba_params.max_basal)

    def reset(self):
        logger.info('Resetting insulin pump ...')
        pass
