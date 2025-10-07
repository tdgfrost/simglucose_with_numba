# from .noise_gen import CGMNoiseGenerator
from .noise_gen import CGMNoise
import pandas as pd
import logging
import pkg_resources
import numba as nb
from numba.experimental import jitclass

logger = logging.getLogger(__name__)
SENSOR_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/sensor_params.csv')


cgm_params_spec = [
    ('Name', nb.types.unicode_type),
    ('PACF', nb.float64),
    ('gamma', nb.float64),
    ('lambda_', nb.float64),
    ('delta', nb.float64),
    ('xi', nb.float64),
    ('sample_time', nb.float64),
    ('min_', nb.float64),
    ('max_', nb.float64),
]

# A complete jitclass using the spec above
@jitclass(cgm_params_spec)
class CGMParams:
    def __init__(self, Name, PACF, gamma, lambda_, delta, xi,
                 sample_time, min_, max_):
        self.Name = Name
        self.PACF = PACF
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        self.xi = xi
        self.sample_time = sample_time
        self.min_ = min_
        self.max_ = max_


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
        self._numba_params = CGMParams(
            self._pandas_params['Name'],
            self._pandas_params['PACF'],
            self._pandas_params['gamma'],
            self._pandas_params['lambda'], # Read from 'lambda'
            self._pandas_params['delta'],
            self._pandas_params['xi'],
            self._pandas_params['sample_time'],
            self._pandas_params['min'],    # Read from 'min'
            self._pandas_params['max']     # Read from 'max'
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


class CGMSensor(object):
    def __init__(self, params, seed=None):
        self._params = ParamManager(params)
        self.name = params.Name
        self.sample_time = params.sample_time
        self.seed = seed
        self._last_CGM = 0

    @classmethod
    def withName(cls, name, **kwargs):
        sensor_params = pd.read_csv(SENSOR_PARA_FILE)
        params = sensor_params.loc[sensor_params.Name == name].squeeze()
        return cls(params, **kwargs)

    def measure(self, patient):
        if patient.t % self.sample_time == 0:
            BG = patient.observation.Gsub
            CGM = BG + next(self._noise_generator)
            CGM = max(CGM, self._params._numba_params.min_)
            CGM = min(CGM, self._params._numba_params.max_)
            self._last_CGM = CGM
            return CGM

        # Zero-Order Hold
        return self._last_CGM

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._noise_generator = CGMNoise(self._params, seed=seed)

    def reset(self):
        logger.debug('Resetting CGM sensor ...')
        self._noise_generator = CGMNoise(self._params, seed=self.seed)
        self._last_CGM = 0


if __name__ == '__main__':
    pass
