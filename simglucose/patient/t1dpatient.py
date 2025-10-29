from .base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
import numba as nb
from numba import njit
from numba.experimental import jitclass

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)

params_spec = [
    ('BW', nb.float64),
    ('u2ss', nb.float64),
    ('kmax', nb.float64),
    ('b', nb.float64),
    ('d', nb.float64),
    ('f', nb.float64),
    ('kmin', nb.float64),
    ('kabs', nb.float64),
    ('kp1', nb.float64),
    ('kp2', nb.float64),
    ('kp3', nb.float64),
    ('ke1', nb.float64),
    ('ke2', nb.int64),      # Mapped to int64
    ('k1', nb.float64),
    ('k2', nb.float64),
    ('Vm0', nb.float64),
    ('Vmx', nb.float64),
    ('Km0', nb.float64),
    ('m1', nb.float64),
    ('m2', nb.float64),
    ('m4', nb.float64),
    ('ka1', nb.float64),
    ('ka2', nb.float64),
    ('Vi', nb.float64),
    ('p2u', nb.float64),
    ('Ib', nb.float64),
    ('ki', nb.float64),
    ('m30', nb.float64),
    ('kd', nb.float64),
    ('ksc', nb.float64),
    ('Fsnc', nb.int64),     # Mapped to int64
]

# A complete, copy-pasteable jitclass using the spec above
@jitclass(params_spec)
class NumbaParams(object):
    def __init__(self, BW, u2ss, kmax, b, d, f, kmin, kabs, kp1, kp2, kp3,
                 ke1, ke2, k1, k2, Vm0, Vmx, Km0, m1, m2, m4, ka1, ka2,
                 Vi, p2u, Ib, ki, m30, kd, ksc, Fsnc):
        self.BW = BW
        self.u2ss = u2ss
        self.kmax = kmax
        self.b = b
        self.d = d
        self.f = f
        self.kmin = kmin
        self.kabs = kabs
        self.kp1 = kp1
        self.kp2 = kp2
        self.kp3 = kp3
        self.ke1 = ke1
        self.ke2 = ke2
        self.k1 = k1
        self.k2 = k2
        self.Vm0 = Vm0
        self.Vmx = Vmx
        self.Km0 = Km0
        self.m1 = m1
        self.m2 = m2
        self.m4 = m4
        self.ka1 = ka1
        self.ka2 = ka2
        self.Vi = Vi
        self.p2u = p2u
        self.Ib = Ib
        self.ki = ki
        self.m30 = m30
        self.kd = kd
        self.ksc = ksc
        self.Fsnc = Fsnc


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
        self._numba_params = NumbaParams(
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


@njit(fastmath=True, cache=True)
def _jitted_model(t, x, last_Qsto, last_foodtaken, action_CHO, action_insulin, params_BW,
                         params_u2ss, params_kmax, params_b, params_d, params_f, params_kmin,
                         params_kabs, params_kp1, params_kp2, params_kp3, params_ke1,
                         params_ke2, params_k1, params_k2, params_Vm0, params_Vmx, params_Km0,
                         params_m1, params_m2, params_m4, params_ka1, params_ka2, params_Vi,
                         params_p2u, params_Ib, params_ki, params_m30, params_kd, params_ksc,
                         params_Fsnc):
    dxdt = np.zeros(13)
    d = action_CHO * 1000  # g -> mg
    insulin = action_insulin * 6000 / params_BW  # U/min -> pmol/kg/min
    basal = params_u2ss * params_BW / 6000  # U/min

    # Glucose in the stomach
    qsto = x[0] + x[1]
    # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
    # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
    # details.
    Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg

    # Stomach solid
    dxdt[0] = -params_kmax * x[0] + d

    if Dbar > 0:
        aa = 5 / (2 * Dbar * (1 - params_b))
        cc = 5 / (2 * Dbar * params_d)
        kgut = params_kmin + (params_kmax - params_kmin) / 2 * (
                np.tanh(aa * (qsto - params_b * Dbar))
                - np.tanh(cc * (qsto - params_d * Dbar))
                + 2
        )
    else:
        kgut = params_kmax

    # stomach liquid
    dxdt[1] = params_kmax * x[0] - x[1] * kgut

    # intestine
    dxdt[2] = kgut * x[1] - params_kabs * x[2]

    # Rate of appearance
    Rat = params_f * params_kabs * x[2] / params_BW
    # Glucose Production
    EGPt = params_kp1 - params_kp2 * x[3] - params_kp3 * x[8]
    # Glucose Utilization
    Uiit = params_Fsnc

    # renal excretion
    if x[3] > params_ke2:
        Et = params_ke1 * (x[3] - params_ke2)
    else:
        Et = 0

    # glucose kinetics
    # plus dextrose IV injection input u[2] if needed
    dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params_k1 * x[3] + params_k2 * x[4]
    dxdt[3] = (x[3] >= 0) * dxdt[3]

    Vmt = params_Vm0 + params_Vmx * x[6]
    Kmt = params_Km0
    Uidt = Vmt * x[4] / (Kmt + x[4])
    dxdt[4] = -Uidt + params_k1 * x[3] - params_k2 * x[4]
    dxdt[4] = (x[4] >= 0) * dxdt[4]

    # insulin kinetics
    dxdt[5] = (
            -(params_m2 + params_m4) * x[5]
            + params_m1 * x[9]
            + params_ka1 * x[10]
            + params_ka2 * x[11]
    )  # plus insulin IV injection u[3] if needed
    It = x[5] / params_Vi
    dxdt[5] = (x[5] >= 0) * dxdt[5]

    # insulin action on glucose utilization
    dxdt[6] = -params_p2u * x[6] + params_p2u * (It - params_Ib)

    # insulin action on production
    dxdt[7] = -params_ki * (x[7] - It)

    dxdt[8] = -params_ki * (x[8] - x[7])

    # insulin in the liver (pmol/kg)
    dxdt[9] = -(params_m1 + params_m30) * x[9] + params_m2 * x[5]
    dxdt[9] = (x[9] >= 0) * dxdt[9]

    # subcutaneous insulin kinetics
    dxdt[10] = insulin - (params_ka1 + params_kd) * x[10]
    dxdt[10] = (x[10] >= 0) * dxdt[10]

    dxdt[11] = params_kd * x[10] - params_ka2 * x[11]
    dxdt[11] = (x[11] >= 0) * dxdt[11]

    # subcutaneous glucose
    dxdt[12] = -params_ksc * x[12] + params_ksc * x[3]
    dxdt[12] = (x[12] >= 0) * dxdt[12]

    return dxdt, action_insulin, basal


def jitted_model(t, x, action, params, last_Qsto, last_foodtaken):
    action_CHO = action.CHO
    action_insulin = action.insulin
    if not isinstance(action_insulin, (float, int)):
        action_insulin = action_insulin[0]
    params_BW = params.BW
    params_u2ss = params.u2ss
    params_kmax = params.kmax
    params_b = params.b
    params_d = params.d
    params_f = params.f
    params_kmin = params.kmin
    params_kabs = params.kabs
    params_kp1 = params.kp1
    params_kp2 = params.kp2
    params_kp3 = params.kp3
    params_ke1 = params.ke1
    params_ke2 = params.ke2
    params_k1 = params.k1
    params_k2 = params.k2
    params_Vm0 = params.Vm0
    params_Vmx = params.Vmx
    params_Km0 = params.Km0
    params_m1 = params.m1
    params_m2 = params.m2
    params_m4 = params.m4
    params_ka1 = params.ka1
    params_ka2 = params.ka2
    params_Vi = params.Vi
    params_p2u = params.p2u
    params_Ib = params.Ib
    params_ki = params.ki
    params_m30 = params.m30
    params_kd = params.kd
    params_ksc = params.ksc
    params_Fsnc = params.Fsnc
    dxdt, action_insulin, basal =  _jitted_model(t, x, last_Qsto, last_foodtaken, action_CHO, action_insulin, params_BW,
                         params_u2ss, params_kmax, params_b, params_d, params_f, params_kmin,
                         params_kabs, params_kp1, params_kp2, params_kp3, params_ke1,
                         params_ke2, params_k1, params_k2, params_Vm0, params_Vmx, params_Km0,
                         params_m1, params_m2, params_m4, params_ka1, params_ka2, params_Vi,
                         params_p2u, params_Ib, params_ki, params_m30, params_kd, params_ksc,
                         params_Fsnc)

    if action_insulin > basal:
        logger.debug("t = {}, injecting insulin: {}".format(t, action_insulin))

    return dxdt


@njit(fastmath=True, cache=True)
def _jitted_jacobian(t, x, last_Qsto, last_foodtaken, action_CHO, action_insulin, params_BW,
                     params_u2ss, params_kmax, params_b, params_d, params_f, params_kmin,
                     params_kabs, params_kp1, params_kp2, params_kp3, params_ke1,
                     params_ke2, params_k1, params_k2, params_Vm0, params_Vmx, params_Km0,
                     params_m1, params_m2, params_m4, params_ka1, params_ka2, params_Vi,
                     params_p2u, params_Ib, params_ki, params_m30, params_kd, params_ksc,
                     params_Fsnc):
    """
    Calculates the 13x13 Jacobian matrix (df/dx) for the jitted model.
    """
    # jac[i, j] = d(dxdt[i]) / d(x[j])
    jac = np.zeros((13, 13))

    # --- 1. Re-calculate intermediate variables needed for derivatives ---

    qsto = x[0] + x[1]
    Dbar = last_Qsto + last_foodtaken * 1000

    dkgut_dx0 = 0.0
    dkgut_dx1 = 0.0

    if Dbar > 0:
        aa = 5 / (2 * Dbar * (1 - params_b))
        cc = 5 / (2 * Dbar * params_d)

        tanh_a = np.tanh(aa * (qsto - params_b * Dbar))
        tanh_c = np.tanh(cc * (qsto - params_d * Dbar))

        kgut = params_kmin + (params_kmax - params_kmin) / 2 * (
                tanh_a - tanh_c + 2
        )

        # Derivatives of kgut w.r.t qsto (which depends on x[0] and x[1])
        # d(tanh(u))/dx = (1 - tanh(u)**2) * du/dx
        dkgut_dqsto = (params_kmax - params_kmin) / 2 * (
                (1 - tanh_a ** 2) * aa - (1 - tanh_c ** 2) * cc
        )
        dkgut_dx0 = dkgut_dqsto  # d(qsto)/d(x[0]) = 1
        dkgut_dx1 = dkgut_dqsto  # d(qsto)/d(x[1]) = 1
    else:
        kgut = params_kmax
        # dkgut_dx0 and dkgut_dx1 remain 0.0

    EGPt = params_kp1 - params_kp2 * x[3] - params_kp3 * x[8]

    dEGPt_dx3 = 0.0
    dEGPt_dx8 = 0.0
    if EGPt > 0:
        dEGPt_dx3 = -params_kp2
        dEGPt_dx8 = -params_kp3

    dEt_dx3 = 0.0
    if x[3] > params_ke2:
        dEt_dx3 = params_ke1

    Vmt = params_Vm0 + params_Vmx * x[6]
    Kmt = params_Km0
    # d(Uidt)/d(x[4]) = d/dx4 [Vmt * x4 / (Kmt + x4)] (quotient rule)
    # = Vmt * [ (Kmt + x4)*1 - x4*1 ] / (Kmt + x4)**2
    # = Vmt * Kmt / (Kmt + x4)**2
    dUidt_dx4 = Vmt * Kmt / (Kmt + x[4]) ** 2

    # d(Uidt)/d(x[6]) = d/dx6 [ (Vm0 + Vmx*x6) * x4 / (Kmt + x4) ]
    # = Vmx * x4 / (Kmt + x4)
    dUidt_dx6 = params_Vmx * x[4] / (Kmt + x[4])

    # It = x[5] / params_Vi
    dIt_dx5 = 1.0 / params_Vi

    # --- 2. Populate the Jacobian matrix row by row ---

    # Row 0: dxdt[0] = -params_kmax * x[0] + d
    jac[0, 0] = -params_kmax

    # Row 1: dxdt[1] = params_kmax * x[0] - x[1] * kgut
    jac[1, 0] = params_kmax - x[1] * dkgut_dx0  # Chain rule
    jac[1, 1] = -kgut - x[1] * dkgut_dx1  # Product rule

    # Row 2: dxdt[2] = kgut * x[1] - params_kabs * x[2]
    jac[2, 0] = dkgut_dx0 * x[1]  # Chain rule
    jac[2, 1] = dkgut_dx1 * x[1] + kgut  # Product rule
    jac[2, 2] = -params_kabs

    # Row 3: dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params_k1 * x[3] + params_k2 * x[4]
    # Rat = params_f * params_kabs * x[2] / params_BW
    jac[3, 2] = params_f * params_kabs / params_BW  # from d(Rat)/d(x[2])
    jac[3, 3] = dEGPt_dx3 - dEt_dx3 - params_k1  # from d(EGPt)/d(x[3]), d(Et)/d(x[3]), d(-k1*x3)/d(x[3])
    jac[3, 4] = params_k2  # from d(k2*x4)/d(x[4])
    jac[3, 8] = dEGPt_dx8  # from d(EGPt)/d(x[8])
    if x[3] < 0:  # Apply non-negativity constraint
        jac[3, :] = 0.0

    # Row 4: dxdt[4] = -Uidt + params_k1 * x[3] - params_k2 * x[4]
    jac[4, 3] = params_k1
    jac[4, 4] = -dUidt_dx4 - params_k2
    jac[4, 6] = -dUidt_dx6
    if x[4] < 0:  # Apply non-negativity constraint
        jac[4, :] = 0.0

    # Row 5: dxdt[5] = -(m2 + m4)*x[5] + m1*x[9] + ka1*x[10] + ka2*x[11]
    jac[5, 5] = -(params_m2 + params_m4)
    jac[5, 9] = params_m1
    jac[5, 10] = params_ka1
    jac[5, 11] = params_ka2
    if x[5] < 0:  # Apply non-negativity constraint
        jac[5, :] = 0.0

    # Row 6: dxdt[6] = -p2u*x[6] + p2u*(It - Ib)
    # It = x[5] / Vi
    jac[6, 5] = params_p2u * dIt_dx5
    jac[6, 6] = -params_p2u

    # Row 7: dxdt[7] = -ki*(x[7] - It)
    jac[7, 5] = -params_ki * (-dIt_dx5)  # = params_ki * dIt_dx5
    jac[7, 7] = -params_ki

    # Row 8: dxdt[8] = -ki*(x[8] - x[7])
    jac[8, 7] = params_ki
    jac[8, 8] = -params_ki

    # Row 9: dxdt[9] = -(m1 + m30)*x[9] + m2*x[5]
    jac[9, 5] = params_m2
    jac[9, 9] = -(params_m1 + params_m30)
    if x[9] < 0:  # Apply non-negativity constraint
        jac[9, :] = 0.0

    # Row 10: dxdt[10] = insulin - (ka1 + kd)*x[10]
    jac[10, 10] = -(params_ka1 + params_kd)
    if x[10] < 0:  # Apply non-negativity constraint
        jac[10, :] = 0.0

    # Row 11: dxdt[11] = kd*x[10] - ka2*x[11]
    jac[11, 10] = params_kd
    jac[11, 11] = -params_ka2
    if x[11] < 0:  # Apply non-negativity constraint
        jac[11, :] = 0.0

    # Row 12: dxdt[12] = -ksc*x[12] + ksc*x[3]
    jac[12, 3] = params_ksc
    jac[12, 12] = -params_ksc
    if x[12] < 0:  # Apply non-negativity constraint
        jac[12, :] = 0.0

    return jac


def jitted_jacobian(t, x, action, params, last_Qsto, last_foodtaken):
    """
    Wrapper for the jitted jacobian function.
    Unpacks action and params objects.
    """
    action_CHO = action.CHO
    action_insulin = action.insulin
    if not isinstance(action_insulin, (float, int)):
        action_insulin = action_insulin[0]
    params_BW = params.BW
    params_u2ss = params.u2ss
    params_kmax = params.kmax
    params_b = params.b
    params_d = params.d
    params_f = params.f
    params_kmin = params.kmin
    params_kabs = params.kabs
    params_kp1 = params.kp1
    params_kp2 = params.kp2
    params_kp3 = params.kp3
    params_ke1 = params.ke1
    params_ke2 = params.ke2
    params_k1 = params.k1
    params_k2 = params.k2
    params_Vm0 = params.Vm0
    params_Vmx = params.Vmx
    params_Km0 = params.Km0
    params_m1 = params.m1
    params_m2 = params.m2
    params_m4 = params.m4
    params_ka1 = params.ka1
    params_ka2 = params.ka2
    params_Vi = params.Vi
    params_p2u = params.p2u
    params_Ib = params.Ib
    params_ki = params.ki
    params_m30 = params.m30
    params_kd = params.kd
    params_ksc = params.ksc
    params_Fsnc = params.Fsnc

    # Call the jitted jacobian
    jac = _jitted_jacobian(t, x, last_Qsto, last_foodtaken, action_CHO, action_insulin, params_BW,
                           params_u2ss, params_kmax, params_b, params_d, params_f, params_kmin,
                           params_kabs, params_kp1, params_kp2, params_kp3, params_ke1,
                           params_ke2, params_k1, params_k2, params_Vm0, params_Vmx, params_Km0,
                           params_m1, params_m2, params_m4, params_ka1, params_ka2, params_Vi,
                           params_p2u, params_Ib, params_ki, params_m30, params_kd, params_ksc,
                           params_Fsnc)

    return jac


class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(self, params, init_state=None, random_init_bg=False, seed=None, t0=0):
        """
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        """
        self._params = ParamManager(params)
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # Get args for ODE solver
        extra_args = (
            action,
            self._params._numba_params,
            self._last_Qsto,
            self._last_foodtaken
        )

        # ODE solver
        self._odesolver.set_f_params(*extra_args)
        self._odesolver.set_jac_params(*extra_args)

        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        return jitted_model(t, x, action, params, last_Qsto, last_foodtaken)

    @staticmethod
    def jacobian(t, x, action, params, last_Qsto, last_foodtaken):
        return jitted_jacobian(t, x, action, params, last_Qsto, last_foodtaken)

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        """
        Reset the patient state to default intial state
        """
        if self._init_state is None:
            self.init_state = np.stack(self._params.iloc[2:15].values).copy()
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            # Only randomize glucose related states, x4, x5, and x13
            mean = [
                1.0 * self.init_state[3],
                1.0 * self.init_state[4],
                1.0 * self.init_state[12],
            ]
            cov = np.diag(
                [
                    0.1 * self.init_state[3],
                    0.1 * self.init_state[4],
                    0.1 * self.init_state[12],
                ]
            )
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model, self.jacobian).set_integrator("lsoda")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    p = T1DPatient.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()
