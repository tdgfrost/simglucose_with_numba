from .base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
from numba import njit

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)


@njit
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
        self._params = params
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

        # ODE solver
        self._odesolver.set_f_params(
            action, self._params, self._last_Qsto, self._last_foodtaken
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        return jitted_model(t, x, action, params, last_Qsto, last_foodtaken)

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
            self.init_state = np.copy(self._params.iloc[2:15].values)
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

        self._odesolver = ode(self.model).set_integrator("dopri5")
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
