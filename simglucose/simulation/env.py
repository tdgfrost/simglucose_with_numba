from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple, deque
from simglucose.simulation.rendering import Viewer

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple("Observation", ["CGM"])
logger = logging.getLogger(__name__)


def risk_return(BG_last_hour):
    if len(BG_last_hour) < 1:
        raise ValueError("No blood glucose values available for reward calculation!")
    else:
        risk_current = risk_index([BG_last_hour[-1]], 1)[-1]
        return risk_current

def bg_in_range_quadratic(BG_last_hour):
    bg = BG_last_hour[0]
    w, u, v = 5.0, 0.01, 0.0001
    if bg < 70:
        r_bg = -w * (70 - bg) ** 2
    elif bg > 180:
        r_bg = -u * (bg - 180) ** 2
    else:
        r_bg = 200

    r_bg /= 1440

    return r_bg


def bg_in_range_magni(BG_last_hour):
    risk_current = risk_return(BG_last_hour)

    reward = 7 - 10 * risk_current

    return reward


def early_termination_reward(done):
    if done:
        return -100
    return 0


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=bg_in_range_magni):
        """
        action is a namedtuple with keys: basal, bolus
        """
        CHO = 0.0
        interval_CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0
        reward = 0.0
        most_recent_CGM = deque(maxlen=3)
        total_CHO = 0.0

        for i in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            interval_CHO += tmp_CHO

            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time
            most_recent_CGM.append(tmp_CGM)
            total_CHO += tmp_CHO
 
            reward += reward_fun([sum(most_recent_CGM) / len(most_recent_CGM)])

            if (i+1) % 3 == 0:
                self.insulin_hist.append(insulin)
                self.CGM_hist.append(CGM)
                self.CHO_hist.append(interval_CHO)
                interval_CHO = 0.0

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute any additional reward, and decide whether game is over
        done = BG < 10 or BG > 600
        reward += early_termination_reward(done)

        obs = Observation(CGM=[sum(most_recent_CGM) / len(most_recent_CGM), insulin, total_CHO])

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state,
            time=self.time,
            bg=BG,
            lbgi=LBGI,
            hbgi=HBGI,
            risk=risk,
        )

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = [0]
        self.insulin_hist = [0]

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        # Obs of shape (L, 2) for CGM, insulin, CHO
        historic_obs = [[CGM, 0, 0]]
        obs = Observation(CGM=[CGM, 0, 0])
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state,
            time=self.time,
            historic_obs=historic_obs,
            bg=self.BG_hist[0],
            lbgi=self.LBGI_hist[0],
            hbgi=self.HBGI_hist[0],
            risk=self.risk_hist[0],
        )

    def render(self, close=False):
        if close:
            self._close_viewer()
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def show_history(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.time_hist)
        df["BG"] = pd.Series(self.BG_hist)
        df["CGM"] = pd.Series(self.CGM_hist)
        df["CHO"] = pd.Series(self.CHO_hist)
        df["insulin"] = pd.Series(self.insulin_hist)
        df["LBGI"] = pd.Series(self.LBGI_hist)
        df["HBGI"] = pd.Series(self.HBGI_hist)
        df["Risk"] = pd.Series(self.risk_hist)
        df = df.set_index("Time")
        return df
