"""MDP environment for the Adaptive Sensor Activation Policy (paper Sec. IV),
calibrated to Table II/III. Energy in Wh per MSS.

Root-cause fix vs earlier versions: utility has DIMINISHING RETURNS in the
number of records collected (information saturates). This gives a well-defined
interior optimum for the sensing duty cycle, so TD3 converges to a stable
adaptive policy instead of collapsing to always-on or always-off.

State   s = {cond(S), battery, energy_ratio, health(S), gamma, hist(K),
             quota_progress(S), t/T}
Action  a in [-1,1]^(S+1): a[:S] -> activation logits (>0 = on); a[S] -> gamma
Reward  R = -E_step + lambda1 * dUtility            (Eq. 16, maximisation form)
"""
import numpy as np
import config as C
from semantic import compression_ratio, utility


class SensorActivationEnv:
    def __init__(self, seed=C.SEED):
        self.rng = np.random.default_rng(seed)
        self.S, self.T, self.K = C.S, C.T, 4
        self.state_dim = self.S + 2 + self.S + 1 + self.K + self.S + 1
        self.action_dim = self.S + 1
        self.lambda1 = C.LAMBDA1                      # paper optimal ~0.6
        # Per-sensor comm energy per slot (raw), so all-on-raw over T == 5.2 Wh.
        self.comm_raw_per = (C.KC_RAW * C.T) / self.S / self.T
        # Utility scale chosen so marginal utility ~ marginal energy near the
        # paper's operating sensing energy (~3.4 Wh => duty ~0.48).
        self.util_scale = 0.9
        self.reset()

    def reset(self):
        self.t = 0
        self.battery = 1.0
        self.energy_used = 0.0
        self.records = np.zeros(self.S)
        self.gamma = 0.5
        self.hist = np.zeros(self.K)
        self.cond = self.rng.random(self.S)
        self.health = np.clip(self.rng.normal(1.0, 0.05, self.S), 0.7, 1.0)
        return self._obs()

    def _obs(self):
        progress = np.clip(self.records / C.MIN_RECORDS, 0.0, 1.0)
        return np.concatenate([
            self.cond,
            [self.battery, self.energy_used / 30.0],
            self.health,
            [self.gamma],
            self.hist,
            progress,
            [self.t / self.T],
        ]).astype(np.float32)

    @staticmethod
    def _concave(n):
        """Diminishing-returns information value of having n records."""
        return np.sqrt(np.maximum(n, 0.0))

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        y = (a[:self.S] > 0.0).astype(np.float32)            # binary activation
        gamma = float(np.clip((a[self.S] + 1.0) / 2.0, C.GAMMA[0], 1.0))
        self.gamma = gamma

        # ----- energy (Wh), calibrated to Table III -----
        e_sense = float((C.SENSOR_ENERGY * y).sum())
        e_comm = float((self.comm_raw_per * y).sum()) * compression_ratio(gamma)
        e_fixed = C.E_FLY_HOVER / self.T                      # constant floor / slot
        e_total = e_fixed + e_sense + e_comm

        # ----- utility: concave gain in records, weighted by fidelity & condition
        u_fid = utility(gamma)                                # U(gamma) = 1 - e^{-a*g}
        before = self._concave(self.records)
        self.records += y
        after = self._concave(self.records)
        d_util = float(((after - before) * self.cond).sum()) * u_fid * self.util_scale

        # quota penalty: strong, only while below required records (Eq.16 indicator)
        deficit = float((self.records < C.MIN_RECORDS).sum())

        reward = -e_total + self.lambda1 * d_util - 0.02 * deficit

        # ----- transition -----
        self.energy_used += e_total
        self.battery = max(0.0, self.battery - e_total / 30.0)
        self.cond = np.clip(self.cond + self.rng.normal(0, 0.1, self.S), 0, 1)
        self.hist = np.roll(self.hist, 1); self.hist[0] = y.mean()
        self.t += 1
        done = self.t >= self.T
        info = {"energy": e_total, "sense": e_sense, "comm": e_comm,
                "duty": float(y.mean()), "gamma": gamma}
        return self._obs(), float(reward), done, info

    def fixed_energy(self):
        """FIXED policy: all sensors on every slot (Table III FIXED row)."""
        return float(C.E_FLY_HOVER + C.SENSOR_ENERGY.sum() * self.T
                     + self.comm_raw_per * self.S * self.T * compression_ratio(0.5))
