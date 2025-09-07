import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Action mapping constants
ACT_FIXATE = 0
ACT_GT = 1
ACT_LT = 2
ACT_KEY_DOWN = 3

# Observation vector indices constants
OBS_FIXATION_CUE = 0 
OBS_F_POS = 1        
OBS_F_NEG = 2        
OBS_DIM = 3          

# Epoch names constants
EPOCH_WAIT_FOR_START = 'wait_for_start'
EPOCH_FIXATION = 'fixation'
EPOCH_F1 = 'f1'
EPOCH_DELAY = 'delay'
EPOCH_F2 = 'f2'
EPOCH_DECISION = 'decision'
EPOCH_END = 'end'


class WorkingMemoryEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self,
                 dt=10,
                 reward_correct=+1.0,
                 reward_aborted=-1.0,
                 reward_incorrect=0.0,
                 reward_holding=+0.1,
                 input_noise_sigma=0.001,
                 duration_params=[1500, 3000, 500, 3000, 500, 500],
                 m_init=1.0,
                 rho_init=0.0,
                 kappa=0.1,
                 epsilon=0.087,
                 latency_a=0.73,
                 latency_noise=0.0,
                 start_window_steps=5,
                 m_min=0.005,
                 m_max=2.0,
                 ):
        super().__init__()
        
        self.dt = dt
        
        # --- Reward configuration ---
        self.R_CORRECT = float(reward_correct)
        self.R_ABORTED = float(reward_aborted)
        self.R_INCORRECT = float(reward_incorrect)
        self.R_HOLDING = float(reward_holding)
        self.R_DEFAULT_STEP = 0.0

        # --- Input noise configuration ---
        self.sigma = np.sqrt(2 * 100 * input_noise_sigma)
        self.noise_scale = 1.0 / np.sqrt(dt / 1000.0) if dt > 0 else 1.0

        # --- Spaces ---
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=34.0, shape=(OBS_DIM,), dtype=np.float32)

        # --- Timing ---
        self._fixation_min_ms, self._fixation_max_ms, self._f1_ms, self._delay_ms, self._f2_ms, self._decision_ms = duration_params
        self.t_f1_steps = self._ms_to_steps(self._f1_ms)
        self.t_delay_steps = self._ms_to_steps(self._delay_ms)
        self.t_f2_steps = self._ms_to_steps(self._f2_ms)
        self.t_decision_steps = self._ms_to_steps(self._decision_ms)

        # --- Trials ---
        self.gt_lts = ['>', '<']
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
        all_freqs = np.ravel(self.fpairs)
        self.fmin, self.fmax = np.min(all_freqs), np.max(all_freqs)

        # --- State ---
        self.m = m_init
        self.m_min = m_min
        self.m_max = m_max
        self.rho = rho_init
        self.kappa = kappa
        self.epsilon = epsilon
        self.latency_a = latency_a
        self.latency_noise = latency_noise
        self.t_start_window = start_window_steps
        self.rng = np.random.default_rng()

    def _upper_level(self, R):
        delta_rho = (self.m * R - self.rho)
        self.rho += self.kappa * delta_rho
        self.m += self.epsilon * delta_rho
        self.m = np.clip(self.m, self.m_min, self.m_max)

    def _latency(self):
        m_safe = self.m if self.m != 0 else 1e-6 
        tau_sec = self.latency_a / m_safe + self.latency_noise * self.rng.normal(0, 1)
        return max(0, tau_sec * 1000)

    def _ms_to_steps(self, ms):
        return max(1, int(round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, fixation_steps, delay_steps):
        t_fix_end = fixation_steps
        t_f1_end = t_fix_end + self.t_f1_steps
        t_delay_end = t_f1_end + delay_steps
        t_f2_end = t_delay_end + self.t_f2_steps
        t_decision_end = t_f2_end + self.t_decision_steps
        self.epochs = {
            EPOCH_FIXATION: (0, t_fix_end),
            EPOCH_F1: (t_fix_end, t_f1_end),
            EPOCH_DELAY: (t_f1_end, t_delay_end),
            EPOCH_F2: (t_delay_end, t_f2_end),
            EPOCH_DECISION: (t_f2_end, t_decision_end),
            'tmax_steps': t_decision_end
        }

    def _get_current_epoch(self, step):
        if not self.trial_has_started:
            return EPOCH_WAIT_FOR_START
        relative_step = step - self.trial_started_step
        for name in [EPOCH_FIXATION, EPOCH_F1, EPOCH_DELAY, EPOCH_F2, EPOCH_DECISION]:
            start, end = self.epochs[name]
            if start <= relative_step < end:
                return name
        return EPOCH_END

    def _select_trial_conditions(self):
        fixation_ms = self.rng.uniform(self._fixation_min_ms, self._fixation_max_ms)
        self.trial_params = {
            'fixation_steps': self._ms_to_steps(fixation_ms),
            'delay_steps': self.t_delay_steps,
            'gt_lt': self.rng.choice(self.gt_lts),
            'fpair': self.rng.choice(self.fpairs)
        }
        
    def _scale_freq(self, f):
        return (f - self.fmin) / (self.fmax - self.fmin) if self.fmax != self.fmin else 0.0

    def _scale_p(self, f): return self._scale_freq(f)
    def _scale_n(self, f): return 1.0 - self._scale_freq(f)

    def _get_observation(self, current_epoch):
        if current_epoch == EPOCH_WAIT_FOR_START:
            return np.zeros(OBS_DIM, np.float32)

        obs = np.zeros(OBS_DIM, np.float32)
        if current_epoch in [EPOCH_FIXATION, EPOCH_F1, EPOCH_DELAY, EPOCH_F2]:
            obs[OBS_FIXATION_CUE] = 1.0
        
        if self.trial_params is not None:
            f_low, f_high = self.trial_params['fpair']
            if self.trial_params['gt_lt'] == '>': f1, f2 = float(f_high), float(f_low)
            else: f1, f2 = float(f_low), float(f_high)
            if current_epoch == EPOCH_F1:
                obs[OBS_F_POS] = self._scale_p(f1) + (self.rng.normal(scale=self.sigma)*self.noise_scale)
                obs[OBS_F_NEG] = self._scale_n(f1) + (self.rng.normal(scale=self.sigma)*self.noise_scale)
            elif current_epoch == EPOCH_F2:
                obs[OBS_F_POS] = self._scale_p(f2) + (self.rng.normal(scale=self.sigma)*self.noise_scale)
                obs[OBS_F_NEG] = self._scale_n(f2) + (self.rng.normal(scale=self.sigma)*self.noise_scale)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        self.current_step = 0
        self.chosen_action_idx = -1
        self.is_correct_choice = None
        self.trial_has_started = False
        self.trial_started_step = -1
        tau_ms = self._latency()
        self.t_start_target = self._ms_to_steps(tau_ms)
        self._select_trial_conditions()
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        obs = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_epoch = self.current_epoch_name
        reward = self.R_DEFAULT_STEP
        terminated = False
        truncated = False
        immediate_reason = None
        
        if not self.trial_has_started:
            # En la fase de espera, la acción correcta para continuar es fijar
            if action == ACT_FIXATE:
                reward = self.R_DEFAULT_STEP
            
            is_in_window = (self.t_start_target - self.t_start_window) <= self.current_step <= (self.t_start_target + self.t_start_window)
            if action == ACT_KEY_DOWN:
                if is_in_window:
                    self.trial_has_started = True
                    self.trial_started_step = self.current_step
                    self._calculate_epochs(self.trial_params['fixation_steps'], self.trial_params['delay_steps'])
                    immediate_reason = 'trial_started'
                    reward = self.R_DEFAULT_STEP # Empezar correctamente no da recompensa, la sobrescribe
                else:
                    reward = self.R_ABORTED
                    terminated = True
                    immediate_reason = 'start_time_missed'
            elif action != ACT_FIXATE: # Cualquier otra acción es un aborto
                reward = self.R_ABORTED
                terminated = True
                immediate_reason = 'wrong_action_at_start'
            
            if not terminated and self.current_step > (self.t_start_target + self.t_start_window):
                reward = self.R_ABORTED
                terminated = True
                immediate_reason = 'start_timeout'

        else: # El ensayo ya ha comenzado
            # En estas épocas, la acción correcta para continuar es fijar
            if prev_epoch in [EPOCH_FIXATION, EPOCH_F1, EPOCH_DELAY, EPOCH_F2]:
                if action == ACT_FIXATE:
                    reward = self.R_HOLDING
                else: # Cualquier otra acción es un aborto
                    reward = self.R_ABORTED
                    terminated = True
                    immediate_reason = 'early_decision_abort'

            elif prev_epoch == EPOCH_DECISION:
                if action in [ACT_GT, ACT_LT]:
                    self.chosen_action_idx = action
                    terminated = True
                    immediate_reason = 'decision'
                    correct_str = self.trial_params['gt_lt']
                    correct_act = ACT_GT if correct_str == '>' else ACT_LT
                    self.is_correct_choice = (action == correct_act)
                    # La recompensa final sobrescribe cualquier otra
                    reward = self.R_CORRECT if self.is_correct_choice else self.R_INCORRECT
                else: # Fijar o presionar KEY_DOWN en la época de decisión es un error
                    # No hay recompensa por mantener, la recompensa se queda en 0.0 por este paso
                    pass

            if not terminated and (self.current_step - self.trial_started_step) >= self.epochs['tmax_steps']:
                reward = self.R_ABORTED 
                terminated = True
                immediate_reason = 'decision_timeout_abort'
        
        self.current_step += 1
        next_epoch = self._get_current_epoch(self.current_step)
        self.current_epoch_name = next_epoch if not terminated else EPOCH_END
        obs = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info['action_taken_this_step'] = action
        info['reward_this_step'] = reward
        if immediate_reason: info['reason'] = immediate_reason
        
        if terminated:
            final_reward_for_update = reward
            self._upper_level(final_reward_for_update)
            info['m_after_update'] = self.m
            info['rho_after_update'] = self.rho

        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = {
            "step": self.current_step,
            "epoch": self.current_epoch_name,
            "is_correct_choice": self.is_correct_choice,
            "motivation_m": self.m,
            "avg_reward_rho": self.rho,
            "target_start_step": self.t_start_target,
            "trial_has_started": self.trial_has_started,
            "rewards_cfg": {
                "correct": self.R_CORRECT,
                "incorrect": self.R_INCORRECT,
                "aborted": self.R_ABORTED,
                "holding": self.R_HOLDING,
            }
        }
        return info

    def render(self): pass
    def close(self): pass