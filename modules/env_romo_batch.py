import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import OrderedDict
import random # Use Python's random for choice as in EC env

# Action mapping constants
ACT_FIXATE = 0
ACT_GT = 1           # Greater Than (f1 > f2)
ACT_LT = 2           # Less Than (f1 < f2)

# Observation vector indices constants
OBS_FIXATION_CUE = 0 # Input signaling fixation is required or decision is possible
OBS_F_POS = 1        # Input representing activity of positively-tuned neurons
OBS_F_NEG = 2        # Input representing activity of negatively-tuned neurons
OBS_DIM = 3          # Total number of observation features

# Epoch names constants
EPOCH_FIXATION = 'fixation'
EPOCH_F1 = 'f1'
EPOCH_DELAY = 'delay'
EPOCH_F2 = 'f2'
EPOCH_DECISION = 'decision'
EPOCH_END = 'end'


class WorkingMemoryEnv(gym.Env):
    """
    Gymnasium environment for the Parametric Working Memory Task with added
    rewards for correct fixation and penalties for fixation during decision.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self,
                 dt=10,
                 reward_correct=+1.0,
                 reward_aborted=-1.0,
                 reward_incorrect=0.0,
                 reward_fixation=+0.1,
                 reward_decide_fixation=-0.1,
                 input_noise_sigma=0.05,
                 duration_params=[1500, 3000, 500, 3000, 500, 500]
                 ):
        super().__init__()

        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.input_noise_sigma = input_noise_sigma

        # --- Reward configuration ---
        self.R_CORRECT = float(reward_correct)
        self.R_ABORTED = float(reward_aborted)
        self.R_INCORRECT = float(reward_incorrect)
        self.R_FIXATION = float(reward_fixation)
        self.R_DECIDE_FIXATION = float(reward_decide_fixation)
        self.R_DEFAULT_STEP = 0.0

        # --- Spaces ---
        self.action_space = spaces.Discrete(3)
        obs_low, obs_high = -0.05, 1.05
        self.observation_space = spaces.Box(
            low=np.array([0.0, obs_low, obs_low], dtype=np.float32),
            high=np.array([1.0, obs_high, obs_high], dtype=np.float32),
            shape=(OBS_DIM,), 
            dtype=np.float32
            )

        # --- Timing ---
        if len(duration_params) != 6:
            raise ValueError("duration_params must be a list of 6 durations")
        self._fixation_min_ms, self._fixation_max_ms, self._f1_ms, self._delay_ms, self._f2_ms, self._decision_ms = duration_params
        # steps
        self.t_fixation_min_steps = self._ms_to_steps(self._fixation_min_ms)
        self.t_fixation_max_steps = self._ms_to_steps(self._fixation_max_ms)
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
        self.rng = np.random.default_rng()
        self.current_step = 0
        self.trial_params = None
        self.epochs = {}
        self.current_epoch_name = EPOCH_END
        self.chosen_action_idx = -1
        self.is_correct_choice = None
        self._last_action_info = None
        self._last_reward_info = None
        self._last_info_dict = None
        self._last_truncated_info = None

    def _ms_to_steps(self, ms):
        return max(1, int(round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, fixation_steps, delay_steps):
        t_fix_end = fixation_steps
        t_f1_end = t_fix_end + self.t_f1_steps
        t_delay_end = t_f1_end + delay_steps
        t_f2_end = t_delay_end + self.t_f2_steps
        t_decision_end = t_f2_end + self.t_decision_steps
        t_max_steps = t_decision_end
        self.epochs = {
            EPOCH_FIXATION: (0, t_fix_end),
            EPOCH_F1: (t_fix_end, t_f1_end),
            EPOCH_DELAY: (t_f1_end, t_delay_end),
            EPOCH_F2: (t_delay_end, t_f2_end),
            EPOCH_DECISION: (t_f2_end, t_decision_end),
            EPOCH_END: (t_max_steps, t_max_steps+1),
            'tmax_steps': t_max_steps
        }

    def _get_current_epoch(self, step):
        for name in [EPOCH_FIXATION, EPOCH_F1, EPOCH_DELAY, EPOCH_F2, EPOCH_DECISION]:
            start, end = self.epochs[name]
            if start <= step < end:
                return name
        return EPOCH_END

    def _select_trial_conditions(self):
        fixation_ms = self.rng.uniform(self._fixation_min_ms, self._fixation_max_ms)
        fixation_steps = self._ms_to_steps(fixation_ms)
        delay_steps = self.t_delay_steps
        gt_lt = self.rng.choice(self.gt_lts)
        fpair = self.rng.choice(self.fpairs)
        
        self._set_trial_params_and_epochs(fixation_steps, self._delay_ms, gt_lt, fpair)

    def _set_trial_params_and_epochs(self, fixation_steps, delay_ms, gt_lt, fpair):
        """Helper para establecer parámetros y calcular épocas."""
        self.trial_params = {
            'fixation_ms': fixation_steps * self.dt,
            'fixation_steps': fixation_steps,
            'delay_ms': delay_ms,
            'delay_steps': self._ms_to_steps(delay_ms),
            'gt_lt': gt_lt,
            'fpair': fpair
        }
        self._calculate_epochs(fixation_steps, self._ms_to_steps(delay_ms))

    def _scale_freq(self, f):
        return (f - self.fmin) / (self.fmax - self.fmin) if self.fmax != self.fmin else 0.0

    def _scale_p(self, f): return self._scale_freq(f)

    def _scale_n(self, f): return 1.0 - self._scale_freq(f)

    def _get_observation(self, current_epoch):
        obs = np.zeros(OBS_DIM, np.float32)
        if current_epoch in [EPOCH_FIXATION, EPOCH_F1, EPOCH_DELAY, EPOCH_F2]:
            obs[OBS_FIXATION_CUE] = 1.0
        
        if self.trial_params is not None:
            f_low, f_high = self.trial_params['fpair']
            if self.trial_params['gt_lt'] == '>': f1, f2 = float(f_high), float(f_low)
            else: f1, f2 = float(f_low), float(f_high)
            
            if current_epoch == EPOCH_F1:
                noise_pos = self.rng.normal(loc=0.0, scale=self.input_noise_sigma)
                noise_neg = self.rng.normal(loc=0.0, scale=self.input_noise_sigma)
                obs[OBS_F_POS] = self._scale_p(f1) + noise_pos
                obs[OBS_F_NEG] = self._scale_n(f1) + noise_neg
            
            elif current_epoch == EPOCH_F2:
                noise_pos = self.rng.normal(loc=0.0, scale=self.input_noise_sigma)
                noise_neg = self.rng.normal(loc=0.0, scale=self.input_noise_sigma)
                obs[OBS_F_POS] = self._scale_p(f2) + noise_pos
                obs[OBS_F_NEG] = self._scale_n(f2) + noise_neg
            
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        self.current_step = 0
        
        if options and 'gt_lt' in options and 'fpair' in options:
            fixation_ms = self.rng.uniform(self._fixation_min_ms, self._fixation_max_ms)
            fixation_steps = self._ms_to_steps(fixation_ms)
            self._set_trial_params_and_epochs(fixation_steps, self._delay_ms, options['gt_lt'], options['fpair'])
        else:
            self._select_trial_conditions()

        self.chosen_action_idx = -1
        self.is_correct_choice = None
        self._last_action_info = None
        self._last_reward_info = None
        self._last_info_dict = None
        self._last_truncated_info = None
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        obs = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info['reward'] = 0.0; info['action'] = None
        if self.render_mode in ['human', 'ansi']: self._render_text()
        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_epoch = self.current_epoch_name
        reward = self.R_DEFAULT_STEP
        terminated = False
        truncated = False
        immediate_reason = None

        # --- Reward and termination logic based on the PREVIOUS epoch ---

        # --- FIXATION EPOCH: reward for correct fixate ---
        if prev_epoch == EPOCH_FIXATION:
            if action == ACT_FIXATE:
                reward = self.R_FIXATION # Get +0.1 for fixating
            elif action in [ACT_GT, ACT_LT]:
                reward = self.R_ABORTED # Get -1.0 for early decision
                terminated = True
                immediate_reason = 'early_decision_abort'

        # --- STIMULUS (F1, F2) and DELAY EPOCHS: reward for fixating, disallow decision ---
        # MODIFIED BLOCK: Added positive reward for ACT_FIXATE here
        elif prev_epoch in [EPOCH_F1, EPOCH_DELAY, EPOCH_F2]:
            if action == ACT_FIXATE:
                reward = self.R_FIXATION # Get +0.1 for fixating in these epochs
            elif action in [ACT_GT, ACT_LT]:
                reward = self.R_ABORTED # Get -1.0 for early decision
                terminated = True
                immediate_reason = 'early_decision_abort'

        # --- DECISION EPOCH: penalty for fixating, reward/abort for decisions ---
        elif prev_epoch == EPOCH_DECISION:
            if action == ACT_FIXATE:
                reward = self.R_DECIDE_FIXATION # Get -0.1 for fixating
                # Trial continues this step, but will be aborted if it passes end of decision window
            elif action in [ACT_GT, ACT_LT]:
                self.chosen_action_idx = action
                terminated = True # Trial ends immediately on decision
                immediate_reason = 'decision'
                correct_str = self.trial_params['gt_lt']
                correct_act = ACT_GT if correct_str == '>' else ACT_LT
                self.is_correct_choice = (action == correct_act)
                reward = self.R_CORRECT if self.is_correct_choice else self.R_INCORRECT


        # --- Advance time ---
        self.current_step += 1
        # Determine the epoch of the *next* step
        next_epoch = self._get_current_epoch(self.current_step)

        # --- Check for truncation / Timeout conditions ---
        # Truncate if we passed the max steps allowed for the trial
        if self.current_step >= self.epochs['tmax_steps']:
             truncated = True
             immediate_reason = immediate_reason or 'end_of_time' # If no immediate reason yet

        # Special case: Abort if agent was in DECISION epoch and chose FIXATE,
        # and now the next step is *outside* the decision epoch.
        if not terminated and prev_epoch == EPOCH_DECISION and next_epoch != EPOCH_DECISION and action == ACT_FIXATE:
            # This scenario means the agent failed to decide within the window
            reward = self.R_ABORTED # Apply abort penalty (overrides R_DECIDE_FIXATION from last step)
            terminated = True
            immediate_reason = 'decision_timeout_abort'


        # Update current epoch name (only if not terminated, otherwise it's the END epoch)
        self.current_epoch_name = next_epoch if not (terminated or truncated) else EPOCH_END

        # Get observation for the *next* step (will be zeros if terminated/truncated)
        obs = self._get_observation(self.current_epoch_name)

        # Prepare info dictionary
        info = self._get_info()
        info['action_taken_this_step'] = action
        info['reward_this_step'] = reward
        if immediate_reason: info['reason'] = immediate_reason # Add specific reason to info

        # Store for rendering (if needed)
        self._last_action_info = action
        self._last_reward_info = reward
        self._last_info_dict = info.copy()
        self._last_truncated_info = truncated

        # Render if mode is set
        if self.render_mode in ['human', 'ansi']:
            self._render_text() # Assuming _render_text uses the stored _last_ info

        # Gymnasium requires truncated implies terminated in v0.26+
        if truncated:
             terminated = True

        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = {
            "step": self.current_step,
            "time_ms": self.current_step * self.dt,
            "epoch": self.current_epoch_name,
            "trial_params": self.trial_params,
            "chosen_action": {0:'FIXATE',1:'>',2:'<'}[self.chosen_action_idx] if self.chosen_action_idx!=-1 else None,
            "is_correct_choice": self.is_correct_choice,
            "rewards_cfg": {
                "correct": self.R_CORRECT,
                "incorrect": self.R_INCORRECT,
                "aborted": self.R_ABORTED,
                "fixation": self.R_FIXATION,
                "decide_fixation": self.R_DECIDE_FIXATION,
                "default_step": self.R_DEFAULT_STEP,
            },
            "fixed_durations_ms": { 'fixation_min':self._fixation_min_ms, 'fixation_max':self._fixation_max_ms,
                                   'f1':self._f1_ms, 'f2':self._f2_ms, 
                                   'delay_ms':self._delay_ms, 'decision':self._decision_ms },
            "trial_delay_ms": self.trial_params['delay_ms'] if self.trial_params else None,
            "epoch_boundaries_steps": self.epochs
        }
        return info

    def _render_text(self): pass

    def render(self): pass
    def close(self): pass