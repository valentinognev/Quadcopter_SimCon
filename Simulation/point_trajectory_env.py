import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from copy import deepcopy
import sys
sys.path.append("../sample-factory")
from sample_factory.envs.env_utils import TrainingInfoInterface

import csv, os, time, atexit
from datetime import datetime

def quad_sim(t, Ts, quad, ctrl, desired):
    
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, None)
    t += Ts

    # Trajectory for Desired States 
    # ---------------------------
    desPos = np.array([0.0, 0.0, 0.0])    # Desired position (x, y, z)
    desVel = np.array([desired[0], desired[1], 0.0])    # Desired velocity (xdot, ydot, zdot)
    desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
    desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
    desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
    desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
    desYawRate = 0.         # Desired yaw speed
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(quad, sDes, Ts)

    return t, sDes
    
class PointTrajectoryEnv(gym.Env, TrainingInfoInterface):
    """
    Gymnasium environment for a 2D drone tracking a moving target.
    Actions: polar velocities [v_r, v_theta, w_yaw].
    Observations: [r, theta, radial_rate].
    Rewards: weighted MSE on r and theta, LOS bonus, radial closure bonus.
    Target follows either a preset trajectory (circle/square) or OU random motion.
    Visualization: top-down and first-person (human) via matplotlib.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, full_env_name: str, cfg=None, env_config=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Base parameters
        self.max_range = 20.0
        self.max_heading = np.pi
        
        # Reward shaping
        self.theta_reward = 33.0
        self.reward_min = 0.5
        self.reward_max = 2.5

        # Episode length 
        self.max_steps = getattr(env_config, 'max_steps', 200)  # 1000 steps * dt(=0.01s) = 10s

        # Action & observation spaces
        self.max_speed = 5.0
        self.max_w = np.deg2rad(30)
        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -self.max_speed, -self.max_w ], dtype=np.float32),   # v_forward, v_right, w
            high=np.array([ self.max_speed,  self.max_speed,  self.max_w ], dtype=np.float32),    
            dtype=np.float32                                                                    
        )
        obs_low  = np.array([0.0,               # r min
                             -1.0,              # sin(theta) min
                             -1.0,              # cos(theta) min
                            #  -self.max_heading, # heading_error min
                            ],    # V_right average
                            dtype=np.float32)
        obs_high = np.array([self.max_range,   # r max
                             1.0,              # sin(theta) max
                             1.0,              # cos(theta) max
                            #  self.max_heading, # heading_error max
                             ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def set_training_info(self, training_info: dict):
        """
        Callback from Sample Factory with training progress.
        Adjusts self.AP_factor on a linear schedule from init_AP to max_AP.
        """
        super().set_training_info(training_info)
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.array([1, sin(0), cos(0), 0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def _finalize_render(self):
        pass

    def close(self):
        pass
