import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import sin, cos
from sample_factory.envs.env_utils import TrainingInfoInterface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Quadcopter_SimCon.Simulation.ctrl import Control, ControlType
from Quadcopter_SimCon.Simulation.trajectory import Trajectory
from Quadcopter_SimCon.Simulation.quadFiles.quad import Quadcopter
from Quadcopter_SimCon.Simulation.utils.windModel import Wind
from Quadcopter_SimCon.Simulation.utils.rotationConversion import *
from Quadcopter_SimCon.Simulation.utils import sameAxisAnimation
import matplotlib.pyplot as plt

from collections import deque

from copy import deepcopy


import csv, os, time, atexit
from datetime import datetime

def quad_sim(t, Ts, quad, ctrl, traj):
    
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, None)
    t += Ts

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj=traj, quad=quad, Ts=Ts)

    return t
    
class PointTrajectoryEnv(gym.Env, TrainingInfoInterface):
    """
    Gymnasium environment for a 2D drone tracking a moving target.
    Actions: velocities [v_forward, v_right].
    Observations: [pos_x_error, pos_y_error, vel_x, vel_y, int_x, int_y].
    Rewards: weighted error on position, velocity, and position integral.
    Target follows either a preset trajectory (circle/square) or OU random motion.
    Visualization: top-down and first-person (human) via matplotlib.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, full_env_name: str, cfg=None, env_config=None, render_mode=None):
        super().__init__()
        self.dt = 0.1   # time step for velocity contol
        self.render_mode = render_mode
        self.globalTime = 0.0
        self.quadTs = 0.01 # time step for quadcopter simulation

        # Base parameters
        self.max_range = 20.0
        self.radius_mean = 10.0
        self.radius_min = 0.0
        self.radius_max = 20.0
        # self.angle_min = -np.pi
        # self.angle_max = np.pi
        self.vf = 0.0
        self.vr = 0.0

        self.quad = Quadcopter(Ti=0)
        self.yawType = 0         # Select Yaw Trajectory Type      (0: none, 1: yaw_waypoint_timed, 2: yaw_waypoint_interp, 3: follow, 4: zero)
        self.ctrl = Control(self.quad, self.yawType)
        self.traj = Trajectory(quad=self.quad, ctrlType=ControlType.XY_VEL_Z_POS)
        
        # Reward shaping
        self.r_reward = 0.4
        self.epsilon_r = 0.4
        self.theta_reward = 33.0
        self.reward_min = 0.5
        self.reward_max = 2.5
        self.goal_factor = 100
        self.goal_radius = 0.1
        self.AP_factor = 1.0  # start with minimal penalty
        
        # Reward coefficients for new reward function
        self.cx = 8.0  # Position error coefficient
        self.cv = 0.4 # Velocity error coefficient
        self.cIx = 0.0  # Position integral error coefficient

        # Episode length 
        self.max_steps = getattr(env_config, 'max_steps', 300)  # 1000 steps * dt(=0.01s) = 10s
        # Position integral scale
        self.int_scale = self.max_range * self.max_steps *self.dt + 1e-6
        # Action & observation spaces
        self.max_speed = 5.0
        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -self.max_speed], dtype=np.float32),   # v_forward, v_right
            high=np.array([ self.max_speed,  self.max_speed], dtype=np.float32),    
            dtype=np.float32                                                                    
        )
        # Observation: [pos_x, pos_y, vel_x, vel_y, int_x, int_y]
        # Position is relative to target (error), velocity is absolute, integral is accumulated error
        # Use large finite bounds for integral to avoid issues with np.inf
        max_integral = self.max_range * self.max_steps * 10  # Large bound for integral
        obs_low  = np.array([-self.max_range,   # pos_x error min
                             -self.max_range,   # pos_y error min
                             -self.max_speed,   # vel_x min
                             -self.max_speed,   # vel_y min
                            ], dtype=np.float32)
        obs_high = np.array([self.max_range,    # pos_x error max
                             self.max_range,    # pos_y error max
                             self.max_speed,    # vel_x max
                             self.max_speed,    # vel_y max
                             ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_angle = 0.0
        self.current_radius = 0.0
        self.step_count = 0
        # Position integral (accumulated position error)
        self.pos_integral = np.zeros(2, dtype=np.float32)
        # Action history
        self.history_length = 10
        self.vf_history = np.zeros(self.history_length)
        self.vr_history = np.zeros(self.history_length)
        self.vf_avg = 0.0
        self.vr_avg = 0.0
        self.hist_i = 0 
        self.lastAction = None

        # Debug parameters
        self.raw_vf = 0.0
        self.raw_vr = 0.0
        # Episode reward component accumulators for TensorBoard
        self.ep_stats = dict(pos_part=0.0, vel_part=0.0, int_part=0.0, total=0.0)

        # Trajectory type: 'circle', 'square', or 'random'
        self.trajectory_type = getattr(cfg, 'trajectory', 'random')
        if self.trajectory_type != 'random':
            self._generate_trajectory(self.trajectory_type)
            self.current_index = 0
        self.waypoint = (-10,0)

        self.theta = 0
        self.vf_cmd = 0
        self.vr_cmd = 0

        self.arrLen=self.max_steps*100
        self.t_all = deque(maxlen=self.arrLen)
        self.s_all = deque(maxlen=self.arrLen)
        self.pos_all = deque(maxlen=self.arrLen)
        self.vel_all = deque(maxlen=self.arrLen)
        self.quat_all = deque(maxlen=self.arrLen)
        self.omega_all = deque(maxlen=self.arrLen)
        self.euler_all = deque(maxlen=self.arrLen)
        self.sDes_traj_all = deque(maxlen=self.arrLen)
        self.sDes_calc_all = deque(maxlen=self.arrLen)
        self.w_cmd_all = deque(maxlen=self.arrLen)
        self.wMotor_all = deque(maxlen=self.arrLen)
        self.thr_all = deque(maxlen=self.arrLen)
        self.tor_all = deque(maxlen=self.arrLen)
        self.reward_all = deque(maxlen=self.arrLen)

        self.reset_count =0 # DEBUG iterator
        
        self.quadStateLog=np.zeros((10000,100))
        self.quadStateLogInd=0 
        self.indT=0; self.indrpy=[1, 2, 3]
        self.indpos=[4,5,6]; self.indquat=[7, 8, 9, 10]; self.indvel=[11, 12, 13]; self.indpqr=[14, 15, 16]
        self.indrpm=[17, 19, 21, 23]; self.indrpmdot=[18, 20, 22, 24]
        self.indpos_sp=[25, 26, 27];  self.indvel_sp=[28, 29, 30]; self.indacc_sp=[31, 32, 33]
        self.indthrust_sp=[34, 35, 36]; self.indeul_sp=[37, 38, 39]; self.indpqr_sp=[40, 41, 42]
        self.indyawFF=[43, 44, 45]; self.indrateCtrl=[46, 47, 48]; self.indqd_full=[49, 50, 51, 52]; self.indypr=[53, 54, 55]


    def set_training_info(self, training_info: dict):
        """
        Callback from Sample Factory with training progress.
        Adjusts self.AP_factor on a linear schedule from init_AP to max_AP.
        """
        super().set_training_info(training_info)
        pass

    def _generate_trajectory(self, traj_type: str):
        N = 75
        if traj_type == 'circle':
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            self.trajectory_points = np.stack([
                self.radius_mean * np.cos(angles),
                self.radius_mean * np.sin(angles)
            ], axis=1)
        elif traj_type == 'square':
            per = N // 4
            pts = []
            # bottom edge
            for i in range(per): pts.append((-self.radius_mean + 2*self.radius_mean*i/per, -self.radius_mean))
            # right edge
            for i in range(per): pts.append(( self.radius_mean, -self.radius_mean + 2*self.radius_mean*i/per))
            # top edge
            for i in range(per): pts.append(( self.radius_mean - 2*self.radius_mean*i/per,  self.radius_mean))
            # left edge
            for i in range(per): pts.append((-self.radius_mean,  self.radius_mean - 2*self.radius_mean*i/per))
            self.trajectory_points = np.array(pts, dtype=np.float32)
        elif traj_type == 'waypoint' or traj_type == 'waypoint2':
            waypoint = (-10,0)
            self.trajectory_points = np.full((N, 2), waypoint, dtype=np.float32)
        else:
            raise ValueError(f"Unknown trajectory: {traj_type}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset agent
        self.agent_pos[:] = 0.0
        self.agent_heading = 0.0

        self.quad = Quadcopter(Ti=0)
        self.globalTime = 0.0
        self.ctrl = Control(self.quad, self.yawType)

        # Reset per-episode accumulators for TB
        self.ep_stats = dict(pos_part=0.0, vel_part=0.0, int_part=0.0, total=0.0)
        
        # Reset position integral
        self.pos_integral[:] = 0.0

        self.reset_count += 1
        
        # Initialize target
        if self.trajectory_type == 'waypoint':
            x =  np.random.uniform(-15.0,15.0)
            y =  np.random.uniform(-15.0,15.0)
            self.target_pos = (x,y)
            self.agent_heading = np.random.uniform(-np.pi,np.pi)
        elif self.trajectory_type == 'waypoint2':
            x = np.random.uniform(-1.0,1.0)
            y = np.random.uniform(-1.0,1.0)
            self.target_pos = (x,y)

        # Initial obs
        self.quad.extended_state()
        obs = self._get_obs()
        self.step_count = 0
        if self.render_mode == 'human':
            self._init_render()

        return obs, {}

    def _get_obs(self):
        """
        Returns observation: [pos_x_error, pos_y_error, vel_x, vel_y, int_x, int_y]
        - Position error is relative to target (target - current)
        - Velocity is absolute 2D velocity
        - Position integral is accumulated position error
        """
        self.quad.extended_state()
        
        # Position error (target - current position)
        pos_error = np.array(self.target_pos) - self.quad.pos[:2]
        
        # Velocity (2D)
        vel = self.quad.vel[:2]
        
        # Combine into observation
        obs = np.concatenate([pos_error, vel], dtype=np.float32)
        
        return obs

    def step(self, action):
        # populate raw actions for DEBUG
        self.lastAction = action
        self.raw_vf, self.raw_vr = action
        # Agent move
        self.vf, self.vr = np.clip(action, self.action_space.low, self.action_space.high)

        if self.globalTime>4:
            pass   #plt.plot(self.quadStateLog[:self.quadStateLogInd, self.indT], self.quadStateLog[:self.quadStateLogInd,self.indvel_sp[0]])

        desired = {'pos': [np.nan, np.nan, 0], 
                   'vel': [self.vf, self.vr, 0.0], 
                   'yaw_rate': 0.0}
        globalTime = 0

        for ind in range(int(self.dt/self.quadTs)):
            if self.globalTime > 0.3 and self.globalTime < 0.45:
                pass
            self.traj.desiredState(t=globalTime, Ts=self.quadTs, quad=self.quad, desired=desired)

            quad_sim(t=self.globalTime+ind*self.quadTs, Ts=self.quadTs, quad=self.quad, ctrl=self.ctrl, traj=self.traj)

        # [self.quad.pos, self.quad.vel, self.ctrl.w_cmd, action]
        self.globalTime += self.dt
        # self.quadStateLog[self.quadStateLogInd, self.indT] = self.globalTime
        # self.quadStateLog[self.quadStateLogInd, self.indrpy] = self.quad.psi,self.quad.theta,self.quad.phi
        # self.quadStateLog[self.quadStateLogInd, self.indpos] = self.quad.state[0:3]    # self.pos
        # self.quadStateLog[self.quadStateLogInd, self.indquat] = self.quad.state[3:7]   # self.quat
        # self.quadStateLog[self.quadStateLogInd, self.indvel] = self.quad.state[7:10]   # self.vel
        # self.quadStateLog[self.quadStateLogInd, self.indpqr] = self.quad.state[10:13]  # self.omega 
        # self.quadStateLog[self.quadStateLogInd, self.indrpm] = np.array([self.quad.state[13], self.quad.state[15], self.quad.state[17], self.quad.state[19]]) #self.quad.wMotor
        # self.quadStateLog[self.quadStateLogInd, self.indrpmdot] = np.array([self.quad.state[14], self.quad.state[16], self.quad.state[18], self.quad.state[20]]) #self.quad.wMotor_dot
        # self.quadStateLog[self.quadStateLogInd, self.indpos_sp] = self.ctrl.pos_sp
        # self.quadStateLog[self.quadStateLogInd, self.indvel_sp] = self.ctrl.vel_sp
        # self.quadStateLog[self.quadStateLogInd, self.indacc_sp] = self.ctrl.acc_sp
        # self.quadStateLog[self.quadStateLogInd, self.indthrust_sp] = self.ctrl.thrust_sp
        # self.quadStateLog[self.quadStateLogInd, self.indeul_sp] = self.ctrl.eul_sp
        # self.quadStateLog[self.quadStateLogInd, self.indpqr_sp] = self.ctrl.pqr_sp
        # self.quadStateLog[self.quadStateLogInd, self.indyawFF] = self.ctrl.yawFF
        # self.quadStateLog[self.quadStateLogInd, self.indrateCtrl] = self.ctrl.rateCtrl
        # self.quadStateLog[self.quadStateLogInd, self.indqd_full] = self.ctrl.qd_full
        # self.quadStateLog[self.quadStateLogInd, self.indypr] = quatToYPR_ZYX(self.ctrl.qd_full)
        self.quadStateLogInd += 1
        
        # Get current state
        self.quad.extended_state()
        
        # Position error (target - current position)
        pos_error = np.array(self.target_pos) - self.quad.pos[:2]
        
        # Desired velocity (should be zero for waypoint tracking)
        desired_vel = np.zeros(2)
        # Velocity error (desired - current)
        vel_error = desired_vel - self.quad.vel[:2]
        
        # Update position integral (accumulate position error)
        self.pos_integral += pos_error * self.dt
        
        # Normalize errors for reward calculation
        # Position error normalized by max_range
        e_xnorm = np.linalg.norm(pos_error) / self.max_range
        
        # Velocity error normalized by max_speed
        e_vnorm = np.linalg.norm(vel_error) / self.max_speed
        
        # Position integral error normalized (using a reasonable scale)
        # Use max_range * max_steps as normalization factor for integral
        e_ixnorm = np.linalg.norm(self.pos_integral) / self.int_scale   # Add small epsilon to avoid division by zero
        
        # Reward: -c1*e_xnorm - c2*e_vnorm - c3*e_ixnorm
        pos_part = -self.cx * e_xnorm
        vel_part = -self.cv * e_vnorm
        goal_part = 1.0 if np.linalg.norm(pos_error) < self.goal_radius else 0.0
        reward = pos_part + vel_part + goal_part
        # Observation
        obs = self._get_obs()
        
        # Accumulate per-episode reward components for TensorBoard
        self.ep_stats['pos_part'] += float(pos_part)
        self.ep_stats['vel_part'] += float(vel_part)
        self.ep_stats['total'] += float(reward)


        self.vf_cmd = self.vf
        self.vr_cmd = self.vr
  
        # Termination
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # Compute distance for info
        r = np.linalg.norm(pos_error)
        info = {"pos_error": float(r), "vel_error": float(np.linalg.norm(vel_error)), 
                "int_error": float(np.linalg.norm(self.pos_integral))}
        
        # Attach episode-end stats so Sample Factory logs to TensorBoard
        if truncated:
            steps = max(1, self.step_count)
            info['episode_extra_stats'] = {
                'R_ep/pos_part_sum': float(self.ep_stats['pos_part']),
                'R_ep/vel_part_sum': float(self.ep_stats['vel_part']),
                'R_ep/int_part_sum': float(self.ep_stats['int_part']),
                'R_ep/total_sum': float(self.ep_stats['total']),
            }
        if self.render_mode=='human':
            self.reward_all.append(np.array([pos_part, vel_part, pos_part + vel_part, reward]))
            if truncated: self._finalize_render()
        return obs, reward, False, truncated, info

    def _init_render(self):
        # Initialize deques for rendering
        self.t_all = deque(maxlen=self.arrLen)
        self.s_all = deque(maxlen=self.arrLen)
        self.pos_all = deque(maxlen=self.arrLen)
        self.vel_all = deque(maxlen=self.arrLen)
        self.quat_all = deque(maxlen=self.arrLen)
        self.omega_all = deque(maxlen=self.arrLen)
        self.euler_all = deque(maxlen=self.arrLen)
        self.sDes_traj_all = deque(maxlen=self.arrLen)
        self.sDes_calc_all = deque(maxlen=self.arrLen)
        self.w_cmd_all = deque(maxlen=self.arrLen)
        self.wMotor_all = deque(maxlen=self.arrLen)
        self.thr_all = deque(maxlen=self.arrLen)
        self.tor_all = deque(maxlen=self.arrLen)
        self.reward_all = deque(maxlen=self.arrLen)


    def render(self, mode='human'):
        if mode!='human': return
        
        self.t_all.append(deepcopy(self.globalTime))
        self.s_all.append(deepcopy(self.quad.state))
        self.pos_all.append(deepcopy(self.quad.pos))
        self.vel_all.append(deepcopy(self.quad.vel))
        self.quat_all.append(deepcopy(self.quad.quat)) 
        self.omega_all.append(deepcopy(self.quad.omega))
        self.euler_all.append(deepcopy(self.quad.euler))
        self.sDes_calc_all.append(deepcopy(self.ctrl.sDesCalc))
        self.w_cmd_all.append(deepcopy(self.ctrl.w_cmd))
        self.wMotor_all.append(deepcopy(self.quad.wMotor))
        self.thr_all.append(deepcopy(self.quad.thr))
        self.tor_all.append(deepcopy(self.quad.tor))
        # self.reward_all accumulates at the end of step()

    def _finalize_render(self):
        '''Convert logs to arrays and draw figures via display.makeFigures.'''
        try:
            from Quadcopter_SimCon.Simulation.utils.display import makeFigures
        except Exception as e:
            print(f"display.py not found or import failed: {e}")
            return

        if len(self.t_all) < 2:
            return

        time_arr     = np.asarray(self.t_all, dtype=float)
        pos_all      = np.asarray(self.pos_all)
        vel_all      = np.asarray(self.vel_all)
        quat_all     = np.asarray(self.quat_all)
        omega_all    = np.asarray(self.omega_all)
        euler_all    = np.asarray(self.euler_all)
        sDes_traj    = np.asarray(self.sDes_traj_all) if len(self.sDes_traj_all) else np.zeros((len(time_arr), 19))
        sDes_calc    = np.asarray(self.sDes_calc_all)
        commands     = np.asarray(self.w_cmd_all)
        wMotor_all   = np.asarray(self.wMotor_all)
        thrust_all   = np.asarray(self.thr_all)
        torque_all   = np.asarray(self.tor_all)
        reward_all   = np.asarray(self.reward_all)

        try:
            makeFigures({}, time_arr, pos_all, vel_all, quat_all, omega_all, euler_all,
                        commands, wMotor_all, thrust_all, torque_all, sDes_traj, sDes_calc,reward_all)
            
            print(f"Target position: {self.target_pos}")
            plt.show(block=False)
            self.step(self.lastAction)
            plt.pause(0.001)
        except Exception as e:
            print(f"Target position: {self.target_pos}")
            plt.plot(time_arr, pos_all[:,0]);plt.plot(time_arr, pos_all[:,1])
            plt.plot(time_arr, vel_all[:,0]);plt.plot(time_arr, vel_all[:,1])
            plt.show()
            print(f"makeFigures failed: {e}")

    def close(self):
        if self.render_mode == "human":
            self._finalize_render()
