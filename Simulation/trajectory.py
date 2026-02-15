# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# Functions get_poly_cc, minSomethingTraj, pos_waypoint_min are derived from Peter Huang's work:
# https://github.com/hbd730/quadcopter-simulation
# author: Peter Huang
# email: hbd730@gmail.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!

import matplotlib.pyplot as plt

import numpy as np
from numpy import pi
from numpy.linalg import norm

# Support both relative and absolute imports
try:
    from .waypoints import makeWaypoints
    from . import config
    from .ctrl import ControlType
except ImportError:
    from waypoints import makeWaypoints
    import config
    from ctrl import ControlType

from enum import Enum


class PositionTrajectoryType(Enum):
    """Position trajectory types for trajSelect[0] / xyzType."""
    HOVER = 0
    POS_WAYPOINT_TIMED = 1
    POS_WAYPOINT_INTERP = 2
    MINIMUM_VELOCITY = 3
    MINIMUM_ACCEL = 4
    MINIMUM_JERK = 5
    MINIMUM_SNAP = 6
    MINIMUM_ACCEL_STOP = 7
    MINIMUM_JERK_STOP = 8
    MINIMUM_SNAP_STOP = 9
    MINIMUM_JERK_FULL_STOP = 10
    MINIMUM_SNAP_FULL_STOP = 11
    POS_WAYPOINT_ARRIVED = 12
    POS_WAYPOINT_ARRIVED_WAIT = 13


class YawTrajectoryType(Enum):
    """Yaw trajectory types for trajSelect[1] / yawType."""
    NONE = 0
    YAW_WAYPOINT_TIMED = 1
    YAW_WAYPOINT_INTERP = 2
    FOLLOW = 3
    ZERO = 4


class WaypointTimeMode(Enum):
    """Waypoint time calculation mode for trajSelect[2] / averVel."""
    WAYPOINT_TIME = 0
    AVERAGE_SPEED = 1


class Trajectory:

    def __init__(self, quads, ctrlType: ControlType, trajSelect=np.zeros(3)):

        self.numOfQuads = quads.numOfQuads
        self.maxThr = quads.params["maxThr"]  # Maximum total thrust [Nt] for converting percentage to physical units
        
        self.ctrlType = ctrlType
        # Convert trajSelect to enums if needed (accepts int or enum)
        xyz_val = trajSelect[0]
        yaw_val = trajSelect[1]
        aver_val = trajSelect[2]
        
        # Convert to enum if int, otherwise use enum directly
        if isinstance(xyz_val, (int, float, np.integer)):
            self.xyzType = PositionTrajectoryType(int(xyz_val))
        elif isinstance(xyz_val, PositionTrajectoryType):
            self.xyzType = xyz_val
        else:
            self.xyzType = PositionTrajectoryType(int(xyz_val))
            
        if isinstance(yaw_val, (int, float, np.integer)):
            self.yawType = YawTrajectoryType(int(yaw_val))
        elif isinstance(yaw_val, YawTrajectoryType):
            self.yawType = yaw_val
        else:
            self.yawType = YawTrajectoryType(int(yaw_val))
            
        if isinstance(aver_val, (int, float, np.integer)):
            self.averVel = WaypointTimeMode(int(aver_val))
        elif isinstance(aver_val, WaypointTimeMode):
            self.averVel = aver_val
        else:
            self.averVel = WaypointTimeMode(int(aver_val))

        self.t_wps, self.wps, self.y_wps, self.v_wp = makeWaypoints(self.numOfQuads)
        self.end_reached = 0

        if (self.ctrlType == ControlType.XYZ_POS):
            self.T_segment = np.diff(self.t_wps)

            if (self.averVel == WaypointTimeMode.AVERAGE_SPEED):
                distance_segment = self.wps[3:,:] - self.wps[:-3,:]
                dist = np.sqrt(distance_segment[0::3,:]**2 + distance_segment[1::3,:]**2 + distance_segment[2::3,:]**2)
                self.T_segment = dist*(np.outer(np.ones(dist.shape[0]), 1/self.v_wp))
                self.t_wps = np.zeros((self.T_segment.shape[0] + 1, self.numOfQuads))
                self.t_wps[1:,:] = np.cumsum(self.T_segment, axis=0)
            
            _min_vel_acc_jerk_snap = (PositionTrajectoryType.MINIMUM_VELOCITY, PositionTrajectoryType.MINIMUM_ACCEL, PositionTrajectoryType.MINIMUM_JERK,  PositionTrajectoryType.MINIMUM_SNAP)
            _min_stop = (PositionTrajectoryType.MINIMUM_ACCEL_STOP, PositionTrajectoryType.MINIMUM_JERK_STOP, PositionTrajectoryType.MINIMUM_SNAP_STOP)
            _min_full_stop = (PositionTrajectoryType.MINIMUM_JERK_FULL_STOP, PositionTrajectoryType.MINIMUM_SNAP_FULL_STOP)
            # wps layout: rows 0,3,6,... = x of wp0,wp1,...; rows 1,4,7,... = y; rows 2,5,8,... = z
            wps_x = self.wps[0::3, 0].flatten()
            wps_y = self.wps[1::3, 0].flatten()
            wps_z = self.wps[2::3, 0].flatten()
            T_seg = self.T_segment[:, 0].flatten() if self.T_segment.ndim > 1 else self.T_segment.flatten()
            if self.xyzType in _min_vel_acc_jerk_snap:
                self.deriv_order = self.xyzType.value - PositionTrajectoryType.MINIMUM_VELOCITY.value + 1  # 1=vel, 2=acc, 3=jerk, 4=snap
                self.coeff_x = minSomethingTraj(wps_x, T_seg, self.deriv_order)
                self.coeff_y = minSomethingTraj(wps_y, T_seg, self.deriv_order)
                self.coeff_z = minSomethingTraj(wps_z, T_seg, self.deriv_order)
            elif self.xyzType in _min_stop:
                self.deriv_order = self.xyzType.value - PositionTrajectoryType.MINIMUM_ACCEL_STOP.value + 2  # 2=acc, 3=jerk, 4=snap
                self.coeff_x = minSomethingTraj_stop(wps_x, T_seg, self.deriv_order)
                self.coeff_y = minSomethingTraj_stop(wps_y, T_seg, self.deriv_order)
                self.coeff_z = minSomethingTraj_stop(wps_z, T_seg, self.deriv_order)
            elif self.xyzType in _min_full_stop:
                self.deriv_order = self.xyzType.value - PositionTrajectoryType.MINIMUM_JERK_FULL_STOP.value + 3  # 3=jerk, 4=snap
                self.coeff_x = minSomethingTraj_faststop(wps_x, T_seg, self.deriv_order)
                self.coeff_y = minSomethingTraj_faststop(wps_y, T_seg, self.deriv_order)
                self.coeff_z = minSomethingTraj_faststop(wps_z, T_seg, self.deriv_order)
        
        if (self.yawType == YawTrajectoryType.ZERO):
            self.y_wps = np.zeros(len(self.t_wps))
        
        # Get initial heading
        self.current_heading = quads.psi
        
        # Initialize trajectory setpoint
        self.desPos = np.zeros((3, self.numOfQuads))    # Desired position (x, y, z)
        self.desVel = np.zeros((3, self.numOfQuads))    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros((3, self.numOfQuads))    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros((3, self.numOfQuads))    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros((3, self.numOfQuads))    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros((3, self.numOfQuads))    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = np.zeros((1, self.numOfQuads))         # Desired yaw speed
        self.sDes = np.concatenate([self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate], axis=0).astype(float)

    def _pos_waypoint_timed(self, t):
        if not (self.t_wps.shape[0] == self.wps[::3,:].shape[0]):
            raise Exception("Time array and waypoint array not the same size.")
        t_time = self.t_wps[:, 0].flatten()
        if (np.diff(t_time) <= 0).any():
            raise Exception("Time array isn't properly ordered.")
        if (t == 0):
            self.desPos = self.wps[0:3,:]
        elif (t >= t_time[-1]):
            self.desPos = self.wps[-3:,:]
        else:
            self.t_idx = np.where(t <= t_time)[0][0]
            self.desPos = self.wps[self.t_idx*3:self.t_idx*3+3,:]

    def _pos_waypoint_interp(self, t):
        if not (self.t_wps.shape[0] == self.wps[::3,:].shape[0]):
            raise Exception("Time array and waypoint array not the same size.")
        t_time = self.t_wps[:, 0].flatten()
        if (np.diff(t_time) <= 0).any():
            raise Exception("Time array isn't properly ordered.")
        if (t == 0):
            self.desPos = self.wps[0:3,:]
        elif (t >= self.t_wps[-1, 0]):
            self.desPos = self.wps[-3:,:]
        else:
            self.t_idx = np.where(t <= t_time)[0][0] - 1
            scale = (t - self.t_wps[self.t_idx, 0]) / self.T_segment[self.t_idx, 0]
            # wps layout: rows 0..2=wp0 xyz, 3..5=wp1 xyz, ...
            self.desPos = (1 - scale) * self.wps[self.t_idx*3:self.t_idx*3+3, :] + scale * self.wps[(self.t_idx+1)*3:(self.t_idx+2)*3, :]

    def _pos_waypoint_min(self, t):
        """Minimum velocity/accel/jerk/snap trajectory through waypoints."""
        if not (self.t_wps.shape[0] == self.wps[::3,:].shape[0]):
            raise Exception("Time array and waypoint array not the same size.")
        nb_coeff = self.deriv_order*2
        if t == 0:
            self.t_idx = 0
            self.desPos = self.wps[0:3,:]
        elif (t >= self.t_wps[-1].max()):
            self.t_idx = self.wps.shape[0] // 3 - 1
            self.desPos = self.wps[-3:,:]
        else:
            self.t_idx = np.where(t <= self.t_wps[:, 0])[0][0] - 1
            scale = (t - self.t_wps[self.t_idx, 0])
            start = nb_coeff * self.t_idx
            end = nb_coeff * (self.t_idx + 1)
            t0 = get_poly_cc(nb_coeff, 0, scale)
            pos_1d = np.array([self.coeff_x[start:end].dot(t0), self.coeff_y[start:end].dot(t0), self.coeff_z[start:end].dot(t0)])
            self.desPos[:, 0] = pos_1d
            if self.numOfQuads > 1:
                for i in range(1, self.numOfQuads):
                    self.desPos[:, i] = pos_1d
            t1 = get_poly_cc(nb_coeff, 1, scale)
            vel_1d = np.array([self.coeff_x[start:end].dot(t1), self.coeff_y[start:end].dot(t1), self.coeff_z[start:end].dot(t1)])
            self.desVel[:, 0] = vel_1d
            if self.numOfQuads > 1:
                for i in range(1, self.numOfQuads):
                    self.desVel[:, i] = vel_1d
            t2 = get_poly_cc(nb_coeff, 2, scale)
            acc_1d = np.array([self.coeff_x[start:end].dot(t2), self.coeff_y[start:end].dot(t2), self.coeff_z[start:end].dot(t2)])
            self.desAcc[:, 0] = acc_1d
            if self.numOfQuads > 1:
                for i in range(1, self.numOfQuads):
                    self.desAcc[:, i] = acc_1d

    def _pos_waypoint_arrived(self, t, quads):
        dist_consider_arrived = 0.2
        n_wp = self.wps.shape[0] // 3
        pos0 = np.atleast_2d(quads.pos)[0, :]
        if (t == 0):
            self.t_idx = 0
            self.end_reached = 0
        elif not(self.end_reached):
            # wps layout: rows t_idx*3, t_idx*3+1, t_idx*3+2 = x,y,z of waypoint t_idx
            wp = self.wps[self.t_idx*3:self.t_idx*3+3, 0]
            distance_to_next_wp = np.sqrt((wp[0]-pos0[0])**2 + (wp[1]-pos0[1])**2 + (wp[2]-pos0[2])**2)
            if (distance_to_next_wp < dist_consider_arrived):
                self.t_idx += 1
                if (self.t_idx >= n_wp):
                    self.end_reached = 1
                    self.t_idx = -1
        # waypoint position: rows t_idx*3..t_idx*3+2, or last 3 rows when t_idx == -1
        wp_pos = self.wps[-3:, :] if self.t_idx == -1 else self.wps[self.t_idx*3:self.t_idx*3+3, :]
        self.desPos[:, 0] = wp_pos[:, 0]
        if self.numOfQuads > 1:
            for i in range(1, self.numOfQuads):
                self.desPos[:, i] = wp_pos[:, i] if wp_pos.shape[1] > i else wp_pos[:, 0]

    def _pos_waypoint_arrived_wait(self, t, quads):
        dist_consider_arrived = 0.2
        n_wp = self.wps.shape[0] // 3
        pos0 = np.atleast_2d(quads.pos)[0, :]
        if (t == 0):
            self.t_idx = 0
            self.t_arrived = 0
            self.arrived = True
            self.end_reached = 0
        elif not(self.end_reached):
            wp = self.wps[self.t_idx*3:self.t_idx*3+3, 0]
            distance_to_next_wp = np.sqrt((wp[0]-pos0[0])**2 + (wp[1]-pos0[1])**2 + (wp[2]-pos0[2])**2)
            if (distance_to_next_wp < dist_consider_arrived) and not self.arrived:
                self.t_arrived = t
                self.arrived = True
            elif self.arrived and (t-self.t_arrived > self.t_wps[self.t_idx, 0]):
                self.t_idx += 1
                self.arrived = False
                if (self.t_idx >= n_wp):
                    self.end_reached = 0
                    self.t_idx = 0
        wp_pos = self.wps[-3:, :] if self.t_idx == -1 else self.wps[self.t_idx*3:self.t_idx*3+3, :]
        self.desPos[:, 0] = wp_pos[:, 0]
        if self.numOfQuads > 1:
            for i in range(1, self.numOfQuads):
                self.desPos[:, i] = wp_pos[:, i] if wp_pos.shape[1] > i else wp_pos[:, 0]

    def _yaw_waypoint_timed(self, t):
        if not (len(self.t_wps) == len(self.y_wps)):
            raise Exception("Time array and waypoint array not the same size.")
        self.desEul[2] = self.y_wps[self.t_idx]

    def _yaw_waypoint_interp(self, t, Ts):
        if not (len(self.t_wps) == len(self.y_wps)):
            raise Exception("Time array and waypoint array not the same size.")
        if (t == 0) or (t >= self.t_wps[-1].max()):
            self.desEul[2] = self.y_wps[self.t_idx]
        else:
            scale = (t - self.t_wps[self.t_idx, 0])/self.T_segment[self.t_idx, 0]
            self.desEul[2] = (1 - scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx + 1]
            delta_psi = self.desEul[2] - self.current_heading
            self.desYawRate = np.atleast_2d(delta_psi / Ts)
            self.current_heading = self.desEul[2]

    def _yaw_follow(self, t, Ts, quads):
        pos0 = np.atleast_2d(quads.pos)[0, :]
        # First quad desired position as (3,) x,y,z; desPos can be (3,n), (3,), or (1,3)
        d = np.asarray(self.desPos).flatten()
        des_pos0 = d[:3] if len(d) >= 3 else np.pad(d, (0, 3 - len(d)), constant_values=0)
        if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED or
            self.xyzType == PositionTrajectoryType.POS_WAYPOINT_INTERP or
            self.xyzType == PositionTrajectoryType.POS_WAYPOINT_ARRIVED):
            if (t == 0):
                self.desEul[2] = 0
            else:
                self.desEul[2] = np.arctan2(des_pos0[1]-pos0[1], des_pos0[0]-pos0[0])
        elif (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_ARRIVED_WAIT):
            if (t == 0):
                self.desEul[2] = 0
                self.prevDesYaw = self.desEul[2]
            else:
                if not (self.arrived):
                    self.desEul[2] = np.arctan2(des_pos0[1]-pos0[1], des_pos0[0]-pos0[0])
                    self.prevDesYaw = self.desEul[2]
                else:
                    self.desEul[2] = self.prevDesYaw
        else:
            if (t == 0) or (t >= self.t_wps[-1].max()):
                self.desEul[2] = self.y_wps[self.t_idx]
            else:
                self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
        sign_changed = (np.sign(self.desEul[2]) != np.sign(self.current_heading))
        large_diff = (np.abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1)
        condition = sign_changed & large_diff
        if condition.any():
            self.current_heading[condition] = self.current_heading[condition] + np.sign(self.desEul[2][condition]) * 2*pi
        delta_psi = self.desEul[2] - self.current_heading
        self.desYawRate = np.atleast_2d(delta_psi / Ts)
        self.current_heading = self.desEul[2]

    def desiredState(self, t, Ts, quads, desired=None):
        
        self.desPos = np.zeros((3, self.numOfQuads))    # Desired position (x, y, z)
        self.desVel = np.zeros((3, self.numOfQuads))    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros((3, self.numOfQuads))    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros((3, self.numOfQuads))    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros((3, self.numOfQuads))    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros((3, self.numOfQuads))    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = np.zeros((1, self.numOfQuads))         # Desired yaw speed

        if (self.ctrlType == ControlType.XYZ_VEL):
            if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED):
                self.sDes = testVelControl(t, ControlType.XYZ_VEL, self.ulgData)

        elif (self.ctrlType == ControlType.XY_VEL_Z_POS):
            if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED):
                self.sDes = testVelControl(t, ControlType.XY_VEL_Z_POS, self.ulgData)
            elif (self.xyzType == PositionTrajectoryType.HOVER and desired is not None):
                # Trajectory for Desired States (e.g. from system_manager)
                # sDes must be (19, numOfQuads) for ctrl.controller indexing
                desPos = np.array([desired['pos'][0], desired['pos'][1], desired['pos'][2]])
                v = desired['vel']
                desVel = np.array([v[0], v[1], v[2] if len(v) > 2 else 0.0])
                desAcc = np.zeros(3)
                desThr = np.zeros(3)
                desEul = np.zeros(3)
                desPQR = np.zeros(3)
                desYawRate = desired['yaw_rate']
                vec = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, np.atleast_1d(desYawRate))).astype(float)
                # Broadcast to (19, numOfQuads)
                self.sDes = np.tile(vec.reshape(-1, 1), (1, self.numOfQuads))
                pass

        elif (self.ctrlType == ControlType.SYSTEM_MANAGER):
            # Desired state from system_manager (vel + yaw_rate; pos used for altitude hold only).
            if desired is not None:
                desPos = np.array([desired['pos'][0], desired['pos'][1], desired['pos'][2]])
                v = desired['vel']
                desVel = np.array([v[0], v[1], v[2] if len(v) > 2 else 0.0])
                desAcc = np.zeros(3)
                desThr = np.zeros(3)
                desEul = np.zeros(3)
                desPQR = np.zeros(3)
                desYawRate = desired['yaw_rate']
                vec = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, np.atleast_1d(desYawRate))).astype(float)
                self.sDes = np.tile(vec.reshape(-1, 1), (1, self.numOfQuads))
            # else: sDes remains zeros from initialization above

        elif (self.ctrlType == ControlType.ATT):
            # Attitude target mode: angles + thrust, rates calculated by attitude_control
            if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED):
                self.sDes = testAttControl(t, self.ulgData, self.maxThr)
        elif (self.ctrlType == ControlType.ATT_RATE):
            # Attitude rate target mode: rates + thrust, bypasses attitude_control
            if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED):
                self.sDes = testAttRateControl(t, self.ulgData, self.maxThr)
        
        elif (self.ctrlType == ControlType.XYZ_POS):
            # Hover at [0, 0, 0]
            if (self.xyzType == PositionTrajectoryType.HOVER):
                pass 
            # For simple testing
            elif (self.xyzType.value == 99):
                self.sDes = testXYZposition(t)   
            else:    
                # List of possible position trajectories
                # ---------------------------
                if (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_TIMED):
                    self._pos_waypoint_timed(t)
                elif (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_INTERP):
                    self._pos_waypoint_interp(t)
                elif (self.xyzType.value >= PositionTrajectoryType.MINIMUM_VELOCITY.value and 
                      self.xyzType.value <= PositionTrajectoryType.MINIMUM_SNAP_FULL_STOP.value):
                    self._pos_waypoint_min(t)
                elif (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_ARRIVED):
                    self._pos_waypoint_arrived(t, quads)
                elif (self.xyzType == PositionTrajectoryType.POS_WAYPOINT_ARRIVED_WAIT):
                    self._pos_waypoint_arrived_wait(t, quads)
                
                # List of possible yaw trajectories
                # ---------------------------
                if (self.yawType == YawTrajectoryType.NONE):
                    pass
                elif (self.yawType == YawTrajectoryType.YAW_WAYPOINT_TIMED):
                    self._yaw_waypoint_timed(t)
                elif (self.yawType == YawTrajectoryType.YAW_WAYPOINT_INTERP):
                    self._yaw_waypoint_interp(t, Ts)
                elif (self.yawType == YawTrajectoryType.FOLLOW):
                    self._yaw_follow(t, Ts, quads)

                self.sDes = np.concatenate((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate), axis=0).astype(float)
        
        return self.sDes


def get_poly_cc(n, k, t):
    """ This is a helper function to get the coeffitient of coefficient for n-th
        order polynomial with k-th derivative at time t.
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D  = np.linspace(n-1, 0, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc


def minSomethingTraj(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum velocity, order = 1. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.

    3.Matrix Design
    Since we have M*N unknown coefficients to solve, and if we are given M*N
    equations(constraints), then the problem becomes solving a linear equation.

    A * Coeff = B

    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are M*N constraints,
    so B matrix will be [M*N, 1].

    Coeff is the final output matrix consists of M*N elements. 
    Since B matrix is only one column, Coeff matrix must be [M*N, 1].

    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [M*N, 1], and B is [M*N, 1], thus
    A matrix must have the form [M*N, M*N].

    A = [A10 A11 ... A1M A20 A21 ... A2M ... AN0 AN1 ... ANM
        ...
        ]

    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...(M-1).
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting position derivatives (up to order) are null
    for k in range(1, order):
        A[2*n+k-1][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 4 - Ending position derivatives (up to order) are null
    for k in range(1, order):
        A[2*n+(order-1)+k-1][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])
    
    # Constraint 5 - All derivatives are continuous at each waypint transition
    for i in range(n-1):
        for k in range(1, nb_coeff-1):
            A[2*n+2*(order-1) + i*2*(order-1)+k-1][i*nb_coeff : (i*nb_coeff+nb_coeff*2)] = np.concatenate((get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))
    
    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff


# Minimum acceleration/jerk/snap Trajectory, but with null velocity, accel and jerk at each waypoint
def minSomethingTraj_stop(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum acceleration, order = 2. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    Unlike the function minSomethingTraj, where continuous equations for velocity, jerk and snap are generated, 
    this function generates trajectories with null velocities, accelerations and jerks at each waypoints. 
    This will make the drone stop for an instant at each waypoint.
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting position derivatives (up to order) for each segment are null
    for i in range(n):
        for k in range(1, order):
            A[2*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 4 - Ending position derivatives (up to order) for each segment are null
    for i in range(n):
        for k in range(1, order):
            A[2*n+(order-1)*n + k-1 + i*(order-1)][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, k, times[i])
    
    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff

# Minimum acceleration/jerk/snap Trajectory, but with null velocity only at each waypoint
def minSomethingTraj_faststop(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum acceleration, order = 2. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    Unlike the function minSomethingTraj, where continuous equations for velocity, jerk and snap are generated, 
    and unlike the function minSomethingTraj_stop, where velocities, accelerations and jerks are equal to 0 at each waypoint,
    this function generates trajectories with only null velocities. Accelerations and above derivatives are continuous. 
    This will make the drone stop for an instant at each waypoint, and then leave in the same direction it came from.
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i+1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        # print(i)
        A[i][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        # print(i+n)
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting velocity for every segment is null
    for i in range(n):
        # print(i+2*n)
        A[i+2*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, 0)

    # Constraint 4 - Ending velocity for every segment is null
    for i in range(n):
        # print(i+3*n)
        A[i+3*n][nb_coeff*i:nb_coeff*(i+1)] = get_poly_cc(nb_coeff, 1, times[i])

    # Constraint 5 - Starting position derivatives (above velocity and up to order) are null
    for k in range(2, order):
        # print(4*n + k-2)
        A[4*n+k-2][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 6 - Ending position derivatives (above velocity and up to order) are null
    for k in range(2, order):
        # print(4*n+(order-2) + k-2)
        A[4*n+k-2+(order-2)][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])

    # Constraint 7 - All derivatives above velocity are continuous at each waypint transition
    for i in range(n-1):
        for k in range(2, nb_coeff-2):
            # print(4*n+2*(order-2)+k-2+i*(nb_coeff-4))
            A[4*n+2*(order-2)+k-2+i*(nb_coeff-4)][i*nb_coeff : (i*nb_coeff+nb_coeff*2)] = np.concatenate((get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))
            

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

## Testing scripts

def testXYZposition(t):
    desPos = np.array([2., 2., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 30.0*pi/180])
    desPQR = np.array([0., 0., 0.])
    desYawRate = 0 #30.0*pi/180
    
    if t >= 1 and t < 4:
        desPos = np.array([2, 2, 1])
    elif t >= 4:
        desPos = np.array([2, -2, -2])
        desEul = np.array([0, 0, pi/3])
    
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)

    return sDes


def testVelControl(t, ctlType: ControlType, ulgdata=None):
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])
    desPQR = np.array([0., 0., 0.])
    desYawRate = 0.

    # Interpolate desired vx, vy and vz from ulgData setpoint fields if available
    if ulgdata is None:
        # Fallback to hardcoded values if ulgdata not provided
        if t >= 1 and t < 4:
            desVel = np.array([3, 2, 0])
        elif t >= 4:
            desVel = np.array([3, -1, 0])
    else:
        # Interpolate vx
        vx_data = ulgdata['vehicle_local_position_setpoint_vx']
        timestamps = vx_data['timestamp']
        values = vx_data['data']
        desVel[0] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        
        # Interpolate vy
        vy_data = ulgdata['vehicle_local_position_setpoint_vy']
        timestamps = vy_data['timestamp']
        values = vy_data['data']
        desVel[1] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        
        # Interpolate vz
        if ctlType == ControlType.XYZ_VEL:
            vz_data = ulgdata['vehicle_local_position_setpoint_vz']
            timestamps = vz_data['timestamp']
            values = vz_data['data']
            desVel[2] = np.interp(t, timestamps, values, left=0.0, right=0.0)
     
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    
    return sDes


def testAttControl(t, ulgdata=None, maxThr=None):
    """
    Attitude target mode: Inject attitude setpoints (roll, pitch, yaw) and thrust setpoints (x, y, z) from ulg data.
    Rate setpoints are calculated by the attitude_control function.
    This function interpolates the setpoints at time t and returns a desired state vector.
    
    Args:
        t: Current time
        ulgdata: Dictionary with ulg data containing setpoints
        maxThr: Maximum total thrust in Newtons (for converting percentage to physical units)
    """
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])
    desPQR = np.array([0., 0., 0.])  # Will be calculated by attitude_control
    desYawRate = 0.

    # Interpolate angle and thrust setpoints from ulgData if available
    if ulgdata is not None:
        # Interpolate roll setpoint
        roll_data = ulgdata['vehicle_attitude_setpoint_roll_body']
        timestamps = roll_data['timestamp']
        values = roll_data['data']
        desEul[0] = np.interp(t, timestamps, values, left=0.0, right=0.0)
    
        # Interpolate pitch setpoint
        pitch_data = ulgdata['vehicle_attitude_setpoint_pitch_body']
        timestamps = pitch_data['timestamp']
        values = pitch_data['data']
        desEul[1] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        
        # Interpolate yaw setpoint
        yaw_data = ulgdata['vehicle_attitude_setpoint_yaw_body']
        timestamps = yaw_data['timestamp']
        values = yaw_data['data']
        desEul[2] = np.interp(t, timestamps, values, left=0.0, right=0.0)
    
        # Interpolate thrust setpoint (x, y, z components) - convert from percentage to Newtons
        thrust_x_data = ulgdata['vehicle_thrust_setpoint_xyz[0]']
        timestamps = thrust_x_data['timestamp']
        values = thrust_x_data['data']
        # Convert from percentage (0-1) to Newtons
        desThr[0] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
        
        thrust_y_data = ulgdata['vehicle_thrust_setpoint_xyz[1]']
        timestamps = thrust_y_data['timestamp']
        values = thrust_y_data['data']
        desThr[1] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
    
        thrust_z_data = ulgdata['vehicle_thrust_setpoint_xyz[2]']
        timestamps = thrust_z_data['timestamp']
        values = thrust_z_data['data']
        desThr[2] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
     
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    
    return sDes


def testAttRateControl(t, ulgdata=None, maxThr=None):
    """
    Attitude rate target mode: Inject rate setpoints (p, q, r) and thrust setpoints (x, y, z) from ulg data.
    This bypasses attitude_control and directly injects rates to rate_control.
    This function interpolates the setpoints at time t and returns a desired state vector.
    
    Args:
        t: Current time
        ulgdata: Dictionary with ulg data containing setpoints
        maxThr: Maximum total thrust in Newtons (for converting percentage to physical units)
    """
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])  # Not used in rate target mode
    desPQR = np.array([0., 0., 0.])
    desYawRate = 0.

    # Interpolate rate and thrust setpoints from ulgData if available
    if ulgdata is not None:
        # Interpolate roll rate setpoint
        roll_rate_data = ulgdata['vehicle_rates_setpoint_roll']
        timestamps = roll_rate_data['timestamp']
        values = roll_rate_data['data']
        desPQR[0] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        
        # Interpolate pitch rate setpoint
        pitch_rate_data = ulgdata['vehicle_rates_setpoint_pitch']
        timestamps = pitch_rate_data['timestamp']
        values = pitch_rate_data['data']
        desPQR[1] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        
        # Interpolate yaw rate setpoint
        yaw_rate_data = ulgdata['vehicle_rates_setpoint_yaw']
        timestamps = yaw_rate_data['timestamp']
        values = yaw_rate_data['data']
        desPQR[2] = np.interp(t, timestamps, values, left=0.0, right=0.0)
        desYawRate = desPQR[2]  # Also set yaw rate feedforward
        
        # Interpolate thrust setpoint (x, y, z components) - convert from percentage to Newtons
        thrust_x_data = ulgdata['vehicle_thrust_setpoint_xyz[0]']
        timestamps = thrust_x_data['timestamp']
        values = thrust_x_data['data']
        # Convert from percentage (0-1) to Newtons
        desThr[0] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
        
        thrust_y_data = ulgdata['vehicle_thrust_setpoint_xyz[1]']
        timestamps = thrust_y_data['timestamp']
        values = thrust_y_data['data']
        desThr[1] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
    
        thrust_z_data = ulgdata['vehicle_thrust_setpoint_xyz[2]']
        timestamps = thrust_z_data['timestamp']
        values = thrust_z_data['data']
        desThr[2] = np.interp(t, timestamps, values, left=0.0, right=0.0) * maxThr
     
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    
    return sDes
