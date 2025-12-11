# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm

import sys
import os
simulation_path = os.path.join(os.path.dirname(__file__), 'Quadcopter_SimCon', 'Simulation')

# Support both relative and absolute imports
try:
    from . import utils
    from . import config
except ImportError:
    import utils
    import config
from copy import deepcopy
from enum import Enum

rad2deg = 180.0/pi
deg2rad = pi/180.0


class ControlType(Enum):
    """Enumeration of available control types."""
    XYZ_POS = 0      # Position control in x, y, z
    XY_VEL_Z_POS = 1 # Velocity control in x, y, position control in z
    XYZ_VEL = 2      # Velocity control in x, y, z
    ATT = 3          # Attitude target mode: angles + thrust, rates calculated
    ATT_RATE = 4     # Attitude rate target mode: rates + thrust, bypasses attitude_control

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
Py    = 1.0*.7
Px    = Py
Pz    = 1.0

pos_P_gain = np.array([Px, Py, Pz])

# Velocity P-D gains
Pxdot = 1.5
Dxdot = .25
Ixdot = 0.1
FFxdot = 0.0
FFdxdot = 0.25

yfactor = 0.7
Pydot = Pxdot*yfactor
Dydot = Dxdot*yfactor
Iydot = Ixdot*yfactor
FFydot = FFxdot*yfactor
FFdydot = FFdxdot*yfactor

Pzdot = 4.0*6*6
Dzdot = 0.5
Izdot = 5.0
FFzdot = 0.0
FFdzdot = 0.0

vel_P_gain = np.array([Pxdot, Pydot, Pzdot])
vel_D_gain = np.array([Dxdot, Dydot, Dzdot])
vel_I_gain = np.array([Ixdot, Iydot, Izdot])
vel_FF_gain = np.array([FFxdot, FFydot, FFzdot])
vel_FF_dot_gain = np.array([FFdxdot, FFdydot, FFdzdot])

# Attitude P gains
Pphi = 10*2
Ptheta = Pphi
Ppsi = 1.5
PpsiStrong = 8

att_P_gain = np.array([Pphi, Ptheta, Ppsi])

# Rate P-D gains
rateFactor = 0.5
Pp = 0.4*rateFactor
Dp = 0.005*2*rateFactor

Pq = Pp
Dq = Dp 

Pr = 1.0*0.25*1.2
Dr = 0.1*0.25*1.2

rate_P_gain = np.array([Pp, Pq, Pr])
rate_D_gain = np.array([Dp, Dq, Dr])

# Max Velocities
uMax = 15.0
vMax = 15.0
wMax = 15.0

velMax = np.array([uMax, vMax, wMax])
velMaxAll = 15.0

saturateVel_separetely = False

# Max tilt
tiltMax = 50.0*deg2rad

# Max Rate
pMax = 200.0*deg2rad
qMax = 200.0*deg2rad
rMax = 150.0*deg2rad

rateMax = np.array([pMax, qMax, rMax])


class Control:
    
    def __init__(self, quad, yawType):
        self.sDesCalc = np.zeros((16, quad.numOfQuads))
        self.w_cmd = np.ones((4, quad.numOfQuads))*quad.params["w_hover"]
        self.thr_int = np.zeros((3, quad.numOfQuads))
        if (yawType == 0):
            att_P_gain[2] = 0
        self.setYawWeight()
        self.pos_sp    = np.zeros((3, quad.numOfQuads))
        self.vel_sp    = np.zeros((3, quad.numOfQuads))
        self.acc_sp    = np.zeros((3, quad.numOfQuads))
        self.thrust_sp = np.zeros((3, quad.numOfQuads))
        self.eul_sp    = np.zeros((3, quad.numOfQuads))
        self.pqr_sp    = np.zeros((3, quad.numOfQuads))
        self.yawFF     = np.zeros(quad.numOfQuads)
        
        self.rate_sp = np.zeros((3, quad.numOfQuads))
        self.qd = np.zeros((4, quad.numOfQuads))
        
        self.prevVel_sp = np.zeros((3, quad.numOfQuads))
    
    def controller(self, traj, quads, Ts):

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp[:]    = traj.sDes[0:3,:]
        self.vel_sp[:]    = traj.sDes[3:6,:]
        self.acc_sp[:]    = traj.sDes[6:9,:]
        self.thrust_sp[:] = traj.sDes[9:12,:]
        self.eul_sp[:]    = traj.sDes[12:15,:]
        self.pqr_sp[:]    = traj.sDes[15:18,:]
        self.yawFF[:]     = traj.sDes[18,:]
        
        # Select Controller
        # ---------------------------
        if (traj.ctrlType == ControlType.XYZ_VEL):
            self.saturateVel()
            self.z_vel_control(quads, Ts)
            self.xy_vel_control(quads, Ts)
            self.thrustToAttitude(quads, Ts)
            self.attitude_control(quads, Ts)
            self.rate_control(quads, Ts)
        elif (traj.ctrlType == ControlType.XY_VEL_Z_POS):
            self.z_pos_control(quads, Ts)
            self.saturateVel()
            self.z_vel_control(quads, Ts)
            self.xy_vel_control(quads, Ts)
            self.thrustToAttitude(quads, Ts)
            self.attitude_control(quads, Ts)
            self.rate_control(quads, Ts)
        elif (traj.ctrlType == ControlType.ATT):
            # Attitude target mode: inject attitude (angles) and thrust setpoints
            # Rate setpoints are calculated by attitude_control from angle setpoints
            # Use thrust setpoint from trajectory (interpolated from ulg file if available)
            # Thrust setpoint from ulg is in body frame, transform to inertial frame using desired attitude
            # Compute qd_full from Euler angle setpoints (roll, pitch, yaw)
            # YPRToQuat expects (yaw, pitch, roll) = (psi, theta, phi)
            # eul_sp is [roll, pitch, yaw] = [phi, theta, psi]
            self.qd_full = np.zeros((4, quads.numOfQuads))
            for i in range(quads.numOfQuads):
                self.qd_full[:, i] = utils.YPRToQuat(self.eul_sp[2, i], self.eul_sp[1, i], self.eul_sp[0, i])
            # Convert desired quaternion to DCM for thrust transformation
            # Transform thrust setpoint from body frame to inertial frame using desired attitude
            thrust_sp_body = self.thrust_sp.copy()
            for i in range(quads.numOfQuads):
                dcm_desired = utils.quat2Dcm(self.qd_full[:, i])
                self.thrust_sp[:, i] = dcm_desired @ thrust_sp_body[:, i]
            
            # Use attitude control to convert angle setpoints to rate setpoints
            # This computes rate_sp from eul_sp
            self.attitude_control(quads, Ts)
            
            # Use rate control with the computed rate setpoints
            self.rate_control(quads, Ts)
            
        elif (traj.ctrlType == ControlType.ATT_RATE):
            # Attitude rate target mode: inject rate and thrust setpoints directly
            # Bypasses attitude_control, goes directly to rate_control
            # Use thrust setpoint from trajectory (interpolated from ulg file if available)
            # Thrust setpoint from ulg is in body frame, transform to inertial frame using desired attitude
            # Convert Euler angle setpoints to DCM for thrust transformation
            # YPRToQuat expects (yaw, pitch, roll) = (psi, theta, phi)
            # eul_sp is [roll, pitch, yaw] = [phi, theta, psi]

            # Transform thrust setpoint from body frame to inertial frame using current attitude
            thrust_sp_body = self.thrust_sp.copy()
            self.thrust_sp[:] = utils.quat2Dcm(quads.quat) @ thrust_sp_body
            
            # Set rate setpoint directly from trajectory (interpolated from ulg file)
            self.rate_sp[:] = self.pqr_sp[:]
            
            # Limit rate setpoint
            self.rate_sp = np.clip(self.rate_sp, -rateMax, rateMax)
            
            # Use rate control directly (bypasses attitude_control)
            self.rate_control(quads, Ts)
        elif (traj.ctrlType == ControlType.XYZ_POS):
            self.z_pos_control(quads, Ts)
            self.xy_pos_control(quads, Ts)
            self.saturateVel(quads)
            self.z_vel_control(quads, Ts)
            self.xy_vel_control(quads, Ts)
            self.thrustToAttitude(quads, Ts)
            self.attitude_control(quads, Ts)
            self.rate_control(quads, Ts)

        # Mixer
        # --------------------------- 
        self.w_cmd = utils.mixerFM(quads, norm(self.thrust_sp, axis=0), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3,:] = self.pos_sp
        self.sDesCalc[3:6,:] = self.vel_sp
        self.sDesCalc[6:9,:] = self.thrust_sp
        self.sDesCalc[9:13,:] = self.qd
        self.sDesCalc[13:16,:] = self.rate_sp
        
        self.prevVel_sp = deepcopy(self.vel_sp)

    def z_pos_control(self, quads, Ts):
       
        # Z Position Control
        # --------------------------- 
        pos_z_error = self.pos_sp[2] - quads.pos.T[2]
        self.vel_sp[2] += pos_P_gain[2]*pos_z_error
        
    
    def xy_pos_control(self, quads, Ts):

        # XY Position Control
        # --------------------------- 
        pos_xy_error = (self.pos_sp[0:2] - quads.pos.T[0:2])
        self.vel_sp[0:2] += (np.outer(pos_P_gain[0:2], np.ones(quads.numOfQuads)))*pos_xy_error
        
        
    def saturateVel(self, quads):

        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (saturateVel_separetely):
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            totalVel_sp = norm(self.vel_sp, axis=0)
            exceeds_limit = totalVel_sp > velMaxAll
            if exceeds_limit.any():
                # Normalize velocity for quads that exceed the limit
                # Divide by current magnitude and multiply by max allowed magnitude
                scale_factor = np.where(exceeds_limit, velMaxAll / totalVel_sp, 1.0)
                self.vel_sp = self.vel_sp * scale_factor[np.newaxis, :]


    def z_vel_control(self, quads, Ts):
        
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_sp_dot = (self.vel_sp - self.prevVel_sp)/Ts
        vel_z_error = self.vel_sp[2] - quads.vel.T[2]
        if (config.orient == "NED"):
            thrust_z_sp = vel_P_gain[2]*vel_z_error - vel_D_gain[2]*quads.vel_dot.T[2] + quads.params["mB"]*(self.acc_sp[2] - quads.params["g"]) + self.thr_int[2] + vel_FF_gain[2]*self.vel_sp[2] + vel_FF_dot_gain[2]*vel_sp_dot[2]
        elif (config.orient == "ENU"):
            thrust_z_sp = vel_P_gain[2]*vel_z_error - vel_D_gain[2]*quads.vel_dot.T[2] + quads.params["mB"]*(self.acc_sp[2] + quads.params["g"]) + self.thr_int[2] + vel_FF_gain[2]*self.vel_sp[2] + vel_FF_dot_gain[2]*vel_sp_dot[2]
        
        # Get thrust limits
        if (config.orient == "NED"):
            # The Thrust limits are negated and swapped due to NED-frame
            uMax = -quads.params["minThr"]
            uMin = -quads.params["maxThr"]
        elif (config.orient == "ENU"):
            uMax = quads.params["maxThr"]
            uMin = quads.params["minThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = ((thrust_z_sp >= uMax) & (vel_z_error >= 0.0)) | ((thrust_z_sp <= uMin) & (vel_z_error <= 0.0))

        # Calculate integral part
        # Only update integral for quads where anti-windup is not active
        update_int = ~stop_int_D
        if update_int.any():
            self.thr_int[2][update_int] += vel_I_gain[2]*vel_z_error[update_int]*Ts * quads.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2][update_int] = np.clip(self.thr_int[2][update_int], -quads.params["maxThr"], quads.params["maxThr"])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)
        pass
    
    def xy_vel_control(self, quads, Ts):
        
        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_sp_dot = (self.vel_sp - self.prevVel_sp)/Ts
        vel_xy_error = self.vel_sp[0:2] - quads.vel.T[0:2]
        thrust_xy_sp = (np.outer(vel_P_gain[0:2], np.ones(quads.numOfQuads)))*vel_xy_error \
                     - (np.outer(vel_D_gain[0:2], np.ones(quads.numOfQuads)))*quads.vel_dot.T[0:2] \
                     + quads.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2] \
                     + (np.outer(vel_FF_gain[0:2], np.ones(quads.numOfQuads)))*self.vel_sp[0:2] \
                     + (np.outer(vel_FF_dot_gain[0:2], np.ones(quads.numOfQuads)))*vel_sp_dot[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(tiltMax)
        thrust_max_xy = sqrt(quads.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = np.min(np.array([thrust_max_xy, thrust_max_xy_tilt]), axis=0)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        thrust_xy_mag_sq = np.sum(self.thrust_sp[0:2]**2, axis=0)
        exceeds_limit = thrust_xy_mag_sq > thrust_max_xy**2
        if exceeds_limit.any():
            # Only saturate quads that exceed the limit
            mag = norm(self.thrust_sp[0:2], axis=0)
            # Create scale factor: thrust_max_xy / mag for quads that exceed, 1.0 for others
            scale_factor = np.where(exceeds_limit, thrust_max_xy / mag, 1.0)
            self.thrust_sp[0:2] = self.thrust_sp[0:2] * scale_factor[np.newaxis, :]
            pass
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*np.outer(arw_gain, np.ones(quads.numOfQuads))
        self.thr_int[0:2] += (np.outer(vel_I_gain[0:2], np.ones(quads.numOfQuads)))*vel_err_lim*Ts * quads.params["useIntergral"]
        pass
    
    def thrustToAttitude(self, quads, Ts):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thrust_sp)
        if (config.orient == "ENU"):
            body_z = -body_z
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        # y_C shape: (3, numOfQuads) - each column is [-sin(yaw), cos(yaw), 0.0] for each quad
        y_C = np.vstack((-sin(yaw_sp), cos(yaw_sp), np.zeros((1, quads.numOfQuads))))
        
        # Desired body_x axis direction
        # np.cross works on last axis, so we need to transpose for proper cross product
        body_x = np.cross(y_C.T, body_z.T).T
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z.T, body_x.T).T
        body_y = utils.vectNormalize(body_y)

        # Desired rotation matrix and quaternion for each quad
        # Loop over quads since RotToQuat expects a single 3x3 matrix
        self.qd_full = np.zeros((4, quads.numOfQuads))
        for i in range(quads.numOfQuads):
            # Build rotation matrix for quad i: columns are body_x, body_y, body_z
            R_sp = np.column_stack([body_x[:, i], body_y[:, i], body_z[:, i]])
            # Full desired quaternion (full because it considers the desired Yaw angle)
            self.qd_full[:, i] = utils.RotToQuat(R_sp)
        pass
        
    def attitude_control(self, quads, Ts):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = quads.dcm[:,2,:].T  # shape: (3, numOfQuads)
        e_z_d = -utils.vectNormalize(self.thrust_sp)  # shape: (3, numOfQuads)
        if (config.orient == "ENU"):
            e_z_d = -e_z_d

        # Initialize output arrays
        self.qd_red = np.zeros((4, quads.numOfQuads))
        self.qd = np.zeros((4, quads.numOfQuads))
        self.qe = np.zeros((4, quads.numOfQuads))
        self.rate_sp = np.zeros((3, quads.numOfQuads))

        # Process each quad separately since quaternion functions expect single quaternions
        for i in range(quads.numOfQuads):
            # Quaternion error between the 2 vectors for quad i
            e_z_i = e_z[:, i]
            e_z_d_i = e_z_d[:, i]
            dot_product = np.dot(e_z_i, e_z_d_i)
            qe_red = np.zeros(4)
            qe_red[0] = dot_product + sqrt(norm(e_z_i)**2 * norm(e_z_d_i)**2)
            qe_red[1:4] = np.cross(e_z_i, e_z_d_i)
            qe_red = utils.vectNormalize(qe_red)
            
            # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
            self.qd_red[:, i] = utils.quatMultiply(qe_red, quads.quat[i, :])

            # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
            q_mix = utils.quatMultiply(utils.inverse(self.qd_red[:, i]), self.qd_full[:, i])
            q_mix = q_mix*np.sign(q_mix[0])
            q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
            q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
            self.qd[:, i] = utils.quatMultiply(self.qd_red[:, i], np.array([cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w*np.arcsin(q_mix[3]))]))

            # Resulting error quaternion
            self.qe[:, i] = utils.quatMultiply(utils.inverse(quads.quat[i, :]), self.qd[:, i])

            # Create rate setpoint from quaternion error
            self.rate_sp[:, i] = (2.0*np.sign(self.qe[0, i])*self.qe[1:4, i])*att_P_gain
            
            # Limit yawFF for this quad
            yawFF_i = np.clip(self.yawFF[i], -rateMax[2], rateMax[2])

            # Add Yaw rate feed-forward
            dcm_inv = utils.quat2Dcm(utils.inverse(quads.quat[i, :]))
            self.rate_sp[:, i] += dcm_inv[:, 2]*yawFF_i

        # Limit rate setpoint
        # rateMax is (3,), need to reshape to (3, 1) for broadcasting with (3, numOfQuads)
        self.rate_sp = np.clip(self.rate_sp, -rateMax[:, np.newaxis], rateMax[:, np.newaxis])
        pass

    def rate_control(self, quads, Ts):
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - quads.omega.T  # (3, numOfQuads)
        self.rateCtrl = (np.outer(rate_P_gain, np.ones(quads.numOfQuads)))*rate_error - (np.outer(rate_D_gain, np.ones(quads.numOfQuads)))*quads.omega_dot.T     # Be sure it is right sign for the D part
        pass
        

    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain

    