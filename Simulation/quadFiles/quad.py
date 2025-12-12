# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode
import sys
import os
simulation_path = os.path.join(os.path.dirname(__file__), 'Quadcopter_SimCon', 'Simulation')
sys.path.append(simulation_path)

# Support both relative and absolute imports
try:
    from .initQuad import sys_params, init_cmd, init_state
    from ..utils.windModel import Wind
    from .. import utils
    from .. import config
except ImportError:
    from quadFiles.initQuad import sys_params, init_cmd, init_state
    from utils.windModel import Wind
    import utils
    import config

deg2rad = pi/180.0

class QuadcopterSwarm:

    def __init__(self, Ti, numOfQuads):
        
        self.numOfQuads = numOfQuads
        # Quad Params
        # ---------------------------
        self.params = sys_params()
        
        # Command for initial stable hover
        # ---------------------------
        ini_hover = init_cmd(self.params)
        self.params["FF"] = ini_hover[0]         # Feed-Forward Command for Hover
        self.params["w_hover"] = ini_hover[1]    # Motor Speed for Hover
        self.params["thr_hover"] = ini_hover[2]  # Motor Thrust for Hover  
        self.thr = np.ones(4)*ini_hover[2]
        self.tor = np.ones(4)*ini_hover[3]
        
        self.psi = np.zeros(numOfQuads)
        self.theta = np.zeros(numOfQuads)
        self.phi = np.zeros(numOfQuads)
        self.euler = np.zeros((numOfQuads, 3))
        self.dcm = np.zeros((numOfQuads, 3, 3))

        # Initial State
        # ---------------------------
        single_state = init_state(self.params)
        self.state = np.tile(single_state, (numOfQuads, 1))

        self.pos   = self.state[:, 0:3]
        self.quat  = self.state[:, 3:7]
        self.vel   = self.state[:, 7:10]
        self.omega = self.state[:, 10:13]
        self.wMotor = np.array([self.state[:, 13], self.state[:, 14], self.state[:, 15], self.state[:, 16]])
        self.vel_dot = np.zeros((numOfQuads, 3))
        self.omega_dot = np.zeros((numOfQuads, 3))
        self.acc = np.zeros((numOfQuads, 3))

        self.extended_state()
        self.forces()

        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot_wrapper).set_integrator('dopri5', first_step='0.00005', atol='1e-6', rtol='1e-6')
        self.integratorState = self.state.flatten()
        self.integrator.set_initial_value(self.integratorState, Ti)

    def setQuadPos(self, pos, quadIndex):
        self.pos[quadIndex,:] = pos
        
    def setQuadQuat(self, quat, quadIndex):
        self.quat[quadIndex,:] = quat
        
    def setQuadEuler(self, RPY, quadIndex):
        self.euler[quadIndex,:] = RPY[::-1] # flip RPY to YPR so that euler state = phi, theta, psi
        self.quat = utils.YPRToQuat(self.euler[quadIndex,0], self.euler[quadIndex,1], self.euler[quadIndex,2])
        
    def setQuadVel(self, vel, quadIndex):
        self.vel[quadIndex,:] = vel
    # def setQuadOmega(self, omega, quadIndex):
    #     self.omega[quadIndex,:] = omega
    # def setQuadWMotor(self, wMotor, quadIndex):
    #     self.wMotor[quadIndex,:] = wMotor

    def extended_state(self):

        # Rotation Matrix of current state (Direct Cosine Matrix)
        for i in range(self.numOfQuads):
            self.dcm[i,:,:] = utils.quat2Dcm(self.quat[i,:])

        # Euler angles of current state
        YPR = np.zeros((self.numOfQuads, 3))
        for i in range(self.numOfQuads):
            YPR[i,:] = utils.quatToYPR_ZYX(self.quat[i,:])
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi[:] = YPR[:,0]
        self.theta[:] = YPR[:,1]
        self.phi[:] = YPR[:,2]

    
    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def state_dot_wrapper(self, t, state_flat, cmd, wind):
        """
        Wrapper function for the ODE integrator.
        Converts flattened state to 2D, calls state_dot for each quad, and flattens the result.
        
        Args:
            t: Time
            state_flat: Flattened state vector (numOfQuads * 17 elements)
            cmd: Motor commands, shape (4, numOfQuads) or (4,) for single quad
            wind: Wind object
            
        Returns:
            Flattened state derivative vector (numOfQuads * 17 elements)
        """
        # Reshape flattened state to (numOfQuads, 17)
        state_2d = state_flat.reshape(self.numOfQuads, 17)
        
        # Initialize output array
        sdot_all = np.zeros((self.numOfQuads, 17))
        
        # Process each quad separately
        sdot_all = self.state_dot(t, state_2d, cmd, wind)
        
        # Flatten the result for the integrator
        return sdot_all.T.flatten()

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        # kp   = self.params["kp"]
        # damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[:,0]
        y      = state[:,1]
        z      = state[:,2]
        q0     = state[:,3]
        q1     = state[:,4]
        q2     = state[:,5]
        q3     = state[:,6]
        xdot   = state[:,7]
        ydot   = state[:,8]
        zdot   = state[:,9]
        p      = state[:,10]
        q      = state[:,11]
        r      = state[:,12]
        wM1    = state[:,13]
        wM2    = state[:,14]
        wM3    = state[:,15]
        wM4    = state[:,16]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        
        uMotor = cmd
        wdotM1 = (- wM1 + uMotor[0])/(tau)
        wdotM2 = (- wM2 + uMotor[1])/(tau)
        wdotM3 = (- wM3 + uMotor[2])/(tau)
        wdotM4 = (- wM4 + uMotor[3])/(tau)
    
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (config.orient == "NED"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (config.orient == "ENU"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([17,self.numOfQuads])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wdotM2
        sdot[15] = wdotM3
        sdot[16] = wdotM4

        # Note: self.acc is updated per-quad in the update() method
        # Don't update it here since state_dot handles a single quad

        return sdot

    def update(self, t, Ts, cmd, wind):

        if wind is None:
            wind = Wind()

        prev_vel   = self.vel
        prev_omega = self.omega

        self.integrator.set_f_params(cmd, wind)
        self.integratorState = self.integrator.integrate(t, t+Ts)
        # Reshape flattened state back to (numOfQuads, 17)
        self.state = self.integratorState.reshape(self.numOfQuads, 17)

        self.pos   = self.state[:,0:3]
        self.quat  = self.state[:,3:7]
        self.vel   = self.state[:,7:10]
        self.omega = self.state[:,10:13]
        self.wMotor = self.state[:,13:17]

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts
        self.acc = self.vel_dot  # Acceleration is the derivative of velocity

        self.extended_state()
        self.forces()
