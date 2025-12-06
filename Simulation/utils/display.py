# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import utils

rad2deg = 180.0/pi
deg2rad = pi/180.0
rads2rpm = 60.0/(2.0*pi)
rpm2rads = 2.0*pi/60.0

# Print complete vector or matrices
def fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    print(*args, **kwargs)
    np.set_printoptions(opt)


def makeFigures(params, time, pos_all, vel_all, quat_all, omega_all, euler_all, commands, wMotor_all, thrust, torque, sDes_traj, sDes_calc, ulgData=None):
    x    = pos_all[:,0]
    y    = pos_all[:,1]
    z    = pos_all[:,2]
    q0   = quat_all[:,0]
    q1   = quat_all[:,1]
    q2   = quat_all[:,2]
    q3   = quat_all[:,3]
    xdot = vel_all[:,0]
    ydot = vel_all[:,1]
    zdot = vel_all[:,2]
    p    = omega_all[:,0]*rad2deg
    q    = omega_all[:,1]*rad2deg
    r    = omega_all[:,2]*rad2deg

    wM1  = wMotor_all[:,0]*rads2rpm
    wM2  = wMotor_all[:,1]*rads2rpm
    wM3  = wMotor_all[:,2]*rads2rpm
    wM4  = wMotor_all[:,3]*rads2rpm

    phi   = euler_all[:,0]*rad2deg
    theta = euler_all[:,1]*rad2deg
    psi   = euler_all[:,2]*rad2deg

    x_sp  = sDes_calc[:,0]
    y_sp  = sDes_calc[:,1]
    z_sp  = sDes_calc[:,2]
    Vx_sp = sDes_calc[:,3]
    Vy_sp = sDes_calc[:,4]
    Vz_sp = sDes_calc[:,5]
    x_thr_sp = sDes_calc[:,6]
    y_thr_sp = sDes_calc[:,7]
    z_thr_sp = sDes_calc[:,8]
    q0Des = sDes_calc[:,9]
    q1Des = sDes_calc[:,10]
    q2Des = sDes_calc[:,11]
    q3Des = sDes_calc[:,12]    
    pDes  = sDes_calc[:,13]*rad2deg
    qDes  = sDes_calc[:,14]*rad2deg
    rDes  = sDes_calc[:,15]*rad2deg

    x_tr  = sDes_traj[:,0]
    y_tr  = sDes_traj[:,1]
    z_tr  = sDes_traj[:,2]
    Vx_tr = sDes_traj[:,3]
    Vy_tr = sDes_traj[:,4]
    Vz_tr = sDes_traj[:,5]
    Ax_tr = sDes_traj[:,6]
    Ay_tr = sDes_traj[:,7]
    Az_tr = sDes_traj[:,8]
    yaw_tr = sDes_traj[:,14]*rad2deg
    
    # Extract Euler angle setpoints from trajectory (columns 12:15 are desEul)
    phi_sp_traj = sDes_traj[:,12]*rad2deg
    theta_sp_traj = sDes_traj[:,13]*rad2deg
    psi_sp_traj = sDes_traj[:,14]*rad2deg
    
    # Extract angular rate setpoints from trajectory (columns 15:18 are desPQR)
    p_sp_traj = sDes_traj[:,15]*rad2deg
    q_sp_traj = sDes_traj[:,16]*rad2deg
    r_sp_traj = sDes_traj[:,17]*rad2deg

    uM1 = commands[:,0]*rads2rpm
    uM2 = commands[:,1]*rads2rpm
    uM3 = commands[:,2]*rads2rpm
    uM4 = commands[:,3]*rads2rpm

    x_err = x_sp - x
    y_err = y_sp - y
    z_err = z_sp - z

    psiDes   = np.zeros(q0Des.shape[0])
    thetaDes = np.zeros(q0Des.shape[0])
    phiDes   = np.zeros(q0Des.shape[0])
    for ii in range(q0Des.shape[0]):
        YPR = utils.quatToYPR_ZYX(sDes_calc[ii,9:13])
        psiDes[ii]   = YPR[0]*rad2deg
        thetaDes[ii] = YPR[1]*rad2deg
        phiDes[ii]   = YPR[2]*rad2deg
    
    plt.show()

    plt.figure()
    plt.suptitle('Position: Simulated vs Setpoint')
    
    # X position
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, x, label='x (simulated)', linewidth=2)
    plt.plot(time, x_sp, '--', label='x (setpoint)', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    
    # Y position
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, y, label='y (simulated)', linewidth=2)
    plt.plot(time, y_sp, '--', label='y (setpoint)', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    
    # Z position
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, z, label='z (simulated)', linewidth=2)
    plt.plot(time, z_sp, '--', label='z (setpoint)', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.draw()



    plt.figure()
    plt.suptitle('Velocity: Simulated vs Setpoint')
    
    # Vx velocity
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, xdot, label='Vx (simulated)', linewidth=2)
    plt.plot(time, Vx_sp, '--', label='Vx (setpoint)', linewidth=2)
    # Add ulg data if available
    if ulgData is not None:
        if 'vehicle_local_position_vx' in ulgData:
            vx_data = ulgData['vehicle_local_position_vx']
            plt.plot(vx_data['timestamp'], vx_data['data'], ':', label='Vx (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Vx (m/s)')
    
    # Vy velocity
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, ydot, label='Vy (simulated)', linewidth=2)
    plt.plot(time, Vy_sp, '--', label='Vy (setpoint)', linewidth=2)
    # Add ulg data if available
    if ulgData is not None:
        if 'vehicle_local_position_vy' in ulgData:
            vy_data = ulgData['vehicle_local_position_vy']
            plt.plot(vy_data['timestamp'], vy_data['data'], ':', label='Vy (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Vy (m/s)')
    
    # Vz velocity
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, zdot, label='Vz (simulated)', linewidth=2)
    plt.plot(time, Vz_sp, '--', label='Vz (setpoint)', linewidth=2)
    # Add ulg data if available
    if ulgData is not None:
        if 'vehicle_local_position_vz' in ulgData:
            vz_data = ulgData['vehicle_local_position_vz']
            plt.plot(vz_data['timestamp'], vz_data['data'], ':', label='Vz (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Vz (m/s)')
    plt.draw()

    plt.figure()
    plt.plot(time, x_thr_sp, time, y_thr_sp, time, z_thr_sp)
    plt.grid(True)
    plt.legend(['x_thr_sp','y_thr_sp','z_thr_sp'])
    plt.xlabel('Time (s)')
    plt.ylabel('Desired Thrust (N)')
    plt.draw()


    # attitude plot
    plt.figure()
    plt.suptitle('Euler Angles: Simulated vs Setpoint')
    
    # Roll
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, phi, label='Roll (simulated)', linewidth=2)
    plt.plot(time, phiDes, '--', label='Roll (setpoint calc)', linewidth=2)
    plt.plot(time, phi_sp_traj, ':', label='Roll (setpoint traj)', linewidth=2)
    if ulgData is not None and 'vehicle_attitude_roll' in ulgData:
        roll_ulg = ulgData['vehicle_attitude_roll']
        plt.plot(roll_ulg['timestamp'], roll_ulg['data'] * rad2deg, '-.', label='Roll (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (°)')
    
    # Pitch
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, theta, label='Pitch (simulated)', linewidth=2)
    plt.plot(time, thetaDes, '--', label='Pitch (setpoint calc)', linewidth=2)
    plt.plot(time, theta_sp_traj, ':', label='Pitch (setpoint traj)', linewidth=2)
    if ulgData is not None and 'vehicle_attitude_pitch' in ulgData:
        pitch_ulg = ulgData['vehicle_attitude_pitch']
        plt.plot(pitch_ulg['timestamp'], pitch_ulg['data'] * rad2deg, '-.', label='Pitch (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (°)')
    
    # Yaw
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, psi, label='Yaw (simulated)', linewidth=2)
    plt.plot(time, psiDes, '--', label='Yaw (setpoint calc)', linewidth=2)
    plt.plot(time, psi_sp_traj, ':', label='Yaw (setpoint traj)', linewidth=2)
    if ulgData is not None and 'vehicle_attitude_yaw' in ulgData:
        yaw_ulg = ulgData['vehicle_attitude_yaw']
        plt.plot(yaw_ulg['timestamp'], yaw_ulg['data'] * rad2deg, '-.', label='Yaw (ulg)', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (°)')
    plt.draw()
    
    # angular rates plot
    plt.figure()
    plt.suptitle('Angular Rates: Simulated vs Setpoint')
    
    # Roll rate (p)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, p, label='Roll Rate (p) - simulated', linewidth=2)
    plt.plot(time, pDes, '--', label='Roll Rate (p) - setpoint calc', linewidth=2)
    plt.plot(time, p_sp_traj, ':', label='Roll Rate (p) - setpoint traj', linewidth=2)
    if ulgData is not None and 'vehicle_angular_velocity_xyz[0]' in ulgData:
        p_ulg = ulgData['vehicle_angular_velocity_xyz[0]']
        plt.plot(p_ulg['timestamp'], p_ulg['data'] * rad2deg, '-.', label='Roll Rate (p) - ulg', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Roll Rate (°/s)')
    
    # Pitch rate (q)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, q, label='Pitch Rate (q) - simulated', linewidth=2)
    plt.plot(time, qDes, '--', label='Pitch Rate (q) - setpoint calc', linewidth=2)
    plt.plot(time, q_sp_traj, ':', label='Pitch Rate (q) - setpoint traj', linewidth=2)
    if ulgData is not None and 'vehicle_angular_velocity_xyz[1]' in ulgData:
        q_ulg = ulgData['vehicle_angular_velocity_xyz[1]']
        plt.plot(q_ulg['timestamp'], q_ulg['data'] * rad2deg, '-.', label='Pitch Rate (q) - ulg', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Rate (°/s)')
    
    # Yaw rate (r)
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, r, label='Yaw Rate (r) - simulated', linewidth=2)
    plt.plot(time, rDes, '--', label='Yaw Rate (r) - setpoint calc', linewidth=2)
    plt.plot(time, r_sp_traj, ':', label='Yaw Rate (r) - setpoint traj', linewidth=2)
    if ulgData is not None and 'vehicle_angular_velocity_xyz[2]' in ulgData:
        r_ulg = ulgData['vehicle_angular_velocity_xyz[2]']
        plt.plot(r_ulg['timestamp'], r_ulg['data'] * rad2deg, '-.', label='Yaw Rate (r) - ulg', linewidth=2, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Rate (°/s)')
    plt.draw()

    plt.figure()
    plt.plot(time, wM1, time, wM2, time, wM3, time, wM4)
    plt.plot(time, uM1, '--', time, uM2, '--', time, uM3, '--', time, uM4, '--')
    plt.grid(True)
    plt.legend(['w1','w2','w3','w4'])
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Angular Velocity (RPM)')
    plt.draw()

    plt.figure()
    ax1 = plt.subplot(2,1,1)
    plt.plot(time, thrust[:,0], time, thrust[:,1], time, thrust[:,2], time, thrust[:,3])
    plt.grid(True)
    plt.legend(['thr1','thr2','thr3','thr4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Thrust (N)')
    plt.draw()

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.plot(time, torque[:,0], time, torque[:,1], time, torque[:,2], time, torque[:,3])
    plt.grid(True)
    plt.legend(['tor1','tor2','tor3','tor4'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor Torque (N*m)')
    plt.draw()

    plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.title('Trajectory Setpoints')
    plt.plot(time, x_tr, time, y_tr, time, z_tr)
    plt.grid(True)
    plt.legend(['x','y','z'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    plt.plot(time, Vx_tr, time, Vy_tr, time, Vz_tr)
    plt.grid(True)
    plt.legend(['Vx','Vy','Vz'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
   
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(time, Ax_tr, time, Ay_tr, time, Az_tr)
    plt.grid(True)
    plt.legend(['Ax','Ay','Az'], loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.draw()

    plt.figure()
    plt.plot(time, x_err, time, y_err, time, z_err)
    plt.grid(True)
    plt.legend(['Pos x error','Pos y error','Pos z error'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.draw()


def plotComparisonWithUlg(time, euler_all, omega_all, ulgData=None):
    """
    Plot comparison between simulated angles/rates and recorded values from ulg file.
    
    Args:
        time: Simulation time array
        euler_all: Simulated Euler angles [N, 3] (roll, pitch, yaw) in radians
        omega_all: Simulated angular velocities [N, 3] (p, q, r) in rad/s
        ulgData: Dictionary with ulg data containing recorded angles and rates
    """
    if ulgData is None:
        return
    
    phi = euler_all[:, 0] * rad2deg
    theta = euler_all[:, 1] * rad2deg
    psi = euler_all[:, 2] * rad2deg
    p = omega_all[:, 0] * rad2deg
    q = omega_all[:, 1] * rad2deg
    r = omega_all[:, 2] * rad2deg
    
    # Plot angle comparison
    plt.figure()
    plt.suptitle('Angle Comparison: Simulated vs Recorded (ULG)')
    
    # Roll
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, phi, label='Simulated Roll', linewidth=2)
    if 'vehicle_attitude_roll' in ulgData:
        roll_data = ulgData['vehicle_attitude_roll']
        plt.plot(roll_data['timestamp'], roll_data['data'] * rad2deg, '--', 
                label='Recorded Roll (ULG)', linewidth=2)
    if 'vehicle_attitude_setpoint_roll_body' in ulgData:
        roll_sp_data = ulgData['vehicle_attitude_setpoint_roll_body']
        plt.plot(roll_sp_data['timestamp'], roll_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Roll (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (°)')
    
    # Pitch
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, theta, label='Simulated Pitch', linewidth=2)
    if 'vehicle_attitude_pitch' in ulgData:
        pitch_data = ulgData['vehicle_attitude_pitch']
        plt.plot(pitch_data['timestamp'], pitch_data['data'] * rad2deg, '--', 
                label='Recorded Pitch (ULG)', linewidth=2)
    if 'vehicle_attitude_setpoint_pitch_body' in ulgData:
        pitch_sp_data = ulgData['vehicle_attitude_setpoint_pitch_body']
        plt.plot(pitch_sp_data['timestamp'], pitch_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Pitch (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (°)')
    
    # Yaw
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, psi, label='Simulated Yaw', linewidth=2)
    if 'vehicle_attitude_yaw' in ulgData:
        yaw_data = ulgData['vehicle_attitude_yaw']
        plt.plot(yaw_data['timestamp'], yaw_data['data'] * rad2deg, '--', 
                label='Recorded Yaw (ULG)', linewidth=2)
    if 'vehicle_attitude_setpoint_yaw_body' in ulgData:
        yaw_sp_data = ulgData['vehicle_attitude_setpoint_yaw_body']
        plt.plot(yaw_sp_data['timestamp'], yaw_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Yaw (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (°)')
    plt.draw()
    
    # Plot rate comparison
    plt.figure()
    plt.suptitle('Angular Rate Comparison: Simulated vs Recorded (ULG)')
    
    # Roll rate (p)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, p, label='Simulated Roll Rate (p)', linewidth=2)
    if 'vehicle_angular_velocity_xyz[0]' in ulgData:
        p_data = ulgData['vehicle_angular_velocity_xyz[0]']
        plt.plot(p_data['timestamp'], p_data['data'] * rad2deg, '--', 
                label='Recorded Roll Rate (ULG)', linewidth=2)
    if 'vehicle_rates_setpoint_roll' in ulgData:
        p_sp_data = ulgData['vehicle_rates_setpoint_roll']
        plt.plot(p_sp_data['timestamp'], p_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Roll Rate (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Roll Rate (°/s)')
    
    # Pitch rate (q)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time, q, label='Simulated Pitch Rate (q)', linewidth=2)
    if 'vehicle_angular_velocity_xyz[1]' in ulgData:
        q_data = ulgData['vehicle_angular_velocity_xyz[1]']
        plt.plot(q_data['timestamp'], q_data['data'] * rad2deg, '--', 
                label='Recorded Pitch Rate (ULG)', linewidth=2)
    if 'vehicle_rates_setpoint_pitch' in ulgData:
        q_sp_data = ulgData['vehicle_rates_setpoint_pitch']
        plt.plot(q_sp_data['timestamp'], q_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Pitch Rate (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Rate (°/s)')
    
    # Yaw rate (r)
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time, r, label='Simulated Yaw Rate (r)', linewidth=2)
    if 'vehicle_angular_velocity_xyz[2]' in ulgData:
        r_data = ulgData['vehicle_angular_velocity_xyz[2]']
        plt.plot(r_data['timestamp'], r_data['data'] * rad2deg, '--', 
                label='Recorded Yaw Rate (ULG)', linewidth=2)
    if 'vehicle_rates_setpoint_yaw' in ulgData:
        r_sp_data = ulgData['vehicle_rates_setpoint_yaw']
        plt.plot(r_sp_data['timestamp'], r_sp_data['data'] * rad2deg, ':', 
                label='Setpoint Yaw Rate (ULG)', linewidth=1.5, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Rate (°/s)')
    plt.draw()