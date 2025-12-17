# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile

from trajectory import Trajectory
from ctrl import Control, ControlType
from quadFiles.quad import Quadcopter
from utils.windModel import Wind 
import utils
import config
from load_ulg import load_ulg
from pyulog.core import ULog


LOAD_ULG_DATA = True

# # Load rotor parameters from ULog file
# def load_rotor_params_from_ulog(ulog_file):
#     """
#     Extract rotor parameters from ULog file.
#     Returns rotor_positions, rotor_thrust_directions, rotor_torque_directions
#     """
#     ulog = ULog(ulog_file)
#     params = ulog.initial_parameters
    
#     'vehicle_local_position'
#     # Get number of rotors
#     num_rotors = int(params.get('CA_ROTOR_COUNT', 4))
    
#     rotor_positions = []
#     rotor_thrust_directions = []
#     rotor_torque_directions = []

#     # rotMat_FLU_FRD=np.array([
#     #     [1, 0, 0],
#     #     [0, -1, 0],
#     #     [0, 0, -1],
#     # ])
#     rotMat_FLU_FRD = np.eye(3)
    
#     for i in range(num_rotors):
#         # Position (PX, PY, PZ)
#         px = params.get(f'CA_ROTOR{i}_PX', 0.0)
#         py = params.get(f'CA_ROTOR{i}_PY', 0.0)
#         pz = params.get(f'CA_ROTOR{i}_PZ', 0.0)
#         rotor_positions.append(rotMat_FLU_FRD @ np.array([px, py, pz]))
        
#         # Thrust direction (-AX, AY, -AZ) - note the sign changes for FLU frame
#         ax = params.get(f'CA_ROTOR{i}_AX', 0.0)
#         ay = params.get(f'CA_ROTOR{i}_AY', 0.0)
#         az = params.get(f'CA_ROTOR{i}_AZ', 1.0)
#         rotor_thrust_directions.append(rotMat_FLU_FRD @ np.array([ax, ay, az]))
        
#         # Torque direction based on sign of KM
#         # km = params.get(f'CA_ROTOR{i}_KM', 0.05)
#         # torque_sign = np.sign(km)
#         # Logic based on position:
#         # +x, +y and -x, -y -> positive torque (sign(px) == sign(py))
#         # -x, +y and +x, -y -> negative torque (sign(px) != sign(py))
#         # Note: px and py here are in the original frame before rotation to FRD
#         # However, the user request says "+x,+y and -x,-y direction is positive", referring to the rotor positions.
#         # Let's use the signs of px and py directly.
        
#         # Handle cases where px or py might be 0 (e.g. X or + configuration aligned with axes)
#         # Assuming standard quadrotor configuration where rotors are in quadrants.
#         # If px or py is 0, this logic might need refinement, but based on "CA_ROTORi_PX/PY" description it seems standard.
#         # Let's assume standard quad X or + where they are not 0, or if 0, we might need to look at the other coordinate.
#         # Given the user prompt: "+x,+y and -x,-y direction is positive", implies quadrants 1 and 3 are positive.
        
#         if np.sign(px) == np.sign(py):
#              torque_sign = 1.0
#         else:
#              torque_sign = -1.0
             
#         rotor_torque_directions.append(rotMat_FLU_FRD @ np.array([0, 0, torque_sign]))
    
#     return (np.array(rotor_positions), 
#             np.array(rotor_thrust_directions), 
#             np.array(rotor_torque_directions))


def quad_sim(t, Ts, quad, ctrl, wind, traj):
    
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Trajectory for Desired States 
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quad)        

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj=traj, quad=quad, Ts=Ts)

    return t
    
def getStartOffboardInds(timestamp, data):
    offboard_inds = np.where(data == 1)[0]
    return offboard_inds[0]

def main():
    start_time = time.time()
    if LOAD_ULG_DATA:
        ulgData = load_ulg("/home/valentin/RL/TESTFLIGHTS/RLCat2_3blades/RLFlights/yawOn/FastYawRate.ulg", 
                        fields_to_extract=[['vehicle_local_position','vx'],['vehicle_local_position','vy'],['vehicle_local_position','vz'],
                                        ['vehicle_local_position_setpoint','vx'],['vehicle_local_position_setpoint','vy'],['vehicle_local_position_setpoint','vz'],
                                        ['vehicle_attitude','roll'],['vehicle_attitude','pitch'],['vehicle_attitude','yaw'],
                                        ['vehicle_attitude_setpoint','roll_body'],['vehicle_attitude_setpoint','pitch_body'],['vehicle_attitude_setpoint','yaw_body'],
                                        ['vehicle_rates_setpoint','pitch'],['vehicle_rates_setpoint','roll'],['vehicle_rates_setpoint','yaw'],
                                        ['vehicle_angular_velocity','xyz[0]'],['vehicle_angular_velocity','xyz[1]'],['vehicle_angular_velocity','xyz[2]'],
                                        ['vehicle_thrust_setpoint','xyz[0]'],['vehicle_thrust_setpoint','xyz[1]'],['vehicle_thrust_setpoint','xyz[2]'],
                                        ['vehicle_control_mode','flag_control_offboard_enabled']],
                        startTime=396, verbose=True)
        # ulgData = load_ulg("/home/valentin/RL/TESTFLIGHTS/RLCat2_3blades/PIDcontrolCircle/severalCircles_light.ulg", 
        #                 fields_to_extract=[['vehicle_local_position','vx'],['vehicle_local_position','vy'],['vehicle_local_position','vz'],
        #                                 ['vehicle_local_position_setpoint','vx'],['vehicle_local_position_setpoint','vy'],['vehicle_local_position_setpoint','vz'],
        #                                 ['vehicle_attitude','roll'],['vehicle_attitude','pitch'],['vehicle_attitude','yaw'],
        #                                 ['vehicle_attitude_setpoint','roll_body'],['vehicle_attitude_setpoint','pitch_body'],['vehicle_attitude_setpoint','yaw_body'],
        #                                 ['vehicle_rates_setpoint','pitch'],['vehicle_rates_setpoint','roll'],['vehicle_rates_setpoint','yaw'],
        #                                 ['vehicle_angular_velocity','xyz[0]'],['vehicle_angular_velocity','xyz[1]'],['vehicle_angular_velocity','xyz[2]'],
        #                                 ['vehicle_thrust_setpoint','xyz[0]'],['vehicle_thrust_setpoint','xyz[1]'],['vehicle_thrust_setpoint','xyz[2]'],
        #                                 ['vehicle_control_mode','flag_control_offboard_enabled']],
        #                 startTime=336, verbose=True)
        # ulgData = load_ulg("/home/valentin/RL/TESTFLIGHTS/RLCat2_3blades/PIDcontrolCircle/severalCircles_heavy.ulg", 
        #                 fields_to_extract=[['vehicle_local_position','vx'],['vehicle_local_position','vy'],['vehicle_local_position','vz'],
        #                                 ['vehicle_local_position_setpoint','vx'],['vehicle_local_position_setpoint','vy'],['vehicle_local_position_setpoint','vz'],
        #                                 ['vehicle_attitude','roll'],['vehicle_attitude','pitch'],['vehicle_attitude','yaw'],
        #                                 ['vehicle_attitude_setpoint','roll_body'],['vehicle_attitude_setpoint','pitch_body'],['vehicle_attitude_setpoint','yaw_body'],
        #                                 ['vehicle_rates_setpoint','pitch'],['vehicle_rates_setpoint','roll'],['vehicle_rates_setpoint','yaw'],
        #                                 ['vehicle_angular_velocity','xyz[0]'],['vehicle_angular_velocity','xyz[1]'],['vehicle_angular_velocity','xyz[2]'],
        #                                 ['vehicle_thrust_setpoint','xyz[0]'],['vehicle_thrust_setpoint','xyz[1]'],['vehicle_thrust_setpoint','xyz[2]'],
        #                                 ['vehicle_control_mode','flag_control_offboard_enabled']],
        #                 startTime=590, verbose=True)
        offinds = getStartOffboardInds(ulgData['vehicle_control_mode_flag_control_offboard_enabled']['timestamp'], ulgData['vehicle_control_mode_flag_control_offboard_enabled']['data'])
        offstartTime=ulgData['vehicle_control_mode_flag_control_offboard_enabled']['timestamp'][offinds]
        Tf = 50 #ulgData['vehicle_control_mode_flag_control_offboard_enabled']['timestamp'][-10]
        # plt.plot(ulgData['vehicle_rates_setpoint_yaw']['timestamp'], ulgData['vehicle_rates_setpoint_yaw']['data'])
        # plt.plot(ulgData['vehicle_angular_velocity_xyz[2]']['timestamp'], ulgData['vehicle_angular_velocity_xyz[2]']['data'])
        # plt.show()
        pass
    else:
        ulgData = None
        Tf=None

    # plt.figure(1)
    # plt.subplot(3, 1, 1)
    # plt.plot(ulgData['vehicle_attitude_setpoint_roll_body']['timestamp'], ulgData['vehicle_attitude_setpoint_roll_body']['data'])
    # plt.plot(ulgData['vehicle_attitude_roll']['timestamp'], ulgData['vehicle_attitude_roll']['data'])
    # plt.title('Roll');  plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)
    # plt.subplot(3, 1, 2)
    # plt.plot(ulgData['vehicle_attitude_setpoint_pitch_body']['timestamp'], ulgData['vehicle_attitude_setpoint_pitch_body']['data'])
    # plt.plot(ulgData['vehicle_attitude_pitch']['timestamp'], ulgData['vehicle_attitude_pitch']['data'])
    # plt.title('Pitch'); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)
    # plt.subplot(3, 1, 3)
    # plt.plot(ulgData['vehicle_attitude_setpoint_yaw_body']['timestamp'], ulgData['vehicle_attitude_setpoint_yaw_body']['data'])
    # plt.plot(ulgData['vehicle_attitude_yaw']['timestamp'], ulgData['vehicle_attitude_yaw']['data'])
    # plt.title('Yaw'); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)


    # plt.figure(2)
    # plt.subplot(3, 1, 1)
    # plt.plot(ulgData['vehicle_rates_setpoint_roll']['timestamp'], ulgData['vehicle_rates_setpoint_roll']['data'])
    # plt.plot(ulgData['vehicle_angular_velocity_xyz[0]']['timestamp'], ulgData['vehicle_angular_velocity_xyz[0]']['data'])
    # plt.title('Roll'); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)
    # plt.subplot(3, 1, 2)
    # plt.plot(ulgData['vehicle_rates_setpoint_pitch']['timestamp'], ulgData['vehicle_rates_setpoint_pitch']['data'])
    # plt.plot(ulgData['vehicle_angular_velocity_xyz[1]']['timestamp'], ulgData['vehicle_angular_velocity_xyz[1]']['data'])
    # plt.title('Pitch'); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)
    # plt.subplot(3, 1, 3)
    # plt.plot(ulgData['vehicle_rates_setpoint_yaw']['timestamp'], ulgData['vehicle_rates_setpoint_yaw']['data'])
    # plt.plot(ulgData['vehicle_angular_velocity_xyz[2]']['timestamp'], ulgData['vehicle_angular_velocity_xyz[2]']['data'])
    # plt.title('Yaw'); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)'); plt.grid(True)
    # plt.show()

    # Simulation Setup
    # --------------------------- 
    Ti = 0
    Ts = 0.002
    Tf = Tf if Tf is not None else 50
    ifsave = 0
 
    # Choose trajectory settings
    # --------------------------- 
    trajSelect = np.zeros(3)

    # Select Control Type
    # For attitude target (angles + thrust, rates calculated): use ControlType.ATT
    # For attitude rate target (rates + thrust, bypasses attitude_control): use ControlType.ATT_RATE
    # ControlType: XYZ_POS, XY_VEL_Z_POS, XYZ_VEL, ATT, ATT_RATE
    ctrlType = ControlType.XY_VEL_Z_POS if (LOAD_ULG_DATA and ulgData is not None) else ControlType.XYZ_POS
    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect[0] = 1 if LOAD_ULG_DATA else 99        
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3           
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1           
    print("Control type: {}".format(ctrlType))

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter(Ti)
    traj = Trajectory(quad, ctrlType, trajSelect, ulgData=None if ulgData is None else ulgData)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States
    # ---------------------------
    sDes = traj.desiredState(0, Ts, quad)        

    # Generate First Commands
    # ---------------------------
    ctrl.controller(traj, quad, Ts)
    
    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all        = np.zeros([numTimeStep, len(quad.thr)])
    tor_all        = np.zeros([numTimeStep, len(quad.tor)])

    t_all[0]            = Ti
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:]     = quad.wMotor
    thr_all[0,:]        = quad.thr
    tor_all[0,:]        = quad.tor

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    while round(t,3) < Tf:
        if t>3.2:
            pass
        t = quad_sim(t, Ts, quad, ctrl, wind, traj)
        
        # print("{:.3f}".format(t))
        try:
            t_all[i]             = t
            s_all[i,:]           = quad.state
            pos_all[i,:]         = quad.pos
            vel_all[i,:]         = quad.vel
            quat_all[i,:]        = quad.quat
            omega_all[i,:]       = quad.omega
            euler_all[i,:]       = quad.euler
            sDes_traj_all[i,:]   = traj.sDes
            sDes_calc_all[i,:]   = ctrl.sDesCalc
            w_cmd_all[i,:]       = ctrl.w_cmd
            wMotor_all[i,:]      = quad.wMotor
            thr_all[i,:]         = quad.thr
            tor_all[i,:]         = quad.tor
        except:
            break
        i += 1
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # View Results
    # ---------------------------

    # utils.fullprint(sDes_traj_all[:,3:6])
    ulgData_for_plots = ulgData if (LOAD_ULG_DATA and ulgData is not None) else None
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all, ulgData_for_plots)
           
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    # plt.show()
    pass

if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception("{} is not a valid orientation. Verify config.py file.".format(config.orient))