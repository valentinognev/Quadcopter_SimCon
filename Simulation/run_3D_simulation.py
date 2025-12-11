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
from quadFiles.quad import QuadcopterSwarm
from utils.windModel import Wind 
import utils
import config
from load_ulg import load_ulg
from pyulog.core import ULog


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
    numOfQuads = 4
    Ti = 0
    Ts = 0.002
    Tf = 50
    quads = QuadcopterSwarm(Ti, numOfQuads)
    quads.setQuadPos(np.array([0, 0, 0]), 0)
    quads.setQuadPos(np.array([1, 0, 0]), 1)
    quads.setQuadPos(np.array([0, 1, 0]), 2)
    quads.setQuadPos(np.array([0, 0, 1]), 3)
    start_time = time.time()

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
    ctrlType = ControlType.XYZ_POS
    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect[0] = 1        
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3           
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1           
    print("Control type: {}".format(ctrlType))

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    traj = Trajectory(quads, ctrlType, trajSelect)
    ctrl = Control(quads, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States
    # ---------------------------
    sDes = traj.desiredState(0, Ts, quads)        

    # Generate First Commands
    # ---------------------------
    ctrl.controller(traj, quads, Ts)
    
    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quads.state)])
    pos_all        = np.zeros([numTimeStep, len(quads.pos)])
    vel_all        = np.zeros([numTimeStep, len(quads.vel)])
    quat_all       = np.zeros([numTimeStep, len(quads.quat)])
    omega_all      = np.zeros([numTimeStep, len(quads.omega)])
    euler_all      = np.zeros([numTimeStep, len(quads.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, len(quads.wMotor)])
    thr_all        = np.zeros([numTimeStep, len(quads.thr)])
    tor_all        = np.zeros([numTimeStep, len(quads.tor)])

    t_all[0]            = Ti
    s_all[0,:]          = quads.state
    pos_all[0,:]        = quads.pos
    vel_all[0,:]        = quads.vel
    quat_all[0,:]       = quads.quat
    omega_all[0,:]      = quads.omega
    euler_all[0,:]      = quads.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:]     = quads.wMotor
    thr_all[0,:]        = quads.thr
    tor_all[0,:]        = quads.tor

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    while round(t,3) < Tf:
        if t>8:
            pass
        t = quad_sim(t, Ts, quads, ctrl, wind, traj)
        
        # print("{:.3f}".format(t))
        try:
            t_all[i]             = t
            s_all[i,:]           = quads.state
            pos_all[i,:]         = quads.pos
            vel_all[i,:]         = quads.vel
            quat_all[i,:]        = quads.quat
            omega_all[i,:]       = quads.omega
            euler_all[i,:]       = quads.euler
            sDes_traj_all[i,:]   = traj.sDes
            sDes_calc_all[i,:]   = ctrl.sDesCalc
            w_cmd_all[i,:]       = ctrl.w_cmd
            wMotor_all[i,:]      = quads.wMotor
            thr_all[i,:]         = quads.thr
            tor_all[i,:]         = quads.tor
        except:
            break
        i += 1
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # View Results
    # ---------------------------

    # utils.fullprint(sDes_traj_all[:,3:6])
    
    utils.makeFigures(quads.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
           
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quads.params, traj.xyzType, traj.yawType, ifsave)
    # plt.show()
    pass

if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception("{} is not a valid orientation. Verify config.py file.".format(config.orient))