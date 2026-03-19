# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import logging
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from trajectory import Trajectory, PositionTrajectoryType, YawTrajectoryType, WaypointTimeMode
from ctrl import Control, ControlType
from quadFiles.quad import QuadcopterSwarm
from utils.windModel import Wind
import utils
import config
from load_ulg import load_ulg
from pyulog.core import ULog


USE_CAT_DRONE_CONFIG = True

# Optional: path to drone + control JSON config. If None or file missing, use defaults from initQuad/ctrl.
if USE_CAT_DRONE_CONFIG:
    DRONE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "cat_drone_config.json")
    DEFAULT_REAL_FLIGHT_ULG = "/home/valentin/RL/TESTFLIGHTS/log_3_2025-12-7-16-23-46.ulg"
    # ULG reproduction: match reference project (RL/t) for reproducing flight logs
    ULG_REPRODUCTION_START_TIME = 200.0  # Crop log from this time (s); timestamps shifted to 0
    ULG_REPRODUCTION_TF = 50.0  # Fixed simulation end time (s)
else:
    DRONE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "iris_drone_config.json")
    DEFAULT_REAL_FLIGHT_ULG = "/home/valentin/RL/TESTFLIGHTS/gazebo/log_0_2025-12-22-21-27-14.ulg"
    # ULG reproduction: match reference project (RL/t) for reproducing flight logs
    ULG_REPRODUCTION_START_TIME = 50.0  # Crop log from this time (s); timestamps shifted to 0
    ULG_REPRODUCTION_TF = 50.0  # Fixed simulation end time (s)

# Optional: path to system_manager mission/config JSON (e.g. missionType, waypointList, controller params).
# If set and file exists, mission is loaded from this file; otherwise minimal defaults are used.
# Example: system_manager/Input/velPIDcontrol.json
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_MANAGER_MISSION_CONFIG = os.path.join(_SIM_DIR, "..", "..", "system_manager", "Input", "velPIDcontrol.json")

# Set True to use system_manager as high-level controller (velocity + yaw_rate from sys_manager_step).
# Requires system_manager at CatSwarm/system_manager/system_managerPY. Only first quad (index 0) is controlled.
USE_SYSTEM_MANAGER = False  # Set True to use system_manager as high-level controller (1 quad, in-process).

REAL_FLIGHT_FIELDS = [
    ["vehicle_local_position", "vx"],
    ["vehicle_local_position", "vy"],
    ["vehicle_local_position", "vz"],
    ["vehicle_local_position_setpoint", "vx"],
    ["vehicle_local_position_setpoint", "vy"],
    ["vehicle_local_position_setpoint", "vz"],
    ["vehicle_attitude", "roll"],
    ["vehicle_attitude", "pitch"],
    ["vehicle_attitude", "yaw"],
    ["vehicle_attitude_setpoint", "roll_body"],
    ["vehicle_attitude_setpoint", "pitch_body"],
    ["vehicle_attitude_setpoint", "yaw_body"],
    ["vehicle_rates_setpoint", "pitch"],
    ["vehicle_rates_setpoint", "roll"],
    ["vehicle_rates_setpoint", "yaw"],
    ["vehicle_angular_velocity", "xyz[0]"],
    ["vehicle_angular_velocity", "xyz[1]"],
    ["vehicle_angular_velocity", "xyz[2]"],
    ["vehicle_thrust_setpoint", "xyz[0]"],
    ["vehicle_thrust_setpoint", "xyz[1]"],
    ["vehicle_thrust_setpoint", "xyz[2]"],
    ["vehicle_control_mode", "flag_control_offboard_enabled"],
]


def quad_sim(t, Ts, quads, ctrl, wind, traj):
    
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quads.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Trajectory for Desired States 
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quads)        

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quads, Ts)

    return t


def quad_sim_system_manager(t, Ts, quads, ctrl, wind, traj, sys_manager, flight_data_from_swarm):
    """One simulation step with system_manager as high-level controller (desired vel + yaw_rate)."""
    # Build flight data from current quad state (quad index 0)
    flight_Data = flight_data_from_swarm(quads, 0, t)
    msg = sys_manager.sys_manager_step(flight_Data=flight_Data, curTime=t, log_data=True)

    # Fallback if msg empty or missing keys
    if not msg or "velCmd" not in msg:
        vel_cmd = [0.0, 0.0, 0.0]
        yaw_rate_cmd = 0.0
    else:
        vel_cmd = list(msg["velCmd"]) if hasattr(msg["velCmd"], "__iter__") else [0.0, 0.0, 0.0]
        if len(vel_cmd) < 3:
            vel_cmd = (vel_cmd + [0.0] * 3)[:3]
        yaw_rate_cmd = float(msg.get("yawRateCmd", 0.0))
        if np.isnan(yaw_rate_cmd):
            yaw_rate_cmd = 0.0

    # Current z for altitude hold
    pos_z = float(quads.pos[0, 2])
    desired = {
        "pos": [np.nan, np.nan, pos_z],
        "vel": vel_cmd,
        "yaw_rate": yaw_rate_cmd,
    }
    traj.desiredState(t, Ts, quads, desired=desired)
    ctrl.controller(traj, quads, Ts)
    quads.update(t, Ts, ctrl.w_cmd, wind)
    return t + Ts
    
def getStartOffboardInds(timestamp, data):
    offboard_inds = np.where(data == 1)[0]
    return offboard_inds[0]


def _load_real_flight_ulg(ulg_path, start_time_s=None):
    """Load ULog for real-flight comparison. start_time_s crops data (None = use full log)."""
    ulg_data = load_ulg(
        ulg_path,
        fields_to_extract=REAL_FLIGHT_FIELDS,
        startTime=start_time_s,
        verbose=True,
    )
    offstart_time = 50.0
    if "vehicle_control_mode_flag_control_offboard_enabled" in ulg_data:
        data = ulg_data["vehicle_control_mode_flag_control_offboard_enabled"]["data"]
        offboard_inds = np.where(data == 1)[0]
        if len(offboard_inds) > 0:
            offstart_time = ulg_data["vehicle_control_mode_flag_control_offboard_enabled"]["timestamp"][offboard_inds[0]]
    tf = 105.0
    for key, val in ulg_data.items():
        if "timestamp" in val and len(val["timestamp"]) > 0:
            tf = max(tf, float(val["timestamp"][-1]))
    return ulg_data, offstart_time, tf


def _real_flight_control_type(ulg_data):
    if (
        "vehicle_rates_setpoint_roll" in ulg_data
        and "vehicle_rates_setpoint_pitch" in ulg_data
        and "vehicle_rates_setpoint_yaw" in ulg_data
    ):
        return ControlType.ATT_RATE
    if (
        "vehicle_attitude_setpoint_roll_body" in ulg_data
        and "vehicle_attitude_setpoint_pitch_body" in ulg_data
        and "vehicle_attitude_setpoint_yaw_body" in ulg_data
    ):
        return ControlType.ATT
    if "vehicle_local_position_setpoint_vz" in ulg_data:
        return ControlType.XYZ_VEL
    return ControlType.XY_VEL_Z_POS


def _real_flight_traj_select(ctrl_type):
    traj_select = np.zeros(3)
    traj_select[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    traj_select[1] = YawTrajectoryType.FOLLOW.value
    traj_select[2] = WaypointTimeMode.AVERAGE_SPEED.value
    if ctrl_type == ControlType.ATT:
        traj_select[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    elif ctrl_type == ControlType.ATT_RATE:
        traj_select[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    elif ctrl_type == ControlType.XYZ_VEL:
        traj_select[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    else:
        traj_select[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    return traj_select


def run_real_flight_comparison(ulg_path):
    """Run ULG reproduction mode: XY_VEL_Z_POS, startTime=200, Tf=50. Uses cat_drone_config for controller gains (reference RL/t constants)."""
    start_time = time.time()
    ulg_data, _, _ = _load_real_flight_ulg(ulg_path, start_time_s=ULG_REPRODUCTION_START_TIME)

    # Load drone + control params from cat_drone_config (contains reference RL/t controller constants)
    drone_params = None
    control_params = None
    if os.path.isfile(DRONE_CONFIG_PATH):
        try:
            from load_drone_config import load_drone_config

            drone_params, control_params = load_drone_config(DRONE_CONFIG_PATH)
            logger.info("Loaded drone and control config from %s (reference RL/t gains)", DRONE_CONFIG_PATH)
        except Exception as e:
            logger.warning("Could not load drone config (%s): %s. Using defaults.", DRONE_CONFIG_PATH, e)

    Ti = 0
    Ts = 0.002
    Tf = ULG_REPRODUCTION_TF
    ifsave = 0

    # Force XY_VEL_Z_POS to match reference (velocity setpoints vx, vy, vz from vehicle_local_position_setpoint)
    ctrlType = ControlType.XY_VEL_Z_POS
    trajSelect = _real_flight_traj_select(ctrlType)
    logger.info("Real-flight comparison mode using control type: %s", ctrlType)

    quads = QuadcopterSwarm(numOfQuads=1, Ti=Ti, params=drone_params)
    traj = Trajectory(quads, ctrlType, trajSelect, ulgData=ulg_data)
    ctrl = Control(quads, traj.yawType, control_params=control_params)
    wind = Wind("None", 2.0, 90, -15)

    traj.desiredState(Ti, Ts, quads)
    ctrl.controller(traj, quads, Ts)

    numTimeStep = int(Tf / Ts + 1)
    t_all = np.zeros(numTimeStep)
    s_all = np.zeros([numTimeStep, quads.state.shape[0], quads.state.shape[1]])
    pos_all = np.zeros([numTimeStep, quads.pos.shape[0], quads.pos.shape[1]])
    vel_all = np.zeros([numTimeStep, quads.vel.shape[0], quads.vel.shape[1]])
    quat_all = np.zeros([numTimeStep, quads.quat.shape[0], quads.quat.shape[1]])
    omega_all = np.zeros([numTimeStep, quads.omega.shape[0], quads.omega.shape[1]])
    euler_all = np.zeros([numTimeStep, quads.euler.shape[0], quads.euler.shape[1]])
    sDes_traj_all = np.zeros([numTimeStep, traj.sDes.shape[0], traj.sDes.shape[1]])
    sDes_calc_all = np.zeros([numTimeStep, ctrl.sDesCalc.shape[0], ctrl.sDesCalc.shape[1]])
    w_cmd_all = np.zeros([numTimeStep, ctrl.w_cmd.shape[0], ctrl.w_cmd.shape[1]])
    # wMotor/thr/tor: makeFigures expects (N, 4, numOfQuads)
    wMotor_all = np.zeros([numTimeStep, 4, quads.numOfQuads])
    thr_all = np.zeros([numTimeStep, 4, quads.numOfQuads])
    tor_all = np.zeros([numTimeStep, 4, quads.numOfQuads])

    t_all[0] = Ti
    s_all[0] = quads.state
    pos_all[0] = quads.pos
    vel_all[0] = quads.vel
    quat_all[0] = quads.quat
    omega_all[0] = quads.omega
    euler_all[0] = quads.euler
    sDes_traj_all[0] = traj.sDes
    sDes_calc_all[0] = ctrl.sDesCalc
    w_cmd_all[0] = ctrl.w_cmd
    wMotor_all[0] = quads.wMotor if quads.wMotor.shape[0] == 4 else quads.wMotor.T
    thr_all[0] = quads.thr if quads.thr.shape[0] == 4 else quads.thr.T
    tor_all[0] = quads.tor if quads.tor.shape[0] == 4 else quads.tor.T

    t = Ti
    i = 1
    while round(t, 3) < Tf:
        t = quad_sim(t, Ts, quads, ctrl, wind, traj)
        try:
            t_all[i] = t
            s_all[i] = quads.state
            pos_all[i] = quads.pos
            vel_all[i] = quads.vel
            quat_all[i] = quads.quat
            omega_all[i] = quads.omega
            euler_all[i] = quads.euler
            sDes_traj_all[i] = traj.sDes
            sDes_calc_all[i] = ctrl.sDesCalc
            w_cmd_all[i] = ctrl.w_cmd
            wMotor_all[i] = quads.wMotor if quads.wMotor.shape[0] == 4 else quads.wMotor.T
            thr_all[i] = quads.thr if quads.thr.shape[0] == 4 else quads.thr.T
            tor_all[i] = quads.tor if quads.tor.shape[0] == 4 else quads.tor.T
        except IndexError as e:
            logger.warning("Simulation index mismatch at t=%.3f (i=%d): %s. Stopping.", t, i, e)
            break
        i += 1

    end_time = time.time()
    logger.info("Simulated %.2fs in %.6fs.", t, end_time - start_time)
    utils.makeFigures(quads.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all, ulgData=ulg_data)
    utils.plotComparisonWithUlg(t_all, euler_all[:, 0, :], omega_all[:, 0, :], ulgData=ulg_data)
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quads.params, traj.xyzType, traj.yawType, ifsave)


def test_all_trajectory_types():
    """Run simulation with each PositionTrajectoryType (trajSelect[0] = 0..13). No display."""
    matplotlib.use("Agg")
    # Load drone and control parameters
    drone_params = None
    control_params = None
    if os.path.isfile(DRONE_CONFIG_PATH):
        try:
            from load_drone_config import load_drone_config
            drone_params, control_params = load_drone_config(DRONE_CONFIG_PATH)
        except Exception:
            pass
    Ti = 0
    Ts = 0.003
    Tf = 5.0  # Short run to verify each trajectory type
    ctrlType = ControlType.XYZ_POS
    wind = Wind("None", 2.0, 90, -15)
    name_by_val = {e.value: e.name for e in PositionTrajectoryType}
    passed = 0
    failed = []
    for xyz_val in range(len(PositionTrajectoryType)):
        name = name_by_val.get(xyz_val, "?")
        trajSelect = np.array([xyz_val, YawTrajectoryType.FOLLOW.value, WaypointTimeMode.AVERAGE_SPEED.value])
        try:
            quads = QuadcopterSwarm(numOfQuads=1, Ti=Ti, params=drone_params)
            quads.setInitialQuadPos(np.array([0.0, 0.0, 0.0]), 0)
            traj = Trajectory(quads, ctrlType, trajSelect)
            ctrl = Control(quads, traj.yawType, control_params=control_params)
            traj.desiredState(0, Ts, quads)
            ctrl.controller(traj, quads, Ts)
            numTimeStep = int(Tf / Ts + 1)
            t = Ti
            for _ in range(numTimeStep - 1):
                if round(t, 3) >= Tf:
                    break
                t = quad_sim(t, Ts, quads, ctrl, wind, traj)
            logger.info("  trajSelect[0] = %s (%s) OK", xyz_val, name)
            passed += 1
        except Exception as e:
            logger.warning("  trajSelect[0] = %s (%s) FAIL: %s", xyz_val, name, e)
            failed.append((xyz_val, name, str(e)))
    logger.info("Result: %s/%s passed.", passed, len(PositionTrajectoryType))
    if failed:
        for xyz_val, name, err in failed:
            logger.warning("  - %s (%s): %s", xyz_val, name, err)
    return len(failed) == 0


def main():
    # Load drone and control parameters from JSON if present
    drone_params = None
    control_params = None
    if os.path.isfile(DRONE_CONFIG_PATH):
        try:
            from load_drone_config import load_drone_config
            drone_params, control_params = load_drone_config(DRONE_CONFIG_PATH)
            logger.info("Loaded drone and control config from %s", DRONE_CONFIG_PATH)
        except Exception as e:
            logger.warning("Could not load drone config (%s): %s. Using defaults.", DRONE_CONFIG_PATH, e)

    # When using system_manager, only first quad (index 0) is controlled; use 1 quad.
    # When drone config is loaded from DRONE_CONFIG_PATH, use single drone.
    numOfQuads = 1 if (USE_SYSTEM_MANAGER or drone_params is not None) else 4
    Ti = 0
    Ts = 0.003
    Tf = 27
    quads = QuadcopterSwarm(numOfQuads=numOfQuads, Ti=Ti, params=drone_params)
    quads.setInitialQuadPos(np.array([0, 0, 0]), 0)
    if numOfQuads > 1:
        quads.setInitialQuadPos(np.array([10, 0, 0]), 1)
        quads.setInitialQuadPos(np.array([0, 10, 0]), 2)
        quads.setInitialQuadPos(np.array([0, 0, 10]), 3)
    start_time = time.time()

    # Simulation Setup
    # ---------------------------
    Ti = 0
    Tf = Tf if Tf is not None else 50
    ifsave = 0

    # Choose trajectory settings
    # ---------------------------
    trajSelect = np.zeros(3)

    # Position Trajectory Type options:
    #   PositionTrajectoryType.HOVER (0), POS_WAYPOINT_TIMED (1), POS_WAYPOINT_INTERP (2),
    #   MINIMUM_VELOCITY (3), MINIMUM_ACCEL (4), MINIMUM_JERK (5), MINIMUM_SNAP (6),
    #   MINIMUM_ACCEL_STOP (7), MINIMUM_JERK_STOP (8), MINIMUM_SNAP_STOP (9),
    #   MINIMUM_JERK_FULL_STOP (10), MINIMUM_SNAP_FULL_STOP (11),
    #   POS_WAYPOINT_ARRIVED (12), POS_WAYPOINT_ARRIVED_WAIT (13)
    trajSelect[0] = PositionTrajectoryType.POS_WAYPOINT_TIMED.value
    # Yaw Trajectory Type options:
    #   YawTrajectoryType.NONE (0), YAW_WAYPOINT_TIMED (1), YAW_WAYPOINT_INTERP (2),
    #   FOLLOW (3), ZERO (4)
    trajSelect[1] = YawTrajectoryType.NONE.value
    # Waypoint Time Mode options:
    #   WaypointTimeMode.WAYPOINT_TIME (0), AVERAGE_SPEED (1)
    trajSelect[2] = WaypointTimeMode.AVERAGE_SPEED.value
 
    if USE_SYSTEM_MANAGER:
        ctrlType = ControlType.SYSTEM_MANAGER
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        from system_manager_adapter import flight_data_from_swarm
        from system_manager import System_Manager, load_config_from_json
        # Mission from JSON if path set and file exists; else minimal defaults
        if SYSTEM_MANAGER_MISSION_CONFIG and os.path.isfile(SYSTEM_MANAGER_MISSION_CONFIG):
            sys_manager_config = load_config_from_json(SYSTEM_MANAGER_MISSION_CONFIG)
            if sys_manager_config is None:
                sys_manager_config = {}
            # Override for simulation: our log dir, sim timestep, sim start time
            sys_manager_config = dict(sys_manager_config)
            sys_manager_config["log_dir"] = log_dir
            sys_manager_config["loop_period"] = Ts
            sys_manager_config["currentTime"] = 0
            logger.info("Loaded system_manager mission from %s", SYSTEM_MANAGER_MISSION_CONFIG)
        else:
            sys_manager_config = {
                "primaryControllerType": "VELOCITYPID",
                "missionType": "WAYPOINT",
                "loop_period": Ts,
            }
        sys_manager = System_Manager(log_dir=log_dir, currentTime=0, config_dict=sys_manager_config, simulation_mode=True)
    else:
        # Select Control Type
        # ControlType: XYZ_POS, XY_VEL_Z_POS, XYZ_VEL, ATT, ATT_RATE
        ctrlType = ControlType.XYZ_POS
        sys_manager = None
        flight_data_from_swarm = None

    logger.info("Control type: %s", ctrlType)
    if USE_SYSTEM_MANAGER:
        logger.info("Using system_manager as high-level controller (non-realtime, in-process).")

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    traj = Trajectory(quads, ctrlType, trajSelect)
    ctrl = Control(quads, traj.yawType, control_params=control_params)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States and first commands
    # ---------------------------
    if USE_SYSTEM_MANAGER:
        flight_Data = flight_data_from_swarm(quads, 0, Ti)
        msg = sys_manager.sys_manager_step(flight_Data=flight_Data, curTime=Ti, log_data=False)
        if msg and "velCmd" in msg:
            v = list(msg["velCmd"]) if hasattr(msg["velCmd"], "__iter__") else [0.0, 0.0, 0.0]
            if len(v) < 3:
                v = (v + [0.0] * 3)[:3]
            yaw_rate = float(msg.get("yawRateCmd", 0.0))
            if np.isnan(yaw_rate):
                yaw_rate = 0.0
            desired = {"pos": [np.nan, np.nan, float(quads.pos[0, 2])], "vel": v, "yaw_rate": yaw_rate}
        else:
            desired = {"pos": [np.nan, np.nan, float(quads.pos[0, 2])], "vel": [0.0, 0.0, 0.0], "yaw_rate": 0.0}
        traj.desiredState(Ti, Ts, quads, desired=desired)
    else:
        sDes = traj.desiredState(0, Ts, quads) 
    ctrl.controller(traj, quads, Ts)
    
    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, quads.state.shape[0], quads.state.shape[1]])
    pos_all        = np.zeros([numTimeStep, quads.pos.shape[0], quads.pos.shape[1]])
    vel_all        = np.zeros([numTimeStep, quads.vel.shape[0], quads.vel.shape[1]])
    quat_all       = np.zeros([numTimeStep, quads.quat.shape[0], quads.quat.shape[1]])
    omega_all      = np.zeros([numTimeStep, quads.omega.shape[0], quads.omega.shape[1]])
    euler_all      = np.zeros([numTimeStep, quads.euler.shape[0], quads.euler.shape[1]])
    sDes_traj_all  = np.zeros([numTimeStep, traj.sDes.shape[0], traj.sDes.shape[1]])
    sDes_calc_all  = np.zeros([numTimeStep, ctrl.sDesCalc.shape[0], ctrl.sDesCalc.shape[1]])
    w_cmd_all      = np.zeros([numTimeStep, ctrl.w_cmd.shape[0], ctrl.w_cmd.shape[1]])
    # makeFigures expects (N, 4, numOfQuads) for wMotor/thr/tor
    wMotor_all     = np.zeros([numTimeStep, 4, quads.numOfQuads])
    thr_all        = np.zeros([numTimeStep, 4, quads.numOfQuads])
    tor_all        = np.zeros([numTimeStep, 4, quads.numOfQuads])

    t_all[0]            = Ti
    s_all[0]          = quads.state
    pos_all[0]        = quads.pos
    vel_all[0]        = quads.vel
    quat_all[0]       = quads.quat
    omega_all[0]      = quads.omega
    euler_all[0]      = quads.euler
    sDes_traj_all[0]  = traj.sDes
    sDes_calc_all[0]  = ctrl.sDesCalc
    w_cmd_all[0]      = ctrl.w_cmd
    wMotor_all[0]     = quads.wMotor if quads.wMotor.shape[0] == 4 else quads.wMotor.T
    thr_all[0]        = quads.thr if quads.thr.shape[0] == 4 else quads.thr.T
    tor_all[0]        = quads.tor if quads.tor.shape[0] == 4 else quads.tor.T

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    while round(t, 3) < Tf:
        if USE_SYSTEM_MANAGER:
            t = quad_sim_system_manager(t, Ts, quads, ctrl, wind, traj, sys_manager, flight_data_from_swarm)
        else:
            t = quad_sim(t, Ts, quads, ctrl, wind, traj)

        try:
            t_all[i]             = t
            s_all[i]             = quads.state
            pos_all[i]           = quads.pos
            vel_all[i]           = quads.vel
            quat_all[i]          = quads.quat
            omega_all[i]         = quads.omega
            euler_all[i]         = quads.euler
            sDes_traj_all[i]     = traj.sDes
            sDes_calc_all[i]     = ctrl.sDesCalc
            w_cmd_all[i]         = ctrl.w_cmd
            wMotor_all[i]        = quads.wMotor.T
            thr_all[i]           = quads.thr.T
            tor_all[i]           = quads.tor.T
        except IndexError as e:
            logger.warning("Simulation index mismatch at t=%.3f (i=%d): %s. Stopping.", t, i, e)
            break
        i += 1

    end_time = time.time()
    logger.info("Simulated %.2fs in %.6fs.", t, end_time - start_time)

    # View Results
    # ---------------------------
    utils.makeFigures(quads.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quads.params, traj.xyzType, traj.yawType, ifsave)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "--test-all-traj":
        logger.info("Testing all PositionTrajectoryType options (trajSelect[0] = 0..13)...")
        ok = test_all_trajectory_types()
        sys.exit(0 if ok else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "--real-flight-compare":
        ulg_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REAL_FLIGHT_ULG
        logger.info("Running real-flight comparison with %s", ulg_path)
        run_real_flight_comparison(ulg_path)
        sys.exit(0)
    if config.orient in ("NED", "ENU"):
        main()
    else:
        raise ValueError("Invalid orientation %r. Verify config.py (use NED or ENU)." % config.orient)