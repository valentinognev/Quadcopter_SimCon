# -*- coding: utf-8 -*-
"""
Adapter to build system_manager Flight_Data from Quadcopter_SimCon QuadcopterSwarm state.
Used when running system_manager as the high-level controller in non-realtime simulation.
"""

import os
import sys
import numpy as np

# Add system_manager package path so we can import common
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Quadcopter_SimCon/Simulation -> go up to CatSwarm, then into system_manager/system_managerPY
_SYSTEM_MANAGER_PY = os.path.join(_SCRIPT_DIR, "..", "..", "system_manager", "system_managerPY")
if os.path.isdir(_SYSTEM_MANAGER_PY) and _SYSTEM_MANAGER_PY not in sys.path:
    sys.path.insert(0, _SYSTEM_MANAGER_PY)

try:
    from common import (
        Flight_Data,
        Quaternion,
        NED,
        Imu,
        LLA,
        Altitude,
        PX4_FLIGHT_STATE,
        FLIGHT_MODE,
        omega_frd2rpyRate,
    )
except ImportError as e:
    raise ImportError(
        "system_manager_adapter requires system_manager package. "
        "Ensure system_manager is at CatSwarm/system_manager/system_managerPY. "
        f"Error: {e}"
    ) from e


def flight_data_from_swarm(quads, quad_index, t):
    """
    Build a system_manager Flight_Data object from one quad in a QuadcopterSwarm.

    Used when calling System_Manager.sys_manager_step(flight_Data=..., curTime=t)
    for non-realtime in-process simulation.

    Args:
        quads: QuadcopterSwarm instance (pos, vel, quat, euler, omega, acc, thr, etc.)
        quad_index: Index of the quad to use (0 for first quad).
        t: Simulation time (seconds), used for timestamps.

    Returns:
        Flight_Data instance populated so that sys_manager_step does not return early
        (gathered flags set, OFFBOARD mode so guidance runs).
    """
    flight_Data = Flight_Data()

    flight_Data.local_ts = t
    flight_Data.timestamp = t
    flight_Data.imu_ts = t

    # Position and velocity NED
    pos = np.asarray(quads.pos[quad_index], dtype=float)
    vel = np.asarray(quads.vel[quad_index], dtype=float)
    flight_Data.pos_ned_m = NED(ned=pos.copy(), vel_ned=vel.copy(), timestamp=t)
    flight_Data.pos_ned_m.ned = pos
    flight_Data.pos_ned_m.vel_ned = vel

    # Quaternion: QuadcopterSimCon uses [w, x, y, z] in state 3:7
    q = quads.quat[quad_index]
    flight_Data.quat_ned_bodyfrd = Quaternion(
        w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])
    )
    flight_Data.quat_ned_bodyfrd.timestamp = t

    # Euler (roll, pitch, yaw) and rpy_rates
    euler = np.asarray(quads.euler[quad_index], dtype=float)
    omega = np.asarray(quads.omega[quad_index], dtype=float)
    flight_Data.rpy = euler.copy()
    flight_Data.rpy_rates = np.asarray(
        omega_frd2rpyRate(euler, omega), dtype=float
    )

    # Heading (yaw) in radians
    flight_Data.heading = float(euler[2])

    # Altitude fields (use z for NED)
    flight_Data.relative_m = float(pos[2])
    flight_Data.amsl_m = float(pos[2])
    flight_Data.local_m = float(pos[2])
    flight_Data.monotonic_m = float(pos[2])
    flight_Data.terrain_m = float(pos[2])
    flight_Data.bottom_clearance_m = float(pos[2])
    flight_Data.altitude_m = Altitude(float(pos[2]), float(pos[2]), timestamp=t)

    # IMU NED: accel and gyro (omega in body FRD)
    acc = getattr(quads, "acc", None)
    if acc is not None and acc.shape[0] > quad_index:
        acc_ned = np.asarray(acc[quad_index], dtype=float)
    else:
        acc_ned = np.zeros(3)
    flight_Data.imu_ned = Imu(
        timestamp=t,
        accel=acc_ned,
        gyro=omega.copy(),
    )

    # Throttle: approximate from thrust if available (body z thrust or sum)
    thr = getattr(quads, "thr", None)
    if thr is not None:
        # thr is (4, numOfQuads) per quad - use sum of motor thrusts as proxy
        try:
            total_thr = float(np.sum(quads.thr[:, quad_index]))
            flight_Data.throttle = total_thr  # or scale to percentage if needed
            flight_Data.current_thrust = total_thr
        except Exception:
            flight_Data.throttle = 0.0
            flight_Data.current_thrust = 0.0
    else:
        flight_Data.throttle = 0.0
        flight_Data.current_thrust = 0.0

    flight_Data.groundspeed = float(np.linalg.norm(vel))

    # LLA placeholder (not used for NED sim)
    flight_Data.raw_pos_lla_deg = LLA(timestamp=t, lla=np.zeros(3))
    flight_Data.filt_pos_lla_deg = LLA(timestamp=t, lla=np.zeros(3))

    # OFFBOARD so system_manager runs guidance
    flight_Data.custom_mode_id = PX4_FLIGHT_STATE.OFFBOARD.value
    flight_Data.mode = FLIGHT_MODE.OFFBOARD
    flight_Data.offboardMode = True

    # Required so sys_manager_step does not return early
    flight_Data.gathered["quat_ned_bodyfrd"] = True
    flight_Data.gathered["pos_ned_m"] = True
    flight_Data.gathered["imu_ned"] = True

    return flight_Data
