# -*- coding: utf-8 -*-
"""
Load drone and control parameters from a JSON config file.
Provides the data needed by the drone model (QuadcopterSwarm / initQuad) and the Control class.
"""

import json
import os
import numpy as np
from numpy import pi

deg2rad = pi / 180.0


def load_drone_config(config_path=None):
    """
    Load drone and control parameters from a JSON file.

    Args:
        config_path: Path to the JSON config file. If None, uses 'drone_config.json'
                     in the same directory as this module.

    Returns:
        tuple: (drone_params, control_params)
            - drone_params: dict suitable for QuadcopterSwarm (includes computed
              fields like mixerFM, invI, maxThr, w_hover after build_drone_params).
            - control_params: dict of gain and limit arrays for Control class.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    drone_params = _build_drone_params(data.get("drone", {}))
    control_params = _build_control_params(data.get("control", {}))
    return drone_params, control_params


def _build_drone_params(raw):
    """Build full drone params dict from JSON 'drone' section (with computed fields)."""
    try:
        from .quadFiles.initQuad import makeMixerFM, init_cmd
    except ImportError:
        from quadFiles.initQuad import makeMixerFM, init_cmd

    IB = np.array(raw["IB"], dtype=float)
    params = {
        "mB": float(raw["mB"]),
        "g": float(raw["g"]),
        "dxm": float(raw["dxm"]),
        "dym": float(raw["dym"]),
        "dzm": float(raw["dzm"]),
        "IB": IB,
        "invI": np.linalg.inv(IB),
        "IRzz": float(raw["IRzz"]),
        "Cd": float(raw.get("Cd", 0.0)),
        "kTh": float(raw["kTh"]),
        "kTo": float(raw.get("kTo", raw["kTh"] * 0.5)),
        "HoverThr": float(raw["HoverThr"]),
        "tau": float(raw.get("tau", 0.054)),
        "useIntergral": bool(raw.get("useIntergral", True)),
    }
    params["mixerFM"] = makeMixerFM(params)
    params["mixerFMinv"] = np.linalg.inv(params["mixerFM"])
    params["minThr"] = float(raw.get("minThr", 0.1))
    params["maxThr"] = params["mB"] * params["g"] / params["HoverThr"]
    params["minWmotor"] = np.sqrt(params["minThr"] / 4 / params["kTh"])
    params["maxWmotor"] = np.sqrt(params["maxThr"] / 4 / params["kTh"])

    ini_hover = init_cmd(params)
    params["FF"] = ini_hover[0]
    params["w_hover"] = ini_hover[1]
    params["thr_hover"] = ini_hover[2]
    # tor_hover = ini_hover[3]  # stored in quad from init_cmd at build time if needed

    return params


def _build_control_params(raw):
    """Build control params dict from JSON 'control' section (numpy arrays, correct units)."""
    if not raw:
        return None
    tiltMax_deg = raw.get("tiltMax_deg", 50.0)
    rateMax_deg_s = raw.get("rateMax_deg_s", [2000.0, 2000.0, 1500.0])
    return {
        "pos_P_gain": np.array(raw["pos_P_gain"], dtype=float),
        "vel_P_gain": np.array(raw["vel_P_gain"], dtype=float),
        "vel_D_gain": np.array(raw["vel_D_gain"], dtype=float),
        "vel_I_gain": np.array(raw["vel_I_gain"], dtype=float),
        "vel_FF_gain": np.array(raw["vel_FF_gain"], dtype=float),
        "vel_FF_dot_gain": np.array(raw["vel_FF_dot_gain"], dtype=float),
        "vel_sp_dot_lpf_cutoff": float(raw.get("vel_sp_dot_lpf_cutoff", 15.0)),
        "att_P_gain": np.array(raw["att_P_gain"], dtype=float).copy(),
        "rate_P_gain": np.array(raw["rate_P_gain"], dtype=float),
        "rate_D_gain": np.array(raw["rate_D_gain"], dtype=float),
        "rate_FF_gain": float(raw.get("rate_FF_gain", 0.0)),
        "rate_FF_dot_gain": float(raw.get("rate_FF_dot_gain", 0.25)),
        "velMax": np.array(raw["velMax"], dtype=float),
        "velMaxAll": float(raw.get("velMaxAll", 15.0)),
        "saturateVel_separetely": bool(raw.get("saturateVel_separetely", False)),
        "tiltMax": float(tiltMax_deg) * deg2rad,
        "rateMax": np.array(rateMax_deg_s, dtype=float) * deg2rad,
    }
