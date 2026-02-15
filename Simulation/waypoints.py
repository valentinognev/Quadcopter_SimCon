# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi

# Support both relative and absolute imports
try:
    from . import config
except ImportError:
    import config

deg2rad = pi/180.0

def makeWaypoints(numOfQuads):
    
    v_average = 1

    # Segment durations (s) between waypoints; must be > 0 so times are strictly increasing
    t_ini = 3
    t_segments = np.array([4.0, 4.0, 4.0, 4.0])  # 4 segments for 5 waypoints, or use 5 segments for 6 waypoints
    
    wp_ini = np.array([0, 0, 0])
    # wp = np.array([[2, 0, 0]])
    wp = np.array([[2, 2, 1],
                   [-2, 3, -3],
                   [-2, -1, -3],
                   [3, -2, 0],
                   wp_ini])
    # wp = wp[3,:]
    yaw_ini = 0    
    yaw = np.array([20, -90, 0, 0])

    # Build cumulative waypoint times (strictly increasing) so trajectory checks pass
    n_wp = wp.shape[0] + 1  # wp_ini + wp rows
    if len(t_segments) >= n_wp - 1:
        t_segments = t_segments[: n_wp - 1]
    else:
        t_segments = np.resize(t_segments, n_wp - 1)
    t = np.concatenate(([float(t_ini)], t_ini + np.cumsum(t_segments))).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.array([np.hstack((yaw_ini, yaw)).astype(float)*deg2rad])
    waypoints = np.outer(wp,np.ones(numOfQuads))
    
    xdelta = np.array(np.arange(numOfQuads))*3
    ydelta = np.zeros(numOfQuads)
    zdelta = np.zeros(numOfQuads)
    deltaposSingle = np.array([xdelta, ydelta, zdelta])
    deltapos = np.concatenate([deltaposSingle for _ in range(wp.shape[0])], axis=0)
    waypoints = waypoints + deltapos
    return np.outer(t,np.ones(numOfQuads)), \
           waypoints, \
           np.outer(yaw,np.ones(numOfQuads)), \
           np.outer(v_average,np.ones(numOfQuads))