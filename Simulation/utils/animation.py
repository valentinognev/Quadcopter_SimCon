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
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

# Support both relative and absolute imports
try:
    from .. import utils
    from .. import config
except ImportError:
    import utils
    import config

numFrames = 8

def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave):
    numOfQuads = pos_all.shape[1]
    
    # Collect all position data for axis limits calculation
    all_x = pos_all[:,:,0]
    all_y = pos_all[:,:,1]
    all_z = pos_all[:,:,2]
    
    if (config.orient == "NED"):
        all_z = -all_z
    
    # Create figure and axes once (outside the loop)
    fig = plt.figure(50)
    ax = p3.Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    
    # Create line objects for each quad
    # Each quad needs: line1 (arm1), line2 (arm2), line3 (trajectory)
    line1_list = []
    line2_list = []
    line3_list = []
    
    # Use different colors for each quad
    colors = plt.cm.tab10(np.linspace(0, 1, numOfQuads))
    
    for qi in range(numOfQuads):
        # Create lines for this quad with unique colors
        line1, = ax.plot([], [], [], lw=2, color=colors[qi])
        line2, = ax.plot([], [], [], lw=2, color=colors[qi])
        line3, = ax.plot([], [], [], '--', lw=1, color=colors[qi], alpha=0.7)
        line1_list.append(line1)
        line2_list.append(line2)
        line3_list.append(line3)
    
    # Setting the axes properties based on ALL quads
    extraEachSide = 0.5
    maxRange = 0.5*np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() + extraEachSide
    mid_x = 0.5*(all_x.max()+all_x.min())
    mid_y = 0.5*(all_y.max()+all_y.min())
    mid_z = 0.5*(all_z.max()+all_z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (config.orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
    elif (config.orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    # Plot waypoints and desired trajectories for all quads
    if (xyzType == 0):
        trajType = 'Hover'
    else:
        for qi in range(numOfQuads):
            x_wp = waypoints[0::3,qi]
            y_wp = waypoints[1::3,qi]
            z_wp = waypoints[2::3,qi]
            
            xDes = sDes_tr_all[:,0,qi]
            yDes = sDes_tr_all[:,1,qi]
            zDes = sDes_tr_all[:,2,qi]
            
            if (config.orient == "NED"):
                z_wp = -z_wp
                zDes = -zDes
            
            ax.scatter(x_wp, y_wp, z_wp, color=colors[qi], alpha=0.6, marker='o', s=25)
            
            if (xyzType == 1 or xyzType == 12):
                trajType = 'Simple Waypoints'
            else:
                ax.plot(xDes, yDes, zDes, ':', lw=1.3, color=colors[qi], alpha=0.5)
                if (xyzType == 2):
                    trajType = 'Simple Waypoint Interpolation'
                elif (xyzType == 3):
                    trajType = 'Minimum Velocity Trajectory'
                elif (xyzType == 4):
                    trajType = 'Minimum Acceleration Trajectory'
                elif (xyzType == 5):
                    trajType = 'Minimum Jerk Trajectory'
                elif (xyzType == 6):
                    trajType = 'Minimum Snap Trajectory'
                elif (xyzType == 7):
                    trajType = 'Minimum Acceleration Trajectory - Stop'
                elif (xyzType == 8):
                    trajType = 'Minimum Jerk Trajectory - Stop'
                elif (xyzType == 9):
                    trajType = 'Minimum Snap Trajectory - Stop'
                elif (xyzType == 10):
                    trajType = 'Minimum Jerk Trajectory - Fast Stop'
                elif (xyzType == 1):
                    trajType = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'

    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   

    def updateLines(i):
        time = t_all[i*numFrames]
        
        for qi in range(numOfQuads):
            pos = pos_all[i*numFrames,qi]
            x = pos[0]
            y = pos[1]
            z = pos[2]

            x_from0 = np.asarray(pos_all[0:i*numFrames,qi,0]).flatten()
            y_from0 = np.asarray(pos_all[0:i*numFrames,qi,1]).flatten()
            z_from0 = np.asarray(pos_all[0:i*numFrames,qi,2]).flatten()
        
            dxm = params["dxm"]
            dym = params["dym"]
            dzm = params["dzm"]
            
            quat = quat_all[i*numFrames,qi]
        
            if (config.orient == "NED"):
                z = -z
                z_from0 = -z_from0
                quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
        
            R = utils.quat2Dcm(quat)    
            motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
            motorPoints = np.dot(R, np.transpose(motorPoints))
            motorPoints[0,:] += x 
            motorPoints[1,:] += y 
            motorPoints[2,:] += z 
            
            # Extract x, y, z coordinates for the first arm (points 0-2)
            x1 = np.asarray(motorPoints[0,0:3]).flatten()
            y1 = np.asarray(motorPoints[1,0:3]).flatten()
            z1 = np.asarray(motorPoints[2,0:3]).flatten()
            line1_list[qi].set_data(x1, y1)
            line1_list[qi].set_3d_properties(z1)
            
            # Extract x, y, z coordinates for the second arm (points 3-5)
            x2 = np.asarray(motorPoints[0,3:6]).flatten()
            y2 = np.asarray(motorPoints[1,3:6]).flatten()
            z2 = np.asarray(motorPoints[2,3:6]).flatten()
            line2_list[qi].set_data(x2, y2)
            line2_list[qi].set_3d_properties(z2)
            
            # Update trajectory line
            line3_list[qi].set_data(x_from0, y_from0)
            line3_list[qi].set_3d_properties(z_from0)
        
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        
        # Return all lines for animation
        return line1_list + line2_list + line3_list + [titleTime]


    def ini_plot():
        for qi in range(numOfQuads):
            line1_list[qi].set_data(np.empty([1]), np.empty([1]))
            line1_list[qi].set_3d_properties(np.empty([1]))
            line2_list[qi].set_data(np.empty([1]), np.empty([1]))
            line2_list[qi].set_3d_properties(np.empty([1]))
            line3_list[qi].set_data(np.empty([1]), np.empty([1]))
            line3_list[qi].set_3d_properties(np.empty([1]))

        return line1_list + line2_list + line3_list + [titleTime]

    
    # Creating the Animation object (once, outside the loop)
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    if (ifsave):
        line_ani.save('Gifs/Raw/animation_{0:.0f}_{1:.0f}.gif'.format(xyzType,yawType), dpi=80, writer='imagemagick', fps=25)
        
    plt.show()
    return line_ani