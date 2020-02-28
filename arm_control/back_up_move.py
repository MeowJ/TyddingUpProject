from automata.Eva import Eva
import time
import json
import logging
import numpy as np

from draw_functions import *
import math

print("Hi babe")

host_ip = '172.16.172.1'
token = '71e699b1-cc5f-4fc5-900a-16e227bcfdd5'

eva = Eva(host_ip, token)

start_joints_deg = [0.3,35.61,-98.41,-2.5,-78.68,70.56] # Read from Go-to on choreograph
# start_joints_deg = [8.58,-21.02,-131.61,5.03,-27.68,-91.19]

# Convert joint angles from deg to rad
start_joints_rad = []
for joint in start_joints_deg:
    start_joints_rad.append( round( joint * (math.pi/180),3) )
# "start_joints_rad = [0.15, -0.367, -2.297, 0.088, -0.483, -1.592]"

# Calculate starting end effector position
start_end_eff = eva.calc_forward_kinematics(start_joints_rad)

# !!check the usage of "start_end_eff"
init_orient_quat = start_end_eff['orientation']

# Convert end effector orientation from quaterion to euler
init_orient_euler = quaternion_to_euler(init_orient_quat['w'],init_orient_quat['x'],init_orient_quat['y'],init_orient_quat['z'])

# Create orientation object that is perpendicular to the floor,
new_quat = euler_to_quaternion(init_orient_euler[0],0,3.14) #yaw, pitch, roll

start_end_eff['orientation'] = {'w': new_quat[0], 'x': new_quat[1], 'y': new_quat[2], 'z': new_quat[3]}

#find joint angles associate with end effector positon with same, x,y,z, but adjusted orientation.
updated_start_joints = eva.calc_inverse_kinematics(start_joints_rad, start_end_eff['position'], start_end_eff['orientation'])['ik']['joints']

# ==========location of the object=============
x = input("Please enter x:")
y = input("Please enter y:")
z = input("Please enter z:")

x = float(x)
y = float(y)
z = float(z)
point_in = [0.0, 0.0, 0.0]
#point_object = [10, 13, -50]
point_object = [x, y, z]
point_out = [25, -20, -10, 5, -55, -15]
#point_out = [20, -20, -15, 20, -35, -15, 10, -55, -15, 5, -55, -15]
points = np.concatenate((point_in, point_object), axis=None)
points = np.concatenate((points, point_out), axis=None)
motion_type = ['linear','linear','spline','linear']

#Functions to turn points to toolpath
parsed_points = parse_fusion_points(points)
waypoints = points_to_waypoints(parsed_points,start_end_eff,updated_start_joints, eva)
print('waypoints')
print(waypoints)
toolpath = waypoints_to_toolpath(waypoints, motion_type)
print('toolpaths')
print(toolpath)

for i in toolpath["waypoints"]:
    print(i)
for j in toolpath["timeline"]:
    print(j)
# Run tool path
with eva.lock():
    eva.control_wait_for_ready()
    eva.toolpaths_use(toolpath)
    eva.control_home()
    eva.control_run(loop=1)


print("Bye")
