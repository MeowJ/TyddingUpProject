#!/usr/bin/env python3

from automata.Eva import Eva
import time
import json
import math

host_ip = '172.16.172.1'
token = '71e699b1-cc5f-4fc5-900a-16e227bcfdd5'

eva = Eva(host_ip, token)

print('connected!')

start_joints_deg = [0.1, 11.16,-74.5,3.15,-115.14,-114.42]    
finish_joints_deg = [-169.83,31.58,-120.37,-7.65,-93.96,-110.53]
# Convert joint angles from deg to rad
start_joints_rad = []
for joint in start_joints_deg:
    start_joints_rad.append( round( joint * (math.pi/180),3) )

finish_joints_rad = []    
for joint in finish_joints_deg:
    finish_joints_rad.append( round( joint * (math.pi/180),3) )

# Send Eva to a waypoint
with eva.lock():
    eva.control_wait_for_ready()
    eva.control_go_to(start_joints_rad, True, 0.5, None)
    eva.control_wait_for_ready()
    eva.control_go_to(finish_joints_rad, True, 0.2, None)
    print("pose1")