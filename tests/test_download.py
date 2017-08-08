# Test couldn't eliminate all errors in the program automatically,
# but it give programmer more confidence that it can make sure some functions works as expected.

import pytest
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from opendap_download import route_based_download
from mission_utlities import route_manager


# Make sure that generated link matches the total length of original mission
test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = route_manager.get_position_df('2014-01-01', test_route, 2)

def test_duration():
    start, end = route_based_download.download_URL(test_mission, debug=True)
    assert end - start == test_mission.index[-1] - test_mission.index[0]
    pass

def test_bearing_angle():
    # This is a walk on South pole
    pointA = (-85, 0)
    pointB = (-80, 0)
    pointC = (-80, 5)
    pointD = (-85, 5)
    assert route_based_download.calculate_initial_compass_bearing(pointA, pointB) == 0
    assert route_based_download.calculate_initial_compass_bearing(pointB, pointA) == 180
    assert abs(route_based_download.calculate_initial_compass_bearing(pointA, pointC)
               -9.924560325228924) < 0.0001
    assert abs(route_based_download.calculate_initial_compass_bearing(pointA, pointD)
               - 92.4904987465253) < 0.0001

for a in range(0,360,45):
    alpha = np.radians(a)
    print(a, ':', 1*np.sin(alpha),1*np.cos(alpha))

def test_wind_angles():
    pass