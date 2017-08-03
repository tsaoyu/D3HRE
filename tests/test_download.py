# Test couldn't eliminate all errors in the program automatically,
# but it give programmer more confidence that it can make sure some functions works as expected.

import pytest
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from opendap_download import route_based_download
from mission_utlities import route_manager

test_route =  route_manager.opencpn_coordinates_processing(
        "50.1871,26.1087,0. 52.5835,26.431,0. 54.1412,26.1087,0. 55.9985,26.5919,0. "
        "57.017,25.9472,0. 57.9157,24.8105,0. 60.3721,23.5535,0. 61.091,20.4993,0. 59.054,"
        "18.1817,0. 56.6575,16.4085,0. 52.5835,14.445,0. 49.4082,13.0484,0. 45.3941,11.937"
        ",0. 43.8364,12.23,0. 41.6197,15.1402,0. 39.8823,18.2955,0. 37.9651,21.7288,"
        "0. 36.1677,24.5928,0. 35.0294,26.431,0. 32.8726,29.2385,0. 30.7158,32.7302,"
        "0. 26.7616,33.8818,0. 20.7705,34.8216,0. 15.678,35.2631,0. 11.4841,37.3867,"
        "0. 6.99077,38.0029,0. 2.55731,37.6243,0. -1.27704,36.6211,0. -7.50785,35.7021,"
        "0. -10.9228,35.0672,0. -17.4532,34.8708,0. -24.8223,34.8708,0. -30.9333,34.4272,"
        "0. -38.1227,34.1797,0. -46.9297,33.483,0. -52.142,33.0321,0. -58.4927,32.4779,0."
        " -64.4838,32.0727,0. -70.0556,31.8694,0. -75.3278,31.8694,0. -78.6829,31.4614,0. "
        "-81.3789,31.4103,0. ")

# MD5 hash checksum is use to link a mission and download file
# The idea is it is hard to give a name of a route or just by name them with
# first way points. We can check the sum of a mission, it will be a unique
# number that help us to track the download file for each mission
# To make possible to store and reuse the data the hash should be right


test_mission = route_manager.get_position_df('2014-01-01', test_route, 2)

def test_hash():
    assert route_manager.hashFor(test_route)[:7] == 'f6141ae'
    assert route_manager.hashFor(test_mission)[:7] == 'e59a602'


# Make sure that generated link matches the total length of original mission

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

