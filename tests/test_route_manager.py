import pytest
import numpy as np

from mission_utlities import route_manager

test_route = np.array(
    [[9.20628817, 171.58565184],
     [9.48566359, 174.60574911],
     [9.95078073, 176.68206597],
     [10.69358, 178.94713892],
     [11.06430687, -176.90022735],
     [10.87900106, -172.27570342],
     [9.95078073, -168.97247204],
     [9.67178793, -166.89615517],
     [8.92669178, -164.53670418],
     [8.36686209, -163.12103359],
     [7.61917834, -161.61098496],
     [7.05755065, -160.66720457],
     [6.30766145, -159.15715593],
     [5.93230149, -158.1189975],
     [-1.60710319, -156.04268063]])

# MD5 hash checksum is use to link a mission and download file
# The idea is it is hard to give a name of a route or just by name them with
# first way points. We can check the sum of a mission, it will be a unique
# number that help us to track the download file for each mission
# To make possible to store and reuse the data the hash should be right

test_mission = route_manager.get_position_df('2014-01-01', test_route, 2)

def test_hash():
    assert route_manager.hashFor(test_route)[:7] == 'fc34214'
    assert route_manager.hashFor(test_mission)[:7] == 'c51dd51'


@pytest.mark.parametrize("test_input, expected",[
    ([1, 1, 2, 2], 157.2),
    ([2, 2, 3, 3], 157.2),
    ([2, 2, 100, -90], 9750),
     ])
def test_haversine(test_input, expected):
    assert route_manager.haversine(test_input[0], test_input[1],
                                   test_input[2], test_input[3]) == pytest.approx(expected, 0.5)

