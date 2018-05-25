import pytest
import numpy as np

from D3HRE.core.mission_utility import *

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

test_mission = get_mission('2014-01-01', test_route, 2)

mission = Mission('2014-01-01', test_route, 2)

print(mission.ID)

def test_hash_value():
    assert hash_value(test_route)[:7] == 'fc34214'
    assert mission.ID == '62d61d56dd476871993a1af318e96656'

def test_variable_speed():
    """
    Mission utility is designed to handle both constant speed and variable speed
    """
    speed = np.linspace(2, 5, num=test_route.shape[0]-1)
    variable_speed_mission = get_mission('2014-01-01', test_route, speed)
    assert variable_speed_mission.speed[0] == 2
    assert variable_speed_mission.speed[-1] == 5

def test_interpolation():
    """
    interpolation function need smart enough to deal with longitude singularity
    e.g from 178 -> -176 the expected behaviour was 178 -> 180 (-180) -> -176
    As a test, the interpolated mission should have a maximum value close to 180
    and a minimum value close to -180
    """
    assert test_mission.lon.min() == pytest.approx(-180, 0.1)
    assert test_mission.lon.max() == pytest.approx( 180, 0.1)



@pytest.mark.parametrize("test_input, expected",[
    ([1, 1, 2, 2], 157.2),
    ([2, 2, 3, 3], 157.2),
    ([2, 2, 100, -90], 9750),
     ])
def test_haversine(test_input, expected):
    assert haversine(test_input[0], test_input[1],
                        test_input[2], test_input[3]) == pytest.approx(expected, 0.5)

