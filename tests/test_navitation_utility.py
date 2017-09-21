import pytest

from D3HRE.core.navigation_utility import *


@pytest.mark.parametrize("test_input, expected",[
    ([(1, 1), (2, 2)], 44.978),
    ([(2, 2), (2, 2)], 0),
    ([(2, 2), (100, -90)], 10.002),
     ])


def test_compass_bearing(test_input, expected):
    assert calculate_initial_compass_bearing(test_input[0], test_input[1]) == pytest.approx(expected, 0.1)