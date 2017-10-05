import pytest

import numpy as np


from D3HRE.core.mission_utility import Mission
from D3HRE.simulation import *


# Make sure that generated link matches the total length of original mission
test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = Mission('2014-01-01', test_route, 2)

s = Sim(test_mission)

def test_simulation():
    s.sim_wind(3)
    s.sim_solar(0, 0, 2, 100)
    s.sim_all(20, 30)
    print(s.battery_energy)
    pass