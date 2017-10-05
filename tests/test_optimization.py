import pytest

from D3HRE.optimization import *
from D3HRE.core.mission_utility import Mission

from PyResis.PyResis import propulsion_power


test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = Mission('2014-01-01', test_route, 2)
test_ship = propulsion_power.Ship()
test_ship.dimension(5.72, 0.248, 0.76, 1.2, 5.72/(0.549)**(1/3),0.613)

sbo = Simulation_based_optimization(test_route, '2014-01-01',2 ,40, ship=test_ship)

def test_optimization():
    sbo.run(discharge_rate=0.01, battery_eff=0.9, power_coefficient=0.28)
    pass