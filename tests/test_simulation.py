import pytest
from PyResis import propulsion_power
import numpy as np


from D3HRE.core.mission_utility import Mission
from D3HRE.simulation import *


# Make sure that generated link matches the total length of original mission
test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])

test_ship = propulsion_power.Ship()
test_ship.dimension(5.72, 0.248, 0.76, 1.2, 5.72/(0.549)**(1/3),0.613)

power_consumption_list = {'single_board_computer': {'power': [2, 10], 'duty_cycle': 0.5},
                              'webcam': {'power': [0.6], 'duty_cycle': 1},
                              'gps': {'power': [0.04, 0.4], 'duty_cycle': 0.9},
                              'imu': {'power': [0.67, 1.1], 'duty_cycle': 0.9},
                              'sonar': {'power': [0.5, 50, 0.2], 'duty_cycle': 0.5},
                              'ph_sensor': {'power': [0.08, 0.1], 'duty_cycle': 0.95},
                              'temp_sensor': {'power': [0.04], 'duty_cycle': 1},
                              'wind_sensor': {'power': [0.67, 1.1], 'duty_cycle': 0.5},
                              'servo_motors': {'power': [0.4, 1.35], 'duty_cycle': 0.5},
                              'radio_transmitter': {'power': [0.5, 20], 'duty_cycle': 0.2}}



test_mission = Mission('2014-01-01', test_route, 2)
test_task = Task(test_mission, test_ship, power_consumption_list)


rea_sim = Reactive_simulation(test_task)
print(rea_sim.run(1,1,300))
#
#
# s = Sim(test_mission)
#
# def test_simulation():
#     s.sim_wind(3)
#     s.sim_solar(0, 0, 2, 100)
#     s.sim_all(20, 30)
#     pass
#
# test_mission.get_hotel_load(power_consumption_list)
# test_mission.get_propulsion_load(test_ship)
#
# rea_sim = Reactive_simulation(test_mission, test_ship, power_consumption_list)
# print(rea_sim.resource_df.columns.values)
#
# rea_sim.run(1,1, 300)