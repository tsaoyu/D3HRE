import pytest
from D3HRE.core.mission_utility import Mission
from D3HRE.simulation import Task
from D3HRE.core.hotel_load_model import HotelLoad
from PyResis import propulsion_power

import numpy as np

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

test_task.get_hotel_load()
test_task.get_load_demand()
# h_normal = HotelLoad(test_mission, power_consumption_list)
# h_full =  HotelLoad(test_mission, power_consumption_list, 'full-power')
#
#
# test_mission = Mission('2014-01-01', test_route, 2, test_ship, power_consumption_list)
#
#
#
# def test_hotel_load():
#     ts_normal = h_normal.generate_power_consumption_timeseries()
#     ts_full =h_full.generate_power_consumption_timeseries()
#
#     assert len(ts_normal) == len(test_mission.df)
#     assert len(ts_full) == len(test_mission.df)
#
# def mission_load():
#     assert test_mission.get_load_demand() == 0 # get_load_demand return 0 because hotel load is missing
#     assert test_mission.get_hotel_load(power_consumption_list) == h_normal
#     assert test_mission.get_hotel_load(power_consumption_list, 'full-power') == h_full
#     assert test_mission.get_load_demand() == 0 #  still full load demand isn't avaliable yet
#     test_mission.get_propulsion_load(test_ship)
#     assert test_mission.get_load_demand() == h_normal + test_ship.prop_power()
