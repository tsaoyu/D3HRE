import pytest
from D3HRE.core.mission_utility import Mission
from D3HRE.core.hotel_load_model import HotelLoad
import numpy as np

test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = Mission('2014-01-01', test_route, 2)

h_normal = HotelLoad(test_mission)
h_full =  HotelLoad(test_mission, 'full-power')

def test_hotel_load():
    ts_normal = h_normal.generate_power_consumption_timeseries()
    ts_full =h_full.generate_power_consumption_timeseries()

    assert len(ts_normal) == len(test_mission.df)
    assert len(ts_full) == len(test_mission.df)