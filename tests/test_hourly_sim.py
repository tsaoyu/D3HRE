import pytest

import numpy as np
import hourly_sim
import pandas as pd
from math import fmod



# Full day cut truncate Pandas DataFrame into full day DataFrame
# Since time resolution in hourly simulation is one hour, the result
# DataFrame should have a length that can be exactly divided by 24
@pytest.mark.parametrize("test_input",[10, 30, 60])
def test_full_day_cut(test_input):
    a = [0] * test_input
    df = pd.DataFrame(a)
    assert fmod(hourly_sim.full_day_cut(df).shape[0], 24) == 0



@pytest.mark.parametrize("test_input, expected_output1, expected_output2",[
    (10, 900, 800), # Limited charge - energy wasted
    (50, 500, 500), # Normal charge
    (100, 0 , 0),   # Energy balance no power stored
    (120, 0,  0)])  # Energy demand too high no energy supplied by battery
def test_min_max_battery_model(test_input, expected_output1, expected_output2):
    p = [100]*10
    power = pd.Series(p)
    energy = np.array(hourly_sim.min_max_model(power, test_input, 100000))
    assert energy[-1] == expected_output1
    energy = np.array(hourly_sim.min_max_model(power, test_input, 1000))
    assert energy[-1] == expected_output1
    energy = np.array(hourly_sim.min_max_model(power, test_input, 800))
    assert energy[-1] == expected_output2

@pytest.mark.parametrize("test_input",[10, 30, 60])
def test_soc_fixed_battery_model(test_input):
    u = test_input
    sim_time = 100
    p = np.random.rand(100) * sim_time
    power = pd.Series(p)
    SOC, energy_history, unmet_history, waste_history, use_history =\
        hourly_sim.soc_model_fixed_load(power, u, 1000, depth_of_discharge=1,
                                        discharge_rate=0, battery_eff=1, discharge_eff=1)
    assert len(energy_history) == len(unmet_history)
    assert len(unmet_history) == len(waste_history)
    assert len(waste_history) == len(use_history)
    # energy conservation check p_in = p_out
    # p_in = p_solar + p_wind
    # p_out = p_use + p_waste + E(t)
    # where E(t) is energy of battery at that time
    assert pytest.approx(p.sum() - energy_history[-1] + np.array(waste_history).sum() + np.array(use_history).sum())
    lpsp_from_use = use_history.count(0)/sim_time
    lpsp_from_unmet = 1 - unmet_history.count(0)/sim_time
    assert pytest.approx(lpsp_from_unmet-lpsp_from_use)
    # state of charge reflect actual energy in battery
