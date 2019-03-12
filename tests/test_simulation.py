import pytest
from tests.test_env import *
from D3HRE.simulation import PowerSim

power_sim = PowerSim(test_task, config)

def test_run():
    assert  power_sim.run(10, 10, 1000) == 0

def test_get_result():
    assert len(power_sim.get_report(10, 10, 1000).columns) == 40

