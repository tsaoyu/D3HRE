import pytest
import numpy as np
import pandas as pd
import ruamel.yaml as yaml

from PyResis import propulsion_power
from D3HRE.simulation import Reactive_simulation, Task
from D3HRE.optimization import Constraint_mixed_objective_optimisation, Mixed_objective_optimization_function
from D3HRE.core import file_reading_utility
from D3HRE.core.mission_utility import Mission

from D3HRE.optimization import *
from PyResis import propulsion_power


test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = Mission('2014-01-01', test_route, 2)
test_ship = propulsion_power.Ship()
test_ship.dimension(5.72, 0.248, 0.76, 1.2, 5.72/(0.549)**(1/3),0.613)

sbo = Simulation_based_optimization(test_route, '2014-01-01',2 ,40, ship=test_ship)

power_consumption_list = {'single_board_computer': {'power': [2, 10], 'duty_cycle': 0.5},
                              'webcam': {'power': [0.6], 'duty_cycle': 1},
                              'gps': {'power': [0.04, 0.4], 'duty_cycle': 0.9},
                              'imu': {'power': [0.67, 1.1], 'duty_cycle': 0.9}}

config = {'load': {'prop_load': {'prop_eff': 0.7,
   'sea_margin': 0.2,
   'temperature': 25}},
 'optimization': {'constraints': {'turbine_diameter_ratio': 1.2,
   'volume_factor': 0.1,
   'water_plane_coff': 0.88},
  'cost': {'battery': 1, 'lpsp': 10000, 'solar': 210, 'wind': 320},
  'method': {'nsga': {'cr': 0.95, 'eta_c': 10, 'eta_m': 50, 'm': 0.01},
   'pso': {'eta1': 2.05,
    'eta2': 2.05,
    'generation': 100,
    'max_vel': 0.5,
    'neighb_param': 4,
    'neighb_type': 2,
    'omega': 0.7298,
    'population': 100,
    'variant': 5}},
  'safe_factor': 0.2},
 'simulation': {'battery': {'B0': 1,
   'DOD': 0.9,
   'SED': 500,
   'eta_in': 0.9,
   'eta_out': 0.8,
   'sigma': 0.005},
  'coupling': {'eff': 0.05}},
 'source': {'solar': {'brl_parameters': {'a0': -5.32,
    'a1': 7.28,
    'b1': -0.03,
    'b2': -0.0047,
    'b3': 1.72,
    'b4': 1.08}}},
 'transducer': {'solar': {'azim': 0,
   'eff': {'k_1': -0.017162,
    'k_2': -0.040289,
    'k_3': -0.004681,
    'k_4': 0.000148,
    'k_5': 0.000169,
    'k_6': 5e-06},
   'loss': 0.1,
   'power_density': 140,
   'tacking': 0,
   'tech': 'csi',
   'tilt': 0},
  'wind': {'power_coef': 0.3,
   'thurse_coef': 0.6,
   'v_in': 2,
   'v_off': 45,
   'v_rate': 15}}}

task = Task(test_mission, test_ship, power_consumption_list)



def test_constraint_mixed_objective_optimisation():
    con_mix_opt = Constraint_mixed_objective_optimisation(task, config=config)
    mix_opt = Mixed_objective_optimization_function(task, config=config)
    champion, champion_x = con_mix_opt.run()
    for opt_x, constraint_x in zip(champion_x, mix_opt.constraints()):
        assert opt_x <= constraint_x




