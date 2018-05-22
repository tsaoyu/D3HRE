from tests.test_env import *

from D3HRE.management import Dynamic_environment, Reactive_follow_management
from D3HRE.optimization import Constraint_mixed_objective_optimisation
from D3HRE.simulation import Reactive_simulation
from D3HRE.core.battery_models import Battery_managed


con_mix_opt = Constraint_mixed_objective_optimisation(test_task, config=config)

champion, champion_x = con_mix_opt.run()

solar_area, wind_area, battery_capacity = champion_x

battery = Battery_managed(battery_capacity, config=config)

rea_sim = Reactive_simulation(test_task, config=config)

result_df = rea_sim.result(solar_area, wind_area, battery_capacity)

resource = (result_df.wind_power + result_df.solar_power).tolist()

demand = (result_df.Load_demand).tolist()


def test_reactive_follow_management():
    management = Reactive_follow_management(demand)

    env = Dynamic_environment(battery, resource, management)

    env.step_over_time()

    print(env.simulation_result())

