from tests.test_env import *

from D3HRE.management import Dynamic_environment, Reactive_follow_management, Absolute_follow_management, Finite_horizon_optimal_management, EWMA_management
from D3HRE.optimization import Constraint_mixed_objective_optimisation
from D3HRE.simulation import Reactive_simulation
from D3HRE.core.battery_models import Battery_managed


# -------------------------------------------------------------------------------------
#
#
#
#
#
# -------------------------------------------------------------------------------------

con_mix_opt = Constraint_mixed_objective_optimisation(test_task, config=config)

#
#
#
#
#
#
#


champion, champion_x = con_mix_opt.run()
solar_area, wind_area, battery_capacity = champion_x

# ---------------------------------------------------------------------------------------
#
#
#
#
#
# ---------------------------------------------------------------------------------------

battery = Battery_managed(battery_capacity, config=config)

#
#
#
#
#
#
#

rea_sim = Reactive_simulation(test_task, config=config)

#
#
#
#
#
#
#

result_df = rea_sim.result(solar_area, wind_area, battery_capacity)

#
#
#
#
#
#

resource = (result_df.wind_power + result_df.solar_power)
demand = (result_df.Load_demand).tolist()



def test_absolute_follow_managemet():

    management = Absolute_follow_management()
    b1 = battery.copy()
    env = Dynamic_environment(b1, resource, management)
    env.set_demand(result_df)
    env.step_over_time()
    env.simulation_result()


def test_reactive_follow_management():

    management = Reactive_follow_management(demand)
    b2 = battery.copy()
    env = Dynamic_environment(b2, resource, management)
    env.set_demand(result_df)
    env.step_over_time()
    env.simulation_result()



def test_global_finite_horizon_optimal_management():
    management = Finite_horizon_optimal_management(resource.index, config=config)
    b3 = battery.copy()
    env = Dynamic_environment(b3, resource, management)
    env.set_demand(result_df)
    env.step_over_time()
    env.simulation_result()

def test_ewm_management():
    management = EWMA_management()
    b4 = battery.copy()
    env = Dynamic_environment(b4, resource, management)
    env.set_demand(result_df)
    env.step_over_time()
    print(env.simulation_result())
    print(env.total_reward)


test_ewm_management()