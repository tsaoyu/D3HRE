from D3HRE.core.battery_models import Soc_model_variable_load, Battery, Battery_managed

from tests.test_env import *

b1 = Battery(10)
model = Soc_model_variable_load(b1, [1,2,3],[1,2,3])
model2 = Soc_model_variable_load(Battery(10), [1,1,1], [10, 10, 10])



def test_lpsp_calc():
    assert model.get_lost_power_supply_probability() == 0
    assert model2.get_lost_power_supply_probability() == 1

# w and w/o config
B = 10

managed_battery = Battery_managed(B)
managed_battery_with_config = Battery_managed(B, config=config)


def test_parameter_setting():
    # Battery capacity should match
    assert managed_battery.capacity == B
    assert managed_battery_with_config.capacity == B

    # Test parameter setting in managed battery object
    assert managed_battery.DOD == 1
    assert managed_battery.init_charge == 1

    assert managed_battery_with_config.DOD == config['simulation']['battery']['DOD']
    assert managed_battery_with_config.init_charge == config['simulation']['battery']['B0']





demand = list(range(15)) *2
power = [10]*30

def simulate_battery():
    managed_battery.reset()
    managed_battery_with_config.reset()
    for d, p in zip(demand, power):
        managed_battery.step(d, p)
        managed_battery_with_config.step(d, p)

def test_simualtion_reset():
    simulate_battery()
    managed_battery_with_config.reset()
    managed_battery.reset()

    assert managed_battery.SOC == []
    assert managed_battery_with_config.SOC == []




simulate_battery()

print(managed_battery.states_list)
print(managed_battery_with_config.states_list)

def test_finite_state_machine():
    DOD = config['simulation']['battery']['DOD']
    # Case 1 P(t) > U(t) && E(t+1) < B
    managed_battery_with_config.energy = 0.25 * B
    managed_battery_with_config.step(1, 4)
    assert managed_battery_with_config.state == 'charge'

    # Case 2 P(t) < U(t) && E(t+1) < (1 - DOD) B
    managed_battery_with_config.energy = 0.25 * B
    managed_battery_with_config.step(4, 1)
    assert managed_battery_with_config.state == 'unmet'

    # Case 2 special case
    #
    # P(t) < U(t) && E(t+1) < B
    managed_battery_with_config.energy = 0.60 * B
    managed_battery_with_config.step(4, 1)
    assert managed_battery_with_config.state == 'discharge'

    # Case 3 P(t) > U(t) && E(t+1) < B
    managed_battery_with_config.energy = 0.50 * B
    managed_battery_with_config.step(2, 4)
    assert managed_battery_with_config.state == 'charge'

    # Case 4 P(t) < U(t) &&  (1-DOD) B < E(t+1) < B
    managed_battery_with_config.energy = (1 - DOD)* B + 0.5
    managed_battery_with_config.step(4, 3.8)
    assert managed_battery_with_config.state == 'discharge'

    # Case 4 special case
    #
    # P(t) < U(t) &&   E(t+1) < (1-DOD) B < B
    managed_battery_with_config.energy = (1 - DOD) * B + 0.5
    managed_battery_with_config.step(4, 3.2)
    assert managed_battery_with_config.state == 'unmet'

    # Case 5 P(t) < U(t) &&  (1-DOD) B < E(t+1) < B
    managed_battery_with_config.energy = (1 - DOD) * B + 0.5
    managed_battery_with_config.step(4, 3.8)
    assert managed_battery_with_config.state == 'discharge'

    # Case 6 P(t) < U(t) &&  (1-DOD) B < E(t+1) < B
    managed_battery_with_config.energy = (1 - DOD) * B + 0.5
    managed_battery_with_config.step(4, 3.8)
    assert managed_battery_with_config.state == 'discharge'

    # Case 7 P(t) > U(t) &&   E(t+1) >= B
    managed_battery_with_config.energy = B - 0.1
    managed_battery_with_config.step(1, 4)
    assert managed_battery_with_config.state == 'float'

    # Case 8 Not possible


    # Case 9 the same as case 7
    managed_battery_with_config.energy = B - 0.1
    managed_battery_with_config.step(1, 4)
    assert managed_battery_with_config.state == 'float'


test_finite_state_machine()