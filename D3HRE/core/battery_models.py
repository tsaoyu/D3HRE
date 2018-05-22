import numpy as np

def min_max_model(power, use, battery_capacity):
    """
    Minimal maximum battery model
    :param power: Pandas TimeSeries of total power from renewable system
    :param use: float unit W fixed load of the power system
    :param battery_capacity: float unit Wh battery capacity
    :return: list energy history in battery
    """
    power = power.tolist()
    energy = 0
    energy_history = []
    for p in power:
        energy = min(battery_capacity, max(0, energy + (p - use) * 1))
        energy_history.append(energy)

    return energy_history


def soc_model_fixed_load(power, use, battery_capacity, depth_of_discharge=1,
                         discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8):
    """
    Battery state of charge model with fixed load.

    :param power: Pandas TimeSeries of total power from renewable system
    :param use: float unit W fixed load of the power system
    :param battery_capacity: float unit Wh battery capacity
    :param depth_of_discharge: float 0 to 1 maximum allowed discharge depth
    :param discharge_rate: self discharge rate
    :param battery_eff: optional 0 to 1 battery energy store efficiency default 0.9
    :param discharge_eff: battery discharge efficiency 0 to 1 default 0.8
    :return: tuple SOC: state of charge, energy history: E in battery,
    unmet_history: unmet energy history, waste_history: waste energy history
    """
    DOD = depth_of_discharge
    power = power.tolist()
    use_history = []
    waste_history = []
    unmet_history = []
    energy_history = []
    energy = 0
    for p in power:
        if p >= use:
            use_history.append(use)
            unmet_history.append(0)
            energy_new = energy * (1 - discharge_rate) + (p - use) * battery_eff
            if energy_new < battery_capacity:
                energy = energy_new  # battery energy got update
                waste_history.append(0)
            else:
                waste_history.append(p - use)
                energy = energy

        elif p < use:
            energy_new = energy * (1 - discharge_rate) + (p - use) / discharge_eff
            if energy_new > (1 - DOD) * battery_capacity:
                energy = energy_new
                unmet_history.append(0)
                waste_history.append(0)
                use_history.append(use)
            elif energy * (1 - discharge_rate) + p * battery_eff < battery_capacity:
                energy = energy * (1 - discharge_rate) + p * battery_eff
                unmet_history.append(use - p)
                use_history.append(0)
                waste_history.append(0)
            else:
                unmet_history.append(use - p)
                use_history.append(0)
                waste_history.append(p)
                energy = energy

        energy_history.append(energy)

    if battery_capacity == 0:
        SOC = np.array(energy_history)
    else:
        SOC = np.array(energy_history) / battery_capacity
    return SOC, energy_history, unmet_history, waste_history, use_history


class Battery():

    def __init__(self, capacity, config={}):
        self.capacity = capacity
        self.config = config
        self.set_parameters()


    def set_parameters(self):
        try :
            self.depth_of_discharge = self.config['simulation']['battery']['DOD']
            self.discharge_rate = self.config['simulation']['battery']['sigma']
            self.battery_eff = self.config['simulation']['battery']['eta_in']
            self.discharge_eff = self.config['simulation']['battery']['eta_out']
            self.init_charge =  self.config['simulation']['battery']['B0']


        except KeyError:
            print('Parameter is not found in config file, default values are used.')
            self.depth_of_discharge = 1
            self.discharge_rate = 0.005
            self.battery_eff = 0.9
            self.discharge_eff = 0.8
            self.init_charge = 1

    def run(self, power, use):

        DOD = self.depth_of_discharge
        battery_capacity = self.capacity
        discharge_rate = self.discharge_rate
        discharge_eff = self.discharge_eff
        battery_eff = self.battery_eff


        use_history = []
        waste_history = []
        unmet_history = []
        energy_history = []
        SOC = []
        energy = self.init_charge * self.capacity
        for p, u in zip(power, use):
            if p >= u:
                use_history.append(u)
                unmet_history.append(0)
                energy_new = energy * (1 - discharge_rate) + (p - u) * battery_eff
                if energy_new < battery_capacity:
                    energy = energy_new  # battery energy got update
                    waste_history.append(0)
                else:
                    waste_history.append(p - u)
                    energy = energy

            elif p < u:
                energy_new = energy * (1 - discharge_rate) + (p - u) / discharge_eff
                if energy_new > (1 - DOD) * battery_capacity:
                    energy = energy_new
                    unmet_history.append(0)
                    waste_history.append(0)
                    use_history.append(u)
                elif energy * (1 - discharge_rate) + p * battery_eff < battery_capacity:
                    energy = energy * (1 - discharge_rate) + p * battery_eff
                    unmet_history.append(u - p)
                    use_history.append(0)
                    waste_history.append(0)
                else:
                    unmet_history.append(u - p)
                    use_history.append(0)
                    waste_history.append(p)
                    energy = energy

            energy_history.append(energy)
            SOC.append(energy/battery_capacity)


            self.SOC = SOC
            self.energy_history = energy_history
            self.unmet_history = unmet_history
            self.waste_history = waste_history
            self.use_history = use_history



    def battery_history(self):
        history = np.vstack((
                    np.array(self.SOC),
                    np.array(self.energy_history),
                    np.array(self.unmet_history),
                    np.array(self.waste_history),
                    np.array(self.use_history)
        ))
        return history

    def lost_power_supply_probability(self):
        LPSP = 1 - self.unmet_history.count(0) / len(self.energy_history)
        return LPSP



class Battery_managed():

    def __init__(self, capacity, config={}):
        self.capacity = capacity
        self.config = config
        self.set_parameters()
        self.init_history()
        self.init_simulation()


    def set_parameters(self):
        try :
            self.depth_of_discharge = self.config['simulation']['battery']['DOD']
            self.discharge_rate = self.config['simulation']['battery']['sigma']
            self.battery_eff = self.config['simulation']['battery']['eta_in']
            self.discharge_eff = self.config['simulation']['battery']['eta_out']
            self.init_charge =  self.config['simulation']['battery']['B0']
            self.DOD = self.depth_of_discharge


        except KeyError:
            print('Parameter is not found in config file, default values are used.')
            self.depth_of_discharge = 1
            self.discharge_rate = 0.005
            self.battery_eff = 0.9
            self.discharge_eff = 0.8
            self.init_charge = 1
            self.DOD = self.depth_of_discharge

    def init_simulation(self):
        self.energy = self.init_charge * self.capacity

    def init_history(self):
        self.supply_history = []
        self.waste_history = []
        self.unmet_history = []
        self.battery_energy_history = []
        self.SOC = []


    def step(self, demand, power):
        if demand >= power:
            self.supply_history.append(power)
            self.unmet_history.append(0)
            energy_new = self.energy * (1 - self.discharge_rate) + (demand - power) * self.battery_eff
            if energy_new < self.capacity:
                self.energy = energy_new  # battery energy got update
                self.waste_history.append(0)
            else:
                self.waste_history.append(demand - power)
                self.energy = self.energy

        elif demand < power:
            self.energy_new = self.energy * (1 - self.discharge_rate) + (demand - power) / self.discharge_eff
            if self.energy_new > (1 - self.DOD) * self.capacity:
                self.energy = self.energy_new
                self.unmet_history.append(0)
                self.waste_history.append(0)
                self.supply_history.append(power)
            elif self.energy * (1 - self.discharge_rate) + demand * self.battery_eff < self.capacity:
                self.energy = self.energy * (1 - self.discharge_rate) + demand * self.battery_eff
                self.unmet_history.append(power - demand)
                self.supply_history.append(0)
                self.waste_history.append(0)
            else:
                self.unmet_history.append(power - demand)
                self.supply_history.append(0)
                self.waste_history.append(demand)
                self.energy = self.energy

        self.battery_energy_history.append(self.energy)
        self.SOC.append(self.energy/self.capacity)


    def history(self):
        battery_history = np.vstack((
                    np.array(self.SOC),
                    np.array(self.battery_energy_history),
                    np.array(self.unmet_history),
                    np.array(self.waste_history),
                    np.array(self.supply_history)
        ))
        return battery_history

    def state(self):
        battery_state = {'current_energy': self.energy, 'usable_capacity':self.DOD * self.capacity}
        return battery_state

    def copy(self):
        return Battery_managed(self.capacity, self.config)

    def deepcopy(self, memodict={}):
        pass


class Soc_model_variable_load():

    def __init__(self, battery, power, load):
        self.battery = battery
        self.battery.run(power, load)

    def get_lost_power_supply_probability(self):

        return self.battery.lost_power_supply_probability()

    def get_battery_history(self):

        return self.battery.battery_history()

    def get_quality_performance_index(self):

        pass



def soc_model_variable_load(power, use, battery_capacity, depth_of_discharge=1,
                         discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8):
    """
    Battery state of charge model with fixed load.

    :param power: Pandas TimeSeries of total power from renewable system
    :param use: float unit W fixed load of the power system
    :param battery_capacity: float unit Wh battery capacity
    :param depth_of_discharge: float 0 to 1 maximum allowed discharge depth
    :param discharge_rate: self discharge rate
    :param battery_eff: optional 0 to 1 battery energy store efficiency default 0.9
    :param discharge_eff: battery discharge efficiency 0 to 1 default 0.8
    :return: tuple SOC: state of charge, energy history: E in battery,
    unmet_history: unmet energy history, waste_history: waste energy history
    """
    DOD = depth_of_discharge
    power = power.tolist()
    use = use.tolist()
    use_history = []
    waste_history = []
    unmet_history = []
    energy_history = []
    energy = 0
    for p, u in zip(power, use):
        if p >= u:
            use_history.append(u)
            unmet_history.append(0)
            energy_new = energy * (1 - discharge_rate) + (p - u) * battery_eff
            if energy_new < battery_capacity:
                energy = energy_new  # battery energy got update
                waste_history.append(0)
            else:
                waste_history.append(p - u)
                energy = energy

        elif p < u:
            energy_new = energy * (1 - discharge_rate) + (p - u) / discharge_eff
            if energy_new > (1 - DOD) * battery_capacity:
                energy = energy_new
                unmet_history.append(0)
                waste_history.append(0)
                use_history.append(use)
            elif energy * (1 - discharge_rate) + p * battery_eff < battery_capacity:
                energy = energy * (1 - discharge_rate) + p * battery_eff
                unmet_history.append(u - p)
                use_history.append(0)
                waste_history.append(0)
            else:
                unmet_history.append(u - p)
                use_history.append(0)
                waste_history.append(p)
                energy = energy

        energy_history.append(energy)

    if battery_capacity == 0:
        SOC = np.array(energy_history)
    else:
        SOC = np.array(energy_history) / battery_capacity
    return SOC, energy_history, unmet_history, waste_history, use_history

if __name__ == '__main__':
    b1 = Battery(10)
    b1.run([1,1,1], [1,1,1])
    b1.run( [1, 1, 1], [10, 10, 10])
    print(b1.lost_power_supply_probability())