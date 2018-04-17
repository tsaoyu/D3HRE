import numpy as np
import pandas as pd
import pygmo as pg


from PyResis import propulsion_power

from D3HRE import simulation
from D3HRE.core.battery_models import soc_model_fixed_load

def objective_warpper(As, Aw, B, demand, route, start_time, speed, kwargs):
    lost_power_supply_probability = temporal_optimization(start_time, route, speed, As, Aw, demand, B,
                                            **kwargs)
    return lost_power_supply_probability

def temporal_optimization(start_time, route, speed, solar_area, wind_area, use, battery_capacity, depth_of_discharge=1,
                           discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8,title=0, azim=0, tracking=0,
                          power_coefficient=0.26, cut_in_speed=2, cut_off_speed=15, technology='csi', system_loss=0.10,
                          angles=None, dataFrame=False, trace_back=False, pandas=False):
    """
    Simulation based optimization for

    :param start_time: str or Dataindex start date of journey
    :param route: numpy nd array (n,2) [lat, lon] of way points
    :param speed: float or list if float is given, it is assumed as constant speed operation mode
                otherwise, a list of n with averaged speed should be given
    :param wind_area: float area of wind turbine area
    :param solar_area: float m^2 area of solar panel area
    :param use: float m^2 load demand of the system
    :param battery_capacity: float Wh total battery capacity of the renewable energy system
    :param title: float degrees title angle of PV panel
    :param azim: float degrees azim angle of PV panel
    :param tracking: int 0 1 or 2 0 for no tracking, 1 for one axis, 2 for two axis
    :param power_coefficient: float power coefficient of wind turbine
    :param cut_in_speed: float m/s cut in speed of wind turbine
    :param cut_off_speed: float m/s cut off speed of wind turbine
    :param technology: optional str 'csi'
    :param system_loss: float system lost of the system
    :param angles: optional solar angle
    :param dataFrame: optional return dataframe or not
    :param trace_back: optional in True give all trace back
    :return: float lost power supply probability (LPSP)
    if trace_back option is on then gives LPSP, SOC, energy history, unmet energy history, water history
    """
    # Pack route to immutable object for caching
    route = tuple(route.flatten())
    solar_power_unit, wind_power_unit = simulation.power_unit_area(start_time, route, speed,
                                                        title=title, azim=azim, tracking=tracking, power_coefficient=power_coefficient,
                                                        cut_in_speed=cut_in_speed, cut_off_speed=cut_off_speed,
                                                        technology=technology, system_loss=system_loss, angles=angles, dataFrame=dataFrame
                                                        )
    solar_power = solar_power_unit * solar_area
    wind_power = wind_power_unit * wind_area
    power = solar_power + wind_power
    SOC, energy_history, unmet_history, waste_history, use_history =\
        soc_model_fixed_load(power, use, battery_capacity, depth_of_discharge,
                             discharge_rate, battery_eff, discharge_eff)
    LPSP = 1- unmet_history.count(0)/len(energy_history)
    if trace_back:
        if pandas:
            all_history = np.vstack((
                np.array(power.tolist()),
                np.array(waste_history),
                np.array(energy_history),
                np.array(use_history),
                np.array(unmet_history),
                np.array(solar_power),
                np.array(wind_power)
            ))
            sim_df = pd.DataFrame(all_history.T, index=power.index,
                            columns=['Power','Waste', 'Battery', 'Use', 'Unmet','Solar_power','Wind_power'])
            return sim_df, LPSP
        else:
            return  LPSP, SOC, energy_history, unmet_history, waste_history, use_history, power
    else:
        return  LPSP


def get_result_df(start_time, route, speed, solar_area, wind_area, use, battery_capacity, depth_of_discharge=1,
                           discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8,title=0, azim=0, tracking=0,
                          power_coefficient=0.26, cut_in_speed=2, cut_off_speed=15, technology='csi', system_loss=0.10,
                          angles=None, dataFrame=False):
    """
    Simulation based optimization for

    :param start_time: str or Dataindex start date of journey
    :param route: numpy nd array (n,2) [lat, lon] of way points
    :param speed: float or list if float is given, it is assumed as constant speed operation mode
                otherwise, a list of n with averaged speed should be given
    :param wind_area: float area of wind turbine area
    :param solar_area: float m^2 area of solar panel area
    :param use: float m^2 load demand of the system
    :param battery_capacity: float Wh total battery capacity of the renewable energy system
    :param title: float degrees title angle of PV panel
    :param azim: float degrees azim angle of PV panel
    :param tracking: int 0 1 or 2 0 for no tracking, 1 for one axis, 2 for two axis
    :param power_coefficient: float power coefficient of wind turbine
    :param cut_in_speed: float m/s cut in speed of wind turbine
    :param cut_off_speed: float m/s cut off speed of wind turbine
    :param technology: optional str 'csi'
    :param system_loss: float system lost of the system
    :param angles: optional solar angle
    :param dataFrame: optional return dataframe or not
    :param trace_back: optional in True give all trace back
    :return: float lost power supply probability (LPSP)
    if trace_back option is on then gives LPSP, SOC, energy history, unmet energy history, water history
    """
    # Pack route to immutable object for caching
    route = tuple(route.flatten())
    solar_power_unit, wind_power_unit = simulation.power_unit_area(start_time, route, speed,
                                                        title=title, azim=azim, tracking=tracking, power_coefficient=power_coefficient,
                                                        cut_in_speed=cut_in_speed, cut_off_speed=cut_off_speed,
                                                        technology=technology, system_loss=system_loss, angles=angles, dataFrame=dataFrame
                                                        )
    solar_power = solar_power_unit * solar_area
    wind_power = wind_power_unit * wind_area
    power = solar_power + wind_power
    SOC, energy_history, unmet_history, waste_history, use_history = \
        soc_model_fixed_load(power, use, battery_capacity, depth_of_discharge,
                             discharge_rate, battery_eff, discharge_eff)

    all_history = np.vstack((
                np.array(power.tolist()),
                np.array(waste_history),
                np.array(energy_history),
                np.array(use_history),
                np.array(unmet_history),
                np.array(solar_power),
                np.array(wind_power)
            ))
    sim_df = pd.DataFrame(all_history.T, index=power.index,
                            columns=['Power','Waste', 'Battery', 'Use', 'Unmet','Solar_power','Wind_power'])
    return sim_df




class Single_mixed_objective_optimization_function:
    def __init__(self, route, start_time, speed, demand,
                 ship, weight=[210, 320, 1, 10000], **kwargs):
        self.route = route
        self.start_time = start_time
        self.speed = speed
        self.demand = demand
        self.weight = weight
        self.ship = ship
        self.parameters = kwargs
    def dimension(self):
        deck_area = self.ship.maximum_deck_area()
        max_wind_area = self.ship.beam ** 2 * np.pi / 4
        max_battery_size = self.ship.displacement * 0.1 * 1000 * 500
        return [deck_area, max_wind_area, max_battery_size]
    def fitness(self, x):
        weight = self.weight
        obj = x[0]*weight[0] + x[1]*weight[1] + x[2]*weight[2]  + \
              weight[3] * objective_warpper(x[0], x[1], x[2],
                            self.demand, self.route, self.start_time, self.speed, self.parameters)
        return [obj]
    def get_bounds(self):
        return [0, 0, 0], self.dimension()
    def get_nobj(self):
        return 1


class Simulation_based_optimization():
    def __init__(self, route, start_time, speed, demand, ship=None):
        self.route = route
        self.start_time = start_time
        self.speed = speed
        self.demand = demand
        self.ship = ship
        self.champion = 0
        self.df = pd.DataFrame()


    def run(self, pop_size=100, gen=100, **kwargs):
        """
        Run optimization with parameters.
        A range of options can be pass into optimization
        depth_of_discharge=1, discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8,title=0, azim=0, tracking=0,
        power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15, technology='csi', system_loss=0.10

        :param pop_size: Population size for the optimization
        :param gen: Generations to be run
        :param kwargs:
        :return:
        """
        prob = pg.problem(Single_mixed_objective_optimization_function(
            self.route, self.start_time, self.speed, self.demand, self.ship, **kwargs))
        algo = pg.algorithm(pg.pso(gen=gen))
        pop = pg.population(prob, pop_size)
        pop = algo.evolve(pop)
        self.champion = pop.champion_x
        return pop.champion_f, pop.champion_x

    def convergence(self, pop_size=100, gen=100, **kwargs):
        """
        Run optimization with parameters.
        A range of options can be pass into optimization
        depth_of_discharge=1, discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8,title=0, azim=0, tracking=0,
        power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15, technology='csi', system_loss=0.10

        :param pop_size: Population size for the optimization
        :param gen: Generations to be run
        :param kwargs:
        :return:
        """

        prob = pg.problem(Single_mixed_objective_optimization_function(
            self.route, self.start_time, self.speed, self.demand, self.ship, **kwargs))
        uda = pg.pso(gen=gen)
        algo = pg.algorithm(uda)
        algo.set_verbosity(1)
        pop = pg.population(prob, pop_size)
        pop = algo.evolve(pop)
        log = algo.extract(type(uda)).get_log()
        return log, pop

    def resource_df(self, **kwargs):
        """
        Preview of the optimisation. Return

        :return:
        """
        route = tuple(self.route.flatten())
        return  simulation.power_unit_area(self.start_time, route, self.speed, **kwargs)

    def power_df(self):
        """
        Show result from pandas time series.

        :return:
        """
        area_solar, area_wind, battery_capacity = self.champion
        return get_result_df(self.start_time,self.route, self.speed,area_solar,area_wind,self.demand,battery_capacity)

    def combined_df(self):
        """
        Get a combined dataframe including power and resource.

        :return:
        """
        power = self.power_df(self)
        resource = self.resource_df(self)
        pass


if __name__ == '__main__':
    ship1 =  propulsion_power.Ship()
    ship1.dimension(5.72, 0.248, 0.76, 1.2, 5.72/(0.549)**(1/3),0.613)
    soo = Single_mixed_objective_optimization_function(1,1,1,1,ship1)
    route = np.array(
      [[  20.93866679,  168.56555458],
       [  18.45091531,  166.77237183],
       [  16.01733564,  165.45107928],
       [  13.92043435,  165.2623232 ],
       [  12.17361734,  165.63983536],
       [  10.50804555,  166.96112791],
       [   9.67178793,  168.94306674],
       [   9.20628817,  171.58565184],
       [   9.48566359,  174.60574911],
       [   9.95078073,  176.68206597],
       [  10.69358   ,  178.94713892],
       [  11.06430687, -176.90022735],
       [  10.87900106, -172.27570342]])
    sbo = Simulation_based_optimization(route, '2014-01-01',2, 40, ship=ship1 )
    print(sbo.run(discharge_rate=0.01, battery_eff=0.9, power_coefficient=0.28, azim=30))
    print(soo.get_bounds())