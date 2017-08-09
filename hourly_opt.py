import numpy as np
import pygmo as pg

import hourly_sim
import monthly_opt
from demand import  propulsion_power

def objective_warpper(As, Aw, B, demand, route, start_time, speed, **kwargs):
    lpsp = hourly_sim.temporal_optimization(start_time, route, speed, As, Aw, demand, B,
                                            power_coefficient=0.26, **kwargs)
    return lpsp


class Single_mixed_objective_optimization_function:
    def __init__(self, route, start_time, speed, demand, ship, weight=None):
        self.route = route
        self.start_time = start_time
        self.speed = speed
        self.demand = demand
        if not weight:
            self.weight = [1, 1, 0.0001, 10000]
            print('Default weight', self.weight, 'is used')
        else:
            self.weight = weight
        self.ship = ship
    def dimension(self):
        deck_area = self.ship.maximum_deck_area()
        max_wind_area = self.ship.beam ** 2 * np.pi / 4
        max_battery_size = self.ship.displacement * 0.1 * 1000 * 500
        return [deck_area, max_wind_area, max_battery_size]
    def fitness(self, x):
        weight = self.weight
        obj = x[0]*weight[0] + x[1]*weight[1] + x[2]*weight[2]  + \
              weight[3] * objective_warpper(x[0], x[1], x[2],
                            self.demand, self.route, self.start_time, self.speed)
        return [obj]
    def get_bounds(self):
        return [0, 0, 0], self.dimension()
    def get_nobj(self):
        return 1

class Route_based_optimization():
    """
    A simple route based optimization that minimize the harvester cost.
    """
    def __init__(self, route, demand):
        """
        Init a route based optimization with only route and demand
        :param route:
        :param demand:
        """
        self.route = route
        self.demand = demand
    def result(self):
        """
        Return optimization result on route based optimization
        :return: Area of solar panel and wind turbine in m^2
        """
        monthly_optimization = monthly_opt.Opt(self.route)
        result = monthly_optimization.route_based_presizing(self.demand)
        return result

class Mission_based_optimization(Route_based_optimization):

    def __init__(self, route, demand, start_time, speed):
        Route_based_optimization.__init__(self, route, demand)
        self.start_time = start_time
        self.speed = speed

    def result(self):
        """
        Return optimization result on route based optimization
        :return: Area of solar panel and wind turbine in m^2
        """
        monthly_optimization = monthly_opt.Opt(self.route)
        monthly_optimization.route_based_presizing(self.demand)
        result = monthly_optimization.mission_based_preszing('2014-01-01', 2)
        return result

class Simulation_based_optimization():
    def __init__(self, route, start_time, speed, demand, ship=None, weight=None):
        self.route = route
        self.start_time = start_time
        self.speed = speed
        self.demand = demand
        self.ship = ship


    def run(self, pop_size=100, gen=100, **kwargs):
        prob = pg.problem(Single_mixed_objective_optimization_function(self.route, self.start_time, self.speed, self.demand, self.ship, **kwargs))
        algo = pg.algorithm(pg.pso(gen=gen))
        pop = pg.population(prob, pop_size)
        pop = algo.evolve(pop)
        return pop.champion_f, pop.champion_x


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
    print(sbo.run())
    print(soo.get_bounds())