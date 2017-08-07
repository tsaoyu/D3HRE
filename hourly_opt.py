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
      [[  46.76694005,  154.22009259],
       [  45.39225571,  154.69198279],
       [  44.72560022,  154.9751169 ],
       [  43.16280372,  155.54138514],
       [  42.47052182,  155.82451926],
       [  41.13387173,  156.57954358],
       [  39.62418619,  157.71208005],
       [  38.45133193,  158.84461652],
       [  37.5589637 ,  159.59964084],
       [  37.03341446,  160.73217731],
       [  36.27628169,  161.48720163],
       [  35.12669961,  162.99725026],
       [  34.19517221,  164.12978673],
       [  33.25323944,  165.45107928],
       [  31.98138877,  166.48923772],
       [  31.01574791,  167.24426203],
       [  29.87668645,  167.90490831],
       [  28.64166925,  168.56555458],
       [  27.47570471,  169.03744478],
       [  25.19216599,  169.50933498],
       [  23.2124053 ,  169.50933498],
       [  20.93866679,  168.56555458],
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
       [  10.87900106, -172.27570342],
       [   9.95078073, -168.97247204],
       [   9.67178793, -166.89615517],
       [   8.92669178, -164.53670418],
       [   8.36686209, -163.12103359],
       [   7.61917834, -161.61098496],
       [   7.05755065, -160.66720457],
       [   6.30766145, -159.15715593],
       [   5.93230149, -158.1189975 ],
       [  -1.60710319, -156.04268063],
       [   2.26159813, -153.68322965],
       [   3.86368074, -152.45631514],
       [   4.89879125, -149.53059591],
       [   6.12001443, -147.45427905],
       [   6.77647758, -143.77353551],
       [   7.80622675, -137.92209707],
       [   8.4602244 , -134.05259745],
       [  10.4152364 , -131.7875245 ],
       [  12.45023942, -130.84374411],
       [  14.65210384, -130.27747587],
       [  16.7417115 , -129.80558568],
       [  19.87723492, -128.29553704],
       [  22.25491519, -125.18106174],
       [  22.77801408, -124.61479351],
       [  23.55890563, -123.10474488],
       [  24.07694426, -121.50031821],
       [  24.33518399, -119.3296233 ],
       [  24.33518399, -116.97017231],
       [  23.29911526, -114.89385545],
       [  22.51671211, -112.72316054],
       [  21.90509054, -111.40186799],
       [  20.85049523, -109.32555112],
       [  20.05464587, -108.09863661],
       [  19.34381503, -106.21107582]])
    sbo = Simulation_based_optimization(route, '2014-01-01',2, 40, ship=ship1 )
    print(sbo.run())
    print(soo.get_bounds())