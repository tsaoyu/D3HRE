import numpy as np
import pygmo as pg
import cloudpickle

from D3HRE import simulation
from D3HRE.core.battery_models import Battery_managed


class Mixed_objective_optimization_function:
    def __init__(self, Task, config={}):
        self.Task = Task
        self.config = config
        self.set_parameters()
        self.set_constraint()

        if config != {}:
            self.sim = simulation.PowerSim(Task, config=config)
        else:
            self.sim = simulation.PowerSim(Task)

    def set_parameters(self):
        try:
            cost = self.config['optimization']['cost']
            self.weight = [cost['solar'], cost['wind'], cost['battery'], cost['lpsp']]

        except KeyError:
            self.weight = [210, 320, 1, 10000]


    def set_constraint(self):
        try:
            SED = self.config['simulation']['battery']['SED']
            volume_factor = self.config['optimization']['constraints']['volume_factor']
            water_plane_coff = self.config['optimization']['constraints']['water_plane_coff']
            turbine_diameter_ratio = self.config['optimization']['constraints']['turbine_diameter_ratio']
            max_solar_area = self.Task.robot.surface_area * water_plane_coff
            max_wind_area = (turbine_diameter_ratio * self.Task.robot.beam) ** 2 * np.pi / 4
            max_battery_capacity = self.Task.robot.displacement * volume_factor * 1000 * SED

        except KeyError:

            max_solar_area = self.config['optimization']['constraints']['solar_area']
            max_wind_area = self.config['optimization']['constraints']['wind_area']
            max_battery_capacity = self.config['optimization']['constraints']['battery_capacity']

        self.max_capacity = [max_solar_area, max_wind_area, max_battery_capacity]

    def constraints(self):

        return self.max_capacity

    def fitness(self, x):
        weight = self.weight
        obj = (
            x[0] * weight[0]
            + x[1] * weight[1]
            + x[2] * weight[2]
            + weight[3] * self.sim.run(x[0], x[1], x[2])
        )
        return [obj]

    def get_bounds(self):
        return [0, 0, 0.001], self.constraints()

    def get_nobj(self):
        return 1

class Multiple_objective_optimization_function(Mixed_objective_optimization_function):

    def fitness(self, x):
        weight = self.weight
        capital_cost = x[0] * weight[0] + x[1] * weight[1] + x[2] * weight[2]
        lpsp = self.sim.run(x[0], x[1], x[2])
        return [capital_cost, lpsp]

    def get_nobj(self):
        return 2

class Constraint_mixed_objective_optimisation(Mixed_objective_optimization_function):
    def __init__(self, Task, config={}):
        """
        Basic class for the constraint mixed objective optimisation problem. The optimisation
        start by providing the Task object to the problem. It will formed as an optimisation
        problem using the mixed objective optimisation function. When all the parameters are set,
        call run() method to run the optimisation.

        :param Task: task object (mission + robot )
        :param config: configuration file if exist will be pass to mixed objective function
        """
        self.config = config
        self.Task = Task
        self.set_parameters()
        if config != {}:
            self.problem = pg.problem(
                Mixed_objective_optimization_function(Task, self.config)
            )
            self.sim = simulation.PowerSim(Task, config=self.config)
        else:
            self.problem = pg.problem(Mixed_objective_optimization_function(Task))
            self.sim = simulation.PowerSim(Task)

    def set_parameters(self):
        try:
            self.generation = self.config['optimization']['method']['pso']['generation']
            self.pop_size = self.config['optimization']['method']['pso']['population']
        except KeyError:
            self.generation = 100
            self.pop_size = 100

    def run(self, converge_info=False, pop_info=False):
        """
        Run the optimisation process using PSO algorithm.
        :param converge_info: optional run the optimisation with convergence information
        :param converge_info: optional run the optimisation with population information
        :return:
        """
        print("Start the optimisation process...")

        if pop_info != False:
            uda = pg.pso(gen=1)
            algo = pg.algorithm(uda)
            algo.set_verbosity(1)
            pop = pg.population(self.problem, self.pop_size)
            self.pop_history = [pop]
            for i in range(int(pop_info)):
                pop = algo.evolve(pop)
                self.pop_history.append(pop)
            self.log = algo.extract(type(uda)).get_log()
            self.pop = pop
        elif converge_info == True:
            uda = pg.pso(gen=self.generation)
            algo = pg.algorithm(uda)
            algo.set_verbosity(1)
            pop = pg.population(self.problem, self.pop_size)
            self.log = algo.extract(type(uda)).get_log()
            self.pop = pop
        else:
            uda = pg.pso(gen=self.generation)
            algo = pg.algorithm(uda)
            pop = pg.population(self.problem, self.pop_size)
            pop = algo.evolve(pop)
        self.champion = pop.champion_x
        return pop.champion_f, pop.champion_x

    def island_run(self):
        uda = pg.pso(gen=self.generation)
        algo = pg.algorithm(uda)
        pop = pg.population(self.problem, self.pop_size)
        island = pg.island(algo=algo, pop=pop, udi=pg.mp_island())
        island.evolve()
        island.wait()
        pop = island.get_population()
        return pop.champion_f, pop.champion_x

    def get_lpsp(self):
        solar_area_opt, wind_area_opt, battery_capacity = self.champion
        return self.sim.run(solar_area_opt, wind_area_opt, battery_capacity)

    def get_report(self):
        """
        Simulate the power system with the optimised configuration.
        :return: DataFrame on the
        """
        solar_area_opt, wind_area_opt, battery_capacity = self.champion
        return self.sim.get_report(solar_area_opt, wind_area_opt, battery_capacity)

    def get_resource_df(self):
        return self.sim.resource_df

    def save_result(self, name='optimisation_result.pkl'):
        """
        Save the optimisation result.

        :param name: save the optimisation result to pickle file
        :return:
        """
        solar_area, wind_area, battery_capacity = self.champion
        system = Battery_managed(battery_capacity, config=self.config)
        result_df = self.get_report()

        system.configuration = self.champion
        resource = result_df.wind_power + result_df.solar_power
        with open(name, 'wb') as f:
            cloudpickle.dump([system, result_df, resource], f)

class Constraint_multiple_objective_optimisation(Multiple_objective_optimization_function):
    def __init__(self, Task, config={}, algorithm='nsga-2'):
        """
        Basic class for the constraint mixed objective optimisation problem. The optimisation
        start by providing the Task object to the problem. It will formed as an optimisation
        problem using the mixed objective optimisation function. When all the parameters are set,
        call run() method to run the optimisation.

        :param Task: task object (mission + vehicle )
        :param config: configuration file if exist will be pass to mixed objective function
        """
        self.config = config
        self.Task = Task
        self.set_parameters()
        self.algorithm_type = algorithm
        if config != {}:
            self.problem = pg.problem(
                Multiple_objective_optimization_function(Task, self.config)
            )
            self.sim = simulation.PowerSim(Task, config=self.config)
        else:
            self.problem = pg.problem(Multiple_objective_optimization_function(Task))
            self.sim = simulation.PowerSim(Task)


    def set_parameters(self):
        try:
            self.generation = self.config['optimization']['method']['pso']['generation']
            self.pop_size = self.config['optimization']['method']['pso']['population']
        except KeyError:
            self.generation = 100
            self.pop_size = 100

    def run(self):
        """
        Run the optimisation process using PSO algorithm.
        :param converge_info: optional run the optimisation with convergence information
        :param converge_info: optional run the optimisation with population information
        :return:
        """
        print("Start the optimisation process...")

        if self.algorithm_type == 'nsga-2':
            uda = pg.nsga2(gen=self.generation)
        elif self.algorithm_type == 'moea-d':
            uda = pg.moead(gen=self.generation)
        elif self.algorithm_type == 'ihs':
            uda = pg.ihs(gen=self.generation)

        algo = pg.algorithm(uda)
        pop = pg.population(self.problem, self.pop_size)
        pop = algo.evolve(pop)
        self.pop = pop

    def plot_non_dominated_fronts(self):
        return pg.plot_non_dominated_fronts(self.pop.get_f())



if __name__ == '__main__':
    pass
