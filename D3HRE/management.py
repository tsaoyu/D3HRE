import visilibity as vis
import logging


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def construct_environment_demo(power_dataframe, battery_capacity):
    aggregated_power = power_dataframe.cumsum().Power
    aggregated_power_lower = []
    aggregated_power_higher = []
    i = 0
    for power in aggregated_power.tolist():
        aggregated_power_lower.append([i, power])
        aggregated_power_higher.append([i, power + battery_capacity])
        i += 1

    # Construct the upper hole in a reverse order
    aggregated_power_higher.reverse()
    aggregated_power_higher.insert(0, aggregated_power_higher[-1])
    aggregated_power_higher.insert(1, [aggregated_power_higher[0][0], aggregated_power_higher[1][1]])
    del aggregated_power_higher[-1]

    # Construct the lower hole in the reverse order
    aggregated_power_lower.append([aggregated_power_lower[-1][0], aggregated_power_lower[0][1]])
    # close the hole
    # aggregated_power_lower.append(aggregated_power_lower[0])

    # Construct the wall list
    wall_list = []
    wall_list.append([aggregated_power_lower[0][0], 0])
    wall_list.append([aggregated_power_lower[-1][0], 0])
    wall_list.append(aggregated_power_higher[2])
    wall_list.append(aggregated_power_higher[1])

    higher_hole_points = [vis.Point(x, y) for x, y in aggregated_power_higher]
    lower_hole_points = [vis.Point(x, y) for x, y in aggregated_power_lower]
    wall_points = [vis.Point(x, y) for x, y in wall_list]

    higher_hole = vis.Polygon(higher_hole_points)

    lower_hole = vis.Polygon(lower_hole_points)

    wall = vis.Polygon(wall_points)
    return wall, higher_hole, lower_hole


class Management_base():

    def __init__(self):
        self.type = 'base'

    def manage(self):
        pass

    def update(self):
        pass

class Absolute_follow_management():
    """
    Absolute follow management as the name indicates, it use all the power that
    is available that the moment. The demand according to this management strategy
    is an absolute follow of resources.
    """
    def __init__(self):
        self.type = 'reactive'

    def manage(self):
        """
        :return: a list of demand which is exact the same as resources
        """
        return self.resources

    def update(self, observation, resources):
        """
        Update internal variables.
        :param observation: provided but will not be used
        :param resources: list in W energy that supplied from the HRES
        """
        self.resources = resources
        pass

class Reactive_follow_management():

    def __init__(self, demand):
        if isinstance(demand, list):
            self.demand = demand
        elif isinstance(demand, pd.Series):
            self.demand = demand.tolist()
        else:
            print('Sorry, I do not accept this kind of demand.')
        self.type = 'reactive'
        self.resources_history = []
        self.demand = demand
        self.time_step = 0

    def manage(self):

        if self.demand[self.time_step] <= self.resources[self.time_step]:
            supply = self.demand[self.time_step]
        elif self.demand[self.time_step] > self.resources[self.time_step]:
            difference = self.demand[self.time_step] - self.resources[self.time_step]
            if self.observation.current_energy - difference > self.observation.usable_capacity:
                supply = self.demand[self.time_step]
            else:
                supply = 0

        return supply

    def update(self, observation, resources):
        """

        :param observation: battery
        :param resources:
        :return:
        """
        self.observation = observation
        self.resources = resources
        self.resources_history.append(resources)
        self.time_step += 1


class Dynamic_environment():

    def __init__(self, battery, resource, management):
        self.battery = battery
        self.resource = resource
        self.resource_list = self.resource.tolist()
        self.management = management

    def observation(self):
        return self.battery.state()

    def step(self, supply, power):
        self.battery.step(supply, power)

    def step_over_time(self):
        if self.management.type == 'predictive':
            frequency = self.management.frequency
            intervals = len(self.resource) // frequency
            remaining = len(self.resource) % frequency

            for i in intervals:
                power_in_period = self.resource[i*frequency: (i+1)*frequency]
                supply = self.management.udpate(self.observation())
                for power in power_in_period:
                    self.step(supply, power)

            power_in_period = self.resource[-remaining:]
            self.management.update(self.observation())
            supply = self.management.manage()
            for power in power_in_period:
                self.step(supply, power)


        elif self.management.type == 'global':
            self.management.update(self.battery, self.resource_list)
            supply = self.management.manage()
            for power in self.resource:
                self.step(supply, power)


        elif self.management.type == 'reactive':
            self.management.update(self.observation(), self.resource_list)
            supply = self.management.manage()
            for power in self.resource:
                self.step(supply, power)
        else:
            print('I don\'t know how to handle this type of management!')

    def simulation_result(self):
        battery_history = self.battery.history()
        history = pd.DataFrame(columns=['SOC', 'Battery', 'Unmet', 'Waste', 'Supply'],
                               index= self.resource.index,
                               data=battery_history.T)
        return history




class Finite_optimal_management():

    def __init__(self, power_series, battery_capacity, strategy='full-empty', epsilon=0.00001, config={}):
        self.power_series = power_series
        self.config = config
        self.time = len(self.power_series)
        self.epsilon = epsilon
        self.strategy = strategy
        self.scale = 0.65
        self.set_parameters()
        self.battery_capacity = battery_capacity
        logging.basicConfig(filename='management.log', level=logging.DEBUG)


    def set_parameters(self):
        try:
            self.DOD = self.config['simulation']['battery']['DOD']
            logging.info('Use value from config file DOD: {}'.format(self.DOD))
        except KeyError:
            self.DOD = 0.5
            logging.info('Default value of DOD: {} is used'.format(self.DOD))

    @property
    def aggregated_power(self):
        return self.power_series.cumsum()

    def get_boundary(self):
        aggregated_power_lower = []
        aggregated_power_higher = []
        aggregated_power_higher_dbg = []
        i = 0
        for power in self.aggregated_power.tolist():
            aggregated_power_lower.append([i, power * self.scale])
            aggregated_power_higher.append([i, power * self.scale + self.battery_capacity * (1 - self.DOD)])
            aggregated_power_higher_dbg.append([i, power + self.battery_capacity])
            #TODO this is hard coded energy bumper
            i += 1

        self.aggregated_power_higher_dbg = aggregated_power_higher_dbg

        # Construct the upper hole in a reverse order
        aggregated_power_higher.reverse()
        aggregated_power_higher.insert(0, aggregated_power_higher[-1])
        aggregated_power_higher.insert(1, [aggregated_power_higher[0][0], aggregated_power_higher[1][1]])
        del aggregated_power_higher[-1]

        # Construct the lower hole in the reverse order
        aggregated_power_lower.append([aggregated_power_lower[-1][0], aggregated_power_lower[0][1]])
        self.aggregated_power_higher = aggregated_power_higher
        self.aggregated_power_lower = aggregated_power_lower
        return aggregated_power_higher, aggregated_power_lower

    def construct_wall(self):
        wall_list = []
        left_x = self.aggregated_power_lower[0][0] - 1  # esstentially time - 1
        right_x = self.aggregated_power_lower[-1][0] + 1  # esstentially time + 1
        bottom_y = 0
        top_y = self.aggregated_power_higher[1][1] + 50
        wall_list.append([left_x, bottom_y])
        wall_list.append([right_x, bottom_y])
        wall_list.append([right_x, top_y])
        wall_list.append([left_x, top_y])
        self.wall_list = wall_list
        return wall_list

    def _convert_to_visilibity_points(self, points_list):
        return [vis.Point(x, y) for x, y in points_list]

    def _convert_to_visilibity_polygon(self, points_list):
        vis_points = [vis.Point(x, y) for x, y in points_list]
        vis_polygon = vis.Polygon(vis_points)
        return vis_polygon

    def construct_env(self):
        self.wall = self._convert_to_visilibity_polygon(self.wall_list)
        self.higher_hole = self._convert_to_visilibity_polygon(self.aggregated_power_higher)
        self.lower_hole = self._convert_to_visilibity_polygon(self.aggregated_power_lower)
        env = vis.Environment([self.wall, self.higher_hole, self.lower_hole])
        self.env = env
        return env

    def check_env(self):
        print('Is the higher hole in standard form?', self.higher_hole.is_in_standard_form())
        print('Is the lower hole in standard form?', self.lower_hole.is_in_standard_form())
        print('Is the wall in standard form?', self.wall.is_in_standard_form())
        print('Is the environment valid?', self.env.is_valid(self.epsilon))

    def plot_env(self):
        wall_list = self.wall_list[:]
        higher_hole = self.aggregated_power_higher[:]
        lower_hole = self.aggregated_power_lower[:]

        wall_list.append(wall_list[0])
        higher_hole.append(higher_hole[0])
        lower_hole.append(lower_hole[0])

        wall_list_x, wall_list_y = np.array(wall_list).T
        higher_hole_x, higher_hole_y = np.array(higher_hole).T
        lower_hole_x, lower_hole_y = np.array(lower_hole).T
        higher_limit_hole_x, higher_limit_hole_y = np.array(self.aggregated_power_higher_dbg).T


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wall_list_x, wall_list_y, 'black')
        ax.plot(higher_limit_hole_x, higher_limit_hole_y)
        ax.plot(higher_hole_x, higher_hole_y)
        ax.plot(lower_hole_x, lower_hole_y)
        return ax

    def find_shortest_path(self):
        base_energy_start = self.aggregated_power_lower[0][1]
        base_energy_end = self.aggregated_power_lower[-2][1]

        if self.strategy == 'full-empty':
            strategy_state = (1, 0)
        elif self.strategy == 'full-full':
            strategy_state = (1, 1)
        elif self.strategy == 'empty-empty':
            strategy_state = (0, 0)
        elif isinstance(self.strategy, tuple):
            strategy_state = self.strategy
        else:
            print('This operation strategy is not supported!')

        self.start_energy = base_energy_start + strategy_state[0] * self.battery_capacity * (1 - self.DOD)
        #TODO this is hard coded
        self.end_energy = base_energy_end + strategy_state[1] * self.battery_capacity

        start = vis.Point(0, self.start_energy)
        end = vis.Point(self.time - 1, self.end_energy)
        start.snap_to_boundary_of(self.env, self.epsilon)
        start.snap_to_vertices_of(self.env, self.epsilon)
        vis_poly = vis.Visibility_Polygon(start, self.env, self.epsilon)
        shortest_path = self.env.shortest_path(start, end, self.epsilon)
        return shortest_path

    def find_optimal_dispatch(self):
        self.get_boundary()
        self.construct_wall()
        self.construct_env()
        vis_path = self.find_shortest_path()
        optimal_dispatch = [[point.x(), point.y()] for point in vis_path.path()]
        self.optimal_dispatch = optimal_dispatch
        return optimal_dispatch

    def plot_result(self):
        ax = self.plot_env()
        ax.plot(0, self.start_energy, 'go')
        ax.plot(self.time - 1, self.end_energy, 'ro')
        optimal_dispatch_x, optimal_dispatch_y = np.array(self.optimal_dispatch).T
        ax.plot(optimal_dispatch_x, optimal_dispatch_y, 'black')



