import numpy as np
import pandas as pd

from functools import lru_cache

from gsee.gsee import brl_model, pv
from D3HRE.core.dataframe_utility import full_day_cut

from D3HRE.core.battery_models import min_max_model, Battery
from D3HRE.core.weather_data_download import resource_df_download_and_process
from D3HRE.core.wind_turbine_model import power_from_turbine, resistance_power
from D3HRE.core.mission_utility import Mission
from D3HRE import MaritimeRobot


class Task:
    """
    Task is a high level object consist of mission object, robot object and
    optional power consumption list.
    """
    def __init__(self, mission, robot, power_consumption_list={}):
        """

        :param mission: object should have df attribute that contains information on
        :param vehicle: object should have prop_power() method that return
        propulsion power of the vehicle
        :param power_consumption_list: dictionary consist of components with power
        requirements and duty cycle
        """
        self.mission = mission
        self.robot = robot
        if power_consumption_list != {}:
            self.power_consumption_list = power_consumption_list
            self.estimate_demand_load_compatible()
        else:
            self.estimate_demand_load()

    def estimate_demand_load_compatible(self):
        new_robot = MaritimeRobot(self.power_consumption_list)
        new_robot.set_from_PyResis(self.robot)
        self.robot = new_robot
        self.estimate_demand_load()

    def estimate_demand_load(self):
        self.robot.estimate_demand_load(self.mission)
        self.hotel_load = self.robot.hotel_load
        self.prop_load = self.robot.prop_load
        self.critical_hotel_load = self.robot.critical_hotel_load
        self.critical_prop_load = self.robot.critical_prop_load

        self.load_demand = self.hotel_load + self.prop_load


class Power_simulation():
    def __init__(self, Mission, config={}):
        self.mission = Mission
        self.config = config
        self.resource_df = resource_df_download_and_process(self.mission)
        self.set_parameter()

    def set_parameter(self):
        try:
            self.power_coefficient = self.config['transducer']['wind']['power_coef']
            self.cut_in_speed = self.config['transducer']['wind']['v_in']
            self.rated_speed = self.config['transducer']['wind']['v_rate']

            self.tilt = self.config['transducer']['solar']['tilt']
            self.azim = self.config['transducer']['solar']['azim']
            self.tracking = self.config['transducer']['solar']['azim']
            self.capacity = self.config['transducer']['solar']['power_density']

        except KeyError:

            self.power_coefficient = 0.3
            self.cut_in_speed = 2
            self.rated_speed = 15

            self.tilt = 0
            self.azim = 180
            self.tracking = 0
            self.capacity = 140

    @property
    @lru_cache(maxsize=32)
    def wind_power_simulation(self):
        wind_df = pd.DataFrame()
        print("Start wind energy power simulation...")
        wind_df['wind_raw'] = self.resource_df.V2.apply(
            lambda x: power_from_turbine(
                x, 1, self.power_coefficient, self.cut_in_speed, self.rated_speed
            )
        )
        wind_df['wind_correction'] = resistance_power(self.resource_df, 1)
        wind_df['wind_power'] = wind_df['wind_raw'] - wind_df['wind_correction']
        self.wind = wind_df
        return wind_df['wind_power']

    @property
    @lru_cache(maxsize=32)
    def solar_power_simulation(self):
        self.solar = pd.DataFrame()
        print("Start solar energy power simulation...")
        self.resource_df['global_horizontal'] = self.resource_df.SWGDN
        self.resource_df['diffuse_fraction'] = brl_model.location_run(self.resource_df)
        self.resource_df['solar_power'] = pv.run_plant_model_location(
            self.resource_df,
            self.tilt,
            self.azim,
            self.tracking,
            self.capacity,
            config=self.config
        )
        self.solar['solar_power'] = self.resource_df['solar_power']
        return self.solar['solar_power']

    def run(self, wind_power_area, solar_power_area):
        wind_power_df = self.wind_power_simulation * wind_power_area
        solar_power_df = self.solar_power_simulation * solar_power_area
        self.simulation_result = pd.concat([wind_power_df, solar_power_df], axis=1)
        return self.simulation_result

class Reactive_simulation():
    def __init__(self, Task, config={}):
        self.Task = Task
        self.resource_df = resource_df_download_and_process(self.Task.mission)
        self.df = self.Task.mission.df
        self.config = config
        self.set_parameter()

    def set_parameter(self):
        try:
            self.power_coefficient = self.config['transducer']['wind']['power_coef']
            self.cut_in_speed = self.config['transducer']['wind']['v_in']
            self.rated_speed = self.config['transducer']['wind']['v_rate']

            self.tilt = self.config['transducer']['solar']['tilt']
            self.azim = self.config['transducer']['solar']['azim']
            self.tracking = self.config['transducer']['solar']['tracking']
            self.capacity = self.config['transducer']['solar']['power_density']

        except KeyError:

            self.power_coefficient = 0.3
            self.cut_in_speed = 2
            self.rated_speed = 15

            self.tilt = 0
            self.azim = 180
            self.tracking = 0
            self.capacity = 140

    @property
    @lru_cache(maxsize=32)
    def wind_power_simulation(self):
        wind_df = pd.DataFrame()
        print("Start wind energy power simulation...")
        wind_df['wind_raw'] = self.resource_df.V2.apply(
            lambda x: power_from_turbine(
                x, 1, self.power_coefficient, self.cut_in_speed, self.rated_speed
            )
        )
        wind_df['wind_correction'] = resistance_power(self.resource_df, 1)
        wind_df['wind_power'] = wind_df['wind_raw'] - wind_df['wind_correction']
        self.wind = wind_df
        return wind_df['wind_power']

    @property
    @lru_cache(maxsize=32)
    def solar_power_simulation(self):
        self.solar = pd.DataFrame()
        print("Start solar energy power simulation...")
        self.resource_df['global_horizontal'] = self.resource_df.SWGDN
        self.resource_df['diffuse_fraction'] = brl_model.location_run(self.resource_df)
        self.resource_df['solar_power'] = pv.run_plant_model_location(
            self.resource_df,
            self.tilt,
            self.azim,
            self.tracking,
            self.capacity,
            config = self.config
        )
        self.solar['solar_power'] = self.resource_df['solar_power']
        return self.solar['solar_power']

    def run(self, solar_area, wind_area, battery_capacity):
        power_supply = (
            self.wind_power_simulation * wind_area
            + self.solar_power_simulation * solar_area
        )

        if self.config != {}:
            battery = Battery(battery_capacity, config=self.config)
            safe_factor = self.config['optimization']['safe_factor']
        else:
            battery = Battery(battery_capacity)
            safe_factor = 0

        supply, load = power_supply.tolist(), (self.Task.load_demand*(1+safe_factor)).tolist()
        battery.run(supply, load)
        lpsp = battery.lost_power_supply_probability()
        return lpsp

    def result(self, solar_area, wind_area, battery_capacity):
        print('====== Post simulation run ======')
        power_supply = (
            self.wind_power_simulation * wind_area
            + self.solar_power_simulation * solar_area
        )
        supply, load = power_supply.tolist(), self.Task.load_demand.tolist()
        if self.config != {}:
            battery = Battery(battery_capacity, config=self.config)
        else:
            battery = Battery(battery_capacity)

        battery.run(supply, load)

        prop_load = (self.Task.load_demand - self.Task.hotel_load).values
        load_demand = self.Task.load_demand.values
        hotel_load = self.Task.hotel_load.values
        critical_load = prop_load + self.Task.critical_hotel_load.values

        load_demand_history = np.vstack((load_demand, prop_load, hotel_load, critical_load))
        load_demand_history_df = pd.DataFrame(
            data=load_demand_history.T,
            index=self.Task.mission.df.index,
            columns=['Load_demand', 'Prop_load', 'Hotel_load', 'Critical_load'],
        )
        load_demand_history_df = full_day_cut(load_demand_history_df)

        battery_history = battery.battery_history()
        battery_history_df = pd.DataFrame(
            data=battery_history.T,
            index=self.df.index,
            columns=['SOC', 'Battery', 'Unmet', 'Waste', 'Supply'],
        )

        results = [
            battery_history_df,
            load_demand_history_df,
            self.solar * solar_area,
            self.wind * wind_area,
        ]
        result_df = pd.concat(results, axis=1)
        return result_df

    def post_run(self, solar_area, wind_area, battery_capacity, dispatch):
        print('====== Post simulation run ======')
        power_supply = (
            self.wind_power_simulation * wind_area
            + self.solar_power_simulation * solar_area
        )

        post_run_len = len(dispatch)  # match length of simulation
        supply, load = power_supply[:post_run_len].tolist(), dispatch['Power'].tolist()

        if self.config != {}:
            battery = Battery(battery_capacity, config=self.config)
        else:
            battery = Battery(battery_capacity)

        battery.run(supply, load)
        prop_load = (self.Task.load_demand - self.Task.hotel_load).values
        load_demand = self.Task.load_demand.values
        hotel_load = self.Task.hotel_load.values
        critical_load = prop_load + self.Task.critical_hotel_load.values

        load_demand_history = np.vstack((load_demand, prop_load, hotel_load, critical_load))
        load_demand_history_df = pd.DataFrame(
            data=load_demand_history.T,
            index=self.Task.mission.df.index,
            columns=['Load_demand', 'Prop_load', 'Hotel_load', 'Critical_load'],
        )

        load_demand_history_df = full_day_cut(load_demand_history_df)

        battery_history = battery.battery_history()
        battery_history_df = pd.DataFrame(
            data=battery_history[:post_run_len].T,
            index=self.df[:post_run_len].index,
            columns=['SOC', 'Battery', 'Unmet', 'Waste', 'Supply'],
        )

        results = [
            battery_history_df,
            load_demand_history_df[:post_run_len],
            self.solar[:post_run_len] * solar_area,
            self.wind[:post_run_len] * wind_area,
        ]
        result_df = pd.concat(results, axis=1)

        return result_df


if __name__ == '__main__':
    pass
