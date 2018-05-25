import numpy as np
import pandas as pd

from functools import lru_cache

from gsee.gsee import brl_model, pv
from D3HRE.core.dataframe_utility import full_day_cut
from D3HRE.core.hotel_load_model import HotelLoad
from D3HRE.core.battery_models import min_max_model, Soc_model_variable_load, Battery
from D3HRE.core.weather_data_download import resource_df_download_and_process
from D3HRE.core.wind_turbine_model import power_from_turbine, resistance_power
from D3HRE.core.mission_utility import Mission
from D3HRE.core.navigation_utility import ocean_current_processing




class Task():

    def __init__(self, mission, vehicle, power_consumption_list):
        self.mission = mission
        self.vehicle = vehicle
        self.power_consumption_list = power_consumption_list

        self.get_load_demand()


    def get_ocean_current(self):
        self.ocean_current_df = ocean_current_processing(self.mission.df)
        pass

    def get_hotel_load(self, strategy='normal'):
        hotel = HotelLoad(self.mission, self.power_consumption_list, strategy)
        self.hotel_load = hotel.generate_power_consumption_timeseries()
        return self.hotel_load

    def get_propulsion_load(self, current=True):
        if current == False:
            self.prop_load = self.vehicle.prop_power()
        else:
            try:
                self.get_ocean_current()
            except FileNotFoundError:
                print('No ocean current file found!')
                self.prop_load = self.vehicle.prop_power()
                return self.prop_load
            if self.ocean_current_df.Vs.isnull().values.any():
                print('No current data from database, fallback on no current')
                self.prop_load = self.vehicle.prop_power()
            else:
                prop_power_list = []
                for v in self.ocean_current_df.Vs:
                    self.vehicle.speed = v
                    prop_power_list.append(self.vehicle.prop_power())

                self.prop_load = pd.Series(index=self.mission.df.index, data=prop_power_list)
        return self.prop_load

    def get_load_demand(self):
        self.load_demand = self.get_hotel_load() + self.get_propulsion_load()
        return self.load_demand

class Reactive_simulation(Task):

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
        wind_df['wind_raw'] = self.resource_df.V2.apply(
            lambda x: power_from_turbine(x, 1, self.power_coefficient ,self.cut_in_speed, self.rated_speed))
        wind_df['wind_correction'] = resistance_power(self.resource_df, 1)
        wind_df['wind_power'] = wind_df['wind_raw'] - wind_df['wind_correction']
        self.wind = wind_df
        return wind_df['wind_power']

    @property
    @lru_cache(maxsize=32)
    def solar_power_simulation(self):
        self.solar = pd.DataFrame()
        self.resource_df['global_horizontal'] = self.resource_df.SWGDN
        self.resource_df['diffuse_fraction'] = brl_model.location_run(self.resource_df)
        self.resource_df['solar_power'] = pv.run_plant_model_location(self.resource_df, self.tilt, self.azim, self.tracking, self.capacity)
        self.solar['solar_power'] = self.resource_df['solar_power']
        return self.solar['solar_power']

    def run(self, solar_area, wind_area, battery_capacity):
        power_supply = self.wind_power_simulation * wind_area + self.solar_power_simulation * solar_area
        supply, load = power_supply.tolist(), self.Task.load_demand.tolist()
        if self.config != {}:
            model = Soc_model_variable_load(Battery(battery_capacity, config=self.config), supply, load)
        else:
            model = Soc_model_variable_load(Battery(battery_capacity), supply, load)
        lpsp = model.get_lost_power_supply_probability()
        return lpsp

    def result(self, solar_area, wind_area, battery_capacity):
        power_supply = self.wind_power_simulation * wind_area + self.solar_power_simulation * solar_area
        supply, load = power_supply.tolist(), self.Task.load_demand.tolist()
        if self.config != {}:
            model = Soc_model_variable_load(Battery(battery_capacity, config=self.config), supply, load)
        else:
            model = Soc_model_variable_load(Battery(battery_capacity), supply, load)

        prop_load = (self.Task.load_demand - self.Task.hotel_load).as_matrix()
        load_demand = self.Task.load_demand.as_matrix()
        hotel_load = self.Task.hotel_load.as_matrix()


        load_demand_history = np.vstack((load_demand, prop_load, hotel_load))
        load_demand_history_df = pd.DataFrame(data=load_demand_history.T, index=self.Task.mission.df.index,
                                              columns=['Load_demand', 'Prop_load', 'Hotel_load'])
        load_demand_history_df = full_day_cut(load_demand_history_df)

        battery_history = model.get_battery_history()
        battery_history_df = pd.DataFrame(data=battery_history.T, index=self.df.index,
                                          columns=['SOC', 'Battery', 'Unmet', 'Waste', 'Supply'])

        results = [battery_history_df, load_demand_history_df, self.solar * solar_area, self.wind * wind_area]
        result_df = pd.concat(results, axis=1)
        return result_df

    def post_run(self, solar_area, wind_area, battery_capacity, dispatch):
        power_supply = self.wind_power_simulation * wind_area + self.solar_power_simulation * solar_area

        post_run_len = len(dispatch) # match length of simulation
        supply, load = power_supply[:post_run_len].tolist(), dispatch['Power'].tolist()

        if self.config != {}:
            model = Soc_model_variable_load(Battery(battery_capacity, config=self.config), supply, load)
        else:
            model = Soc_model_variable_load(Battery(battery_capacity), supply, load)

        prop_load = (self.Task.load_demand[:post_run_len] - self.Task.hotel_load[:post_run_len]).as_matrix()
        load_demand = self.Task.load_demand[:post_run_len].as_matrix()
        hotel_load = self.Task.hotel_load[:post_run_len].as_matrix()

        load_demand_history = np.vstack((load_demand, prop_load, hotel_load))
        load_demand_history_df = pd.DataFrame(data=load_demand_history[:post_run_len].T,
                                              index=self.Task.mission.df[:post_run_len].index,
                                              columns=['Load_demand', 'Prop_load', 'Hotel_load'])

        load_demand_history_df = full_day_cut(load_demand_history_df)

        battery_history = model.get_battery_history()
        battery_history_df = pd.DataFrame(data=battery_history[:post_run_len].T, index=self.df[:post_run_len].index,
                                          columns=['SOC', 'Battery', 'Unmet', 'Waste', 'Supply'])

        results = [battery_history_df, load_demand_history_df[:post_run_len],
                   self.solar[:post_run_len] * solar_area, self.wind[:post_run_len] * wind_area]
        result_df = pd.concat(results, axis=1)

        return result_df

class Sim:

    def __init__(self, mission):
        self.mission = full_day_cut(mission.df)
        self.resource_df = None
        self.solar_power = 0
        self.wind_power = 0
        self.battery_energy = 0

    def own_fun(self, inputs):
        return inputs+1

    @property
    def get_resource_df(self):
        self.resource_df = resource_df_download_and_process(self.mission)
        return self.resource_df

    def sim_wind(self, area, power_coefficient=0.3, cut_in_speed=2, rated_speed=15):
        """
        Simulation wind power generation considering the added resistance of wind due the wind turbine.

        :param area: float m^2 wind turbine swept area
        :param power_coefficient: float power coefficient of wind turbine
        :param cut_in_speed: float m/s cut-in speed of wind turbine
        :param rated_speed: float m/s cut-out speed of wind turbine
        :return: Nothing returned but wind_power Pandas series was created in class
        """

        wind_df = self.get_resource_df
        # Apply wind speed to ideal wind turbine model, get power production correction due to speed
        wind_df['wind_power'] = wind_df.V2.apply(
            lambda x: power_from_turbine(x, area, power_coefficient, cut_in_speed, rated_speed)) - \
                                resistance_power(wind_df, area)
        self.wind_power_raw = wind_df.V2.apply(
            lambda x: power_from_turbine(x, area, power_coefficient, cut_in_speed, rated_speed))
        self.wind_power = wind_df.wind_power
        pass

    def sim_solar(self, tilt, azim, tracking, capacity,
                  technology='csi', system_loss=0.10, angles=None, dataFrame=False,
                  **kwargs):
        """
        Simulate solar energy production based on various input of renewable energy system.

        :param tilt: float degrees title angle of PV panel
        :param azim: float degrees azim angle of PV panel
        :param tracking: int 0 1 or 2 0 for no tracking, 1 for one axis, 2 for two axis
        :param technology: optional str 'csi'
        :param system_loss: float system lost of the system
        :param angles: optional solar angle
        :param dataFrame: optional return dataframe or not
        :return: tuple of Pandas series solar power and wind power with datatime index
        """
        solar_df = self.get_resource_df
        solar_df['global_horizontal'] = solar_df.SWGDN
        solar_df['diffuse_fraction'] = brl_model.location_run(solar_df)
        solar_df['solar_power'] = pv.run_plant_model_location(solar_df, tilt, azim,
                                                              tracking, capacity, technology,
                                                              system_loss, angles, dataFrame, **kwargs)
        self.solar_power = solar_df.solar_power
        pass

    def sim_all(self, use, battery_capacity, shading_coef=0.05):
        """
        Simulation on both wind and PV system for hybrid renewable energy system.

        :param use:float Load of the system
        :param battery_capacity: float Wh total capacity of battery
        :return: None but update the remianing energy in battery
        """
        power = (1 - shading_coef)*self.solar_power + self.wind_power
        battery_energy = min_max_model(power, use, battery_capacity)
        self.battery_energy = pd.Series(battery_energy, index=self.mission.index)
        return self.battery_energy

@lru_cache(maxsize=32)
def power_unit_area(start_time, route, speed, power_per_square=140,
                    title=0, azim=0, tracking=0, power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15,
                    technology='csi', system_loss=0.10, angles=None, dataFrame=False
                    ):
    """
    Get power output of wind and PV system in a 1 metre square unit area.

    :param start_time: str or Dateindex start date of journey
    :param route: numpy nd array (n,2) [lat, lon] of way points
    :param speed: float or list if float is given, it is assumed as constant speed operation mode
                otherwise, a list of n with averaged speed should be given
    :param power_per_square: float W/m^2 rated power per squared meter
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
    :return: tuple of Pandas series solar power and wind power with datatime index
    """
    route = np.array(route).reshape(-1, 2)
    # Unpack route to ndarray for route based processing
    mission = Mission(start_time, route, speed)
    sim = Sim(mission)
    sim.sim_wind(1, power_coefficient, cut_in_speed, cut_off_speed)
    sim.sim_solar(title, azim, tracking, power_per_square, technology, system_loss, angles, dataFrame)
    solar_power = sim.solar_power
    wind_power = sim.wind_power
    return solar_power, wind_power






if __name__ == '__main__':
    test_route = np.array([[10.69358, -178.94713892], [11.06430687, +176.90022735]])
    from PyResis import propulsion_power

    test_ship = propulsion_power.Ship()
    test_ship.dimension(5.72, 0.248, 0.76, 1.2, 5.72 / (0.549) ** (1 / 3), 0.613)

    power_consumption_list = {'single_board_computer': {'power': [2, 10], 'duty_cycle': 0.5},
                              'webcam': {'power': [0.6], 'duty_cycle': 1},
                              'gps': {'power': [0.04, 0.4], 'duty_cycle': 0.9},
                              'imu': {'power': [0.67, 1.1], 'duty_cycle': 0.9},
                              'sonar': {'power': [0.5, 50, 0.2], 'duty_cycle': 0.5},
                              'ph_sensor': {'power': [0.08, 0.1], 'duty_cycle': 0.95},
                              'temp_sensor': {'power': [0.04], 'duty_cycle': 1},
                              'wind_sensor': {'power': [0.67, 1.1], 'duty_cycle': 0.5},
                              'servo_motors': {'power': [0.4, 1.35], 'duty_cycle': 0.5},
                              'radio_transmitter': {'power': [0.5, 20], 'duty_cycle': 0.2}}

    test_mission = Mission('2014-01-01', test_route, 2)
    test_task = Task(test_mission, test_ship, power_consumption_list)

    rea_sim = Reactive_simulation(test_task)
    print(rea_sim.run(1, 1, 300))


