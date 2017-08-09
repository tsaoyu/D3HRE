from functools import lru_cache

import numpy as np
import pandas as pd

from mission_utlities import route_manager
from opendap_download import route_based_download
from gsee.gsee import brl_model, pv


def full_day_cut(df):
    '''
    Cut mission into fully day length mission for simulation
    :param df: pandas dataframe
    :return: pandas dataframe that end at full day
    '''
    df = df[0:int(np.floor(len(df) / 24)) * 24]
    return df


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


def soc_model_fixed_load(power, use, battery_capacity, depth_of_discharge=0.6,
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
    energy = DOD * battery_capacity
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

    if not battery_capacity:
        SOC = np.array(energy_history) / battery_capacity
    else:
        # Zero battery size selected, energy history is returned instead
        SOC = np.array(energy_history)
    return SOC, energy_history, unmet_history, waste_history, use_history

@lru_cache(maxsize=32)
def power_unit_area(start_time, route, speed, power_per_square=140,
                    title=0, azim=0, tracking=0, power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15,
                    technology='csi', system_loss=0.10, angles=None, dataFrame=False
                    ):
    """

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
    sim = Simulation(start_time, route, speed)
    sim.sim_wind(1, power_coefficient, cut_in_speed, cut_off_speed)
    sim.sim_solar(title, azim, tracking, power_per_square, technology, system_loss, angles, dataFrame)
    solar_power = sim.solar_power
    wind_power = sim.wind_power
    return solar_power, wind_power


def temporal_optimization(start_time, route, speed, solar_area, wind_area, use, battery_capacity, depth_of_discharge=0.6,
                           discharge_rate=0.005, battery_eff=0.9, discharge_eff=0.8,title=0, azim=0, tracking=0,
                          power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15, technology='csi', system_loss=0.10,
                          angles=None, dataFrame=False, trace_back=False, performance_index=False
                          ):
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
    solar_power_unit, wind_power_unit = power_unit_area(start_time, route, speed,
                                                        title=0, azim=0, tracking=0, power_coefficient=0.3,
                                                        cut_in_speed=2, cut_off_speed=15,
                                                        technology='csi', system_loss=0.10, angles=None, dataFrame=False
                                                        )
    solar_power = solar_power_unit * solar_area
    wind_power = wind_power_unit * wind_area
    power = solar_power + wind_power
    SOC, energy_history, unmet_history, waste_history, use_history =\
        soc_model_fixed_load(power, use, battery_capacity, depth_of_discharge,
                             discharge_rate, battery_eff, discharge_eff)
    LPSP = 1- unmet_history.count(0)/len(energy_history)
    if trace_back:
        if performance_index:
            return LPSP, np.array(energy_history), np.array(unmet_history).sum(), np.array(waste_history).sum()
        return  LPSP, SOC, energy_history, unmet_history, waste_history
    else:
        return  LPSP


class Simulation:
    def __init__(self, start_time, route, speed):
        """
        :param start_time: str or Dateindex start date of journey
        :param route: numpy nd array (n,2) [lat, lon] of way points
        :param speed: float or list if float is given, it is assumed as constant speed operation mode
                otherwise, a list of n with averaged speed should be given
        """
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.mission = route_manager.get_position_df(self.start_time, self.route, self.speed)
        self.resource_df = None
        self.solar_power = 0
        self.wind_power = 0
        self.battery_energy = 0

    @property
    def get_resource_df(self):
        self.resource_df = route_based_download.resource_df_download_and_process(self.mission)
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

        def power_from_turbine(wind_speed, area, power_coefficient, cut_in_speed, cut_off_speed):
            Cp = power_coefficient
            A = area
            power = 0
            v = wind_speed
            if v < cut_in_speed:
                power = 0
            elif cut_in_speed < v < cut_off_speed:
                power = 1 / 2 * Cp * A * v ** 3
            elif cut_off_speed < v < 3 * cut_off_speed:
                power = 1 / 2 * Cp * A * cut_off_speed ** 3
            elif v > 3 * cut_off_speed:
                power = 0

            return power

        def ship_speed_correction(df, area):
            """
            :param df: Pandas dataframe contains apparent wind direction wind speed and speed of boat
            :param area: area of wind turbine
            :return: power correction term of due to the apparent wind angle
            """
            A = area
            power_correction = 1 / 2 * A * 0.6 * df.V2 ** 2 * np.cos(df.apparent_wind_direction) * df.speed
            return power_correction

        wind_df = self.get_resource_df
        # Apply wind speed to ideal wind turbine model, get power production correction due to speed
        wind_df['wind_power'] = wind_df.V2.apply(
            lambda x: power_from_turbine(x, area, power_coefficient, cut_in_speed, rated_speed)) - \
                                ship_speed_correction(wind_df, area)
        self.wind_power_raw = wind_df.V2.apply(
            lambda x: power_from_turbine(x, area, power_coefficient, cut_in_speed, rated_speed))
        self.wind_power = wind_df.wind_power
        pass

    def sim_solar(self, title, azim, tracking, capacity,
                  technology='csi', system_loss=0.10, angles=None, dataFrame=False,
                  **kwargs):
        """
        Simulate solar energy production based on various input of renewable energy system.
        :param title: float degrees title angle of PV panel
        :param azim: float degrees azim angle of PV panel
        :param tracking: int 0 1 or 2 0 for no tracking, 1 for one axis, 2 for two axis
        :param technology: optional str 'csi'
        :param system_loss: float system lost of the system
        :param angles: optional solar angle
        :param dataFrame: optional return dataframe or not
        :return: tuple of Pandas series solar power and wind power with datatime index
        """
        solar_df = self.get_resource_df
        solar_df = full_day_cut(solar_df).copy()
        solar_df['global_horizontal'] = solar_df.SWGDN
        solar_df['diffuse_fraction'] = brl_model.location_run(solar_df)
        solar_df['solar_power'] = pv.run_plant_model_location(solar_df, title, azim,
                                                              tracking, capacity, technology,
                                                              system_loss, angles, dataFrame, **kwargs)
        self.solar_power = solar_df.solar_power
        pass

    def sim_all(self, use, battery_capacity):
        """
        :param use:float Load of the system
        :param battery_capacity: float Wh total capacity of battery
        :return: None but update the remianing energy in battery
        """
        power = self.solar_power + self.wind_power
        battery_energy = min_max_model(power, use, battery_capacity)
        self.battery_energy = pd.Series(battery_energy, index=self.mission.index)
        return self.battery_energy


class Simulation_synthetic:
    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.position_df = route_manager.get_position_df(self.start_time, self.route, self.speed)
        self.solar_df = None

    @property
    def generate_solar(self):
        if self.solar_df is None:
            import supply
            self.solar_df = supply.solar_synthetic.synthetic_radiation(self.position_df)
        return self.solar_df

    def sim(self, title, azim, tracking, capacity,
            technology='csi', system_loss=0.10, angles=None, dataFrame=False,
            **kwargs):
        df = full_day_cut(self.generate_solar).copy()
        df['diffuse_fraction'] = brl_model.location_run(df)
        df['solar_synthetic_power'] = pv.run_plant_model_location(df, title, azim, tracking, capacity,
                                                                  technology, system_loss, angles, dataFrame, **kwargs)
        return df


if __name__ == '__main__':
    route = np.array(
        [[20.93866679, 168.56555458],
         [18.45091531, 166.77237183],
         [16.01733564, 165.45107928],
         [13.92043435, 165.2623232],
         [12.17361734, 165.63983536],
         [10.50804555, 166.96112791],
         [9.67178793, 168.94306674],
         [9.20628817, 171.58565184],
         [9.48566359, 174.60574911],
         [9.95078073, 176.68206597],
         [10.69358, 178.94713892],
         [11.06430687, -176.90022735],
         [10.87900106, -172.27570342]])

    s = Simulation('2014-12-01', route, 3)
    s.sim_wind(3)
    s.sim_solar(0, 0, 2, 100)
    s.sim_all(20, 30)

    print(temporal_optimization('2014-12-01', route, 3, 1, 1, 2, 100))
