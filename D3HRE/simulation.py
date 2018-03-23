import numpy as np
import pandas as pd

from functools import lru_cache

from gsee.gsee import brl_model, pv
from D3HRE.core.dataframe_utility import full_day_cut
from D3HRE.core.battery_models import min_max_model
from D3HRE.core.weather_data_download import resource_df_download_and_process
from D3HRE.core.wind_turbine_model import power_from_turbine, resistance_power
from D3HRE.core.mission_utility import Mission


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
    mis = Mission('2014-12-01', route, 3)
    s = Sim(mis)
    s.sim_wind(3)
    s.sim_solar(0, 0, 2, 100)
    s.sim_all(20, 30)
