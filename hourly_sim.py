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
    power = power.tolist()
    SOC = 0
    SOC_history = []
    for p in power:
        SOC = min(battery_capacity,
                max(0, SOC + (p - use) *1 ))
        SOC_history.append(SOC)

    return SOC_history


class Simulation:
    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.mission = route_manager.get_position_df(self.start_time,                                                      self.route, self.speed)
        self.resource_df = None
        self.solar_power = 0
        self.wind_power = 0
        self.battery_energy = 0

    @property
    def get_resource_df(self):
        self.resource_df = route_based_download.resource_df_download_and_process(self.mission)
        return self.resource_df

    def sim_wind(self, area, power_coefficient=0.3, cut_in_speed=2, cut_off_speed=15):

        def power_from_turbine(wind_speed, area, power_coefficient, cut_in_speed, cut_off_speed):
            Cp = power_coefficient
            A = area
            power = 0
            v = wind_speed
            if v < cut_in_speed:
                power = 0
            elif cut_in_speed < v < cut_off_speed:
                power = 1 / 2 * Cp * A * v ** 3
            elif cut_off_speed < v < 3*cut_off_speed :
                power = 1 / 2 * Cp * A * cut_off_speed ** 3
            elif v > 3*cut_off_speed:
                power = 0

            return power

        def ship_speed_correction(df, area):
            A = area
            power_correction = 1/2*A*0.6*df.V2**2*np.cos(df.apparent_wind_direction)*df.speed
            return power_correction
        wind_df = self.get_resource_df
        wind_df['wind_power'] = wind_df.V2.apply(lambda x: power_from_turbine(x,area,
                            power_coefficient, cut_in_speed, cut_off_speed)) -\
                                ship_speed_correction(wind_df, area)

        self.wind_power = wind_df.wind_power
        pass


    def sim_solar(self, title, azim, tracking, capacity,
                  technology='csi', system_loss=0.10, angles=None, dataFrame=False,
                  **kwargs):

        solar_df = self.get_resource_df
        solar_df = full_day_cut(solar_df).copy()
        solar_df['global_horizontal'] = solar_df.SWGDN
        solar_df['diffuse_fraction'] = brl_model.location_run(solar_df)
        solar_df['solar_power'] = pv.run_plant_model_location(solar_df, title, azim,
                                                        tracking, capacity, technology,
                                                        system_loss, angles, dataFrame, **kwargs)
        self.solar_power = solar_df.solar_power
        pass

    def sim_all(self, use, battery_capcity):
        power = self.solar_power + self.wind_power
        battery_energy = min_max_model(power, use, battery_capcity)
        self.battery_energy = pd.Series(battery_energy, index=self.mission.index)
        return  self.battery_energy


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
            self.solar_df = synthetic_radiation(self.position_df)
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
    route1 = route_manager.opencpn_coordinates_processing(
        "50.1871,26.1087,0. 52.5835,26.431,0. 54.1412,26.1087,0. 55.9985,26.5919,0. 57.017,25.9472,0. 57.9157,24.8105,0. 60.3721,23.5535,0. 61.091,20.4993,0. 59.054,18.1817,0. 56.6575,16.4085,0. 52.5835,14.445,0. 49.4082,13.0484,0. 45.3941,11.937,0. 43.8364,12.23,0. 41.6197,15.1402,0. 39.8823,18.2955,0. 37.9651,21.7288,0. 36.1677,24.5928,0. 35.0294,26.431,0. 32.8726,29.2385,0. 30.7158,32.7302,0. 26.7616,33.8818,0. 20.7705,34.8216,0. 15.678,35.2631,0. 11.4841,37.3867,0. 6.99077,38.0029,0. 2.55731,37.6243,0. -1.27704,36.6211,0. -7.50785,35.7021,0. -10.9228,35.0672,0. -17.4532,34.8708,0. -24.8223,34.8708,0. -30.9333,34.4272,0. -38.1227,34.1797,0. -46.9297,33.483,0. -52.142,33.0321,0. -58.4927,32.4779,0. -64.4838,32.0727,0. -70.0556,31.8694,0. -75.3278,31.8694,0. -78.6829,31.4614,0. -81.3789,31.4103,0. ")
    route2 = route_manager.opencpn_coordinates_processing(
        "-69.4565,20.7236,0. -68.3182,21.8401,0. -67.4195,23.4436,0. -66.0415,25.2989,0. -64.7235,26.7525,0. -63.5851,28.2932,0. -61.6081,29.7599,0. -59.2715,31.2055,0. -56.5156,33.1325,0. -54.4786,34.2788,0. -52.7411,35.0672,0. -50.2248,36.1388,0. -47.5288,37.196,0. -45.4319,37.8139,0. -42.1368,39.2201,0. -38.6619,40.2795,0. -36.8046,41.0972,0. -33.5095,42.2162,0. -30.7536,43.0974,0. -28.5368,43.9229,0. -25.721,44.7372,0. -23.3844,45.624,0. -20.928,46.4144,0. -17.6928,47.1936,0. -14.877,47.7202,0. -12.5404,48.0417,0. -9.84439,48.7181,0. -6.90872,49.1902,0. -3.7334,49.89,0. -1.39685,50.7696,0.")
    route3 = route_manager.opencpn_coordinates_processing(
        "-37.7862,52.9074,0. -39.8625,52.1032,0. -40.7119,51.4021,0. -41.8444,50.1489,0. -42.3163,49.0479,0. -43.0713,47.8587,0. -43.732,46.6415,0. -44.2983,45.2637,0. -44.6758,44.1232,0. -44.7702,43.2359,0. -44.7702,41.9857,0. -45.0533,40.7104,0. -45.0533,39.8464,0. -43.3545,39.7014,0. -42.5995,40.3518,0. -41.9388,41.3512,0. -41.2782,42.8911,0. -40.9007,44.1232,0. -39.8625,45.6609,0. -39.2018,46.6415,0. -38.6356,47.4135,0. -38.0693,48.0483,0. -37.503,48.862,0. -36.8424,49.1714,0. -36.0874,49.9064,0. -35.4267,50.4503,0. -34.1054,50.9881,0. -32.4066,51.9871,0. -31.2741,52.5071,0. -29.9528,53.304,0. -27.9709,54.1416,0. -26.272,55.1788,0. -25.517,55.6076,0. -24.5732,56.6074,0. -24.3845,57.2767,0. -24.3845,57.8838,0. -24.2901,58.4316,0. -26.1777,59.4059,0. -28.3484,59.4539,0. -30.3303,59.1165,0. -31.4628,58.3326,0. -32.5954,57.7834,0. -33.256,57.1233,0. -33.8223,56.2945,0. -34.4829,55.3401,0. -34.8605,54.9084,0. -35.238,54.6908,0. -36.5593,53.8643,0. -38.3524,53.5851,0.")


    s = Simulation('2014-12-01', route2, 3)
    s.sim_wind(3)
    s.sim_solar(0, 0, 2, 100)
    print(s.sim_all(20,30))

