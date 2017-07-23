import unittest

import pandas as pd
import numpy as np
from scipy.optimize import linprog

from mission_utlities import route_manager

def coordinates_processing(coordinates):
    """
    Process coordinate from OpenCPN xml file to numpy array of [lat, lon] points.
    :param coordinates: string coordinates from OpenCPN
    :return: (n, 2) array with n way points [lat, lon] 
    """
    str_list = list(map(lambda x: x.split(','), coordinates.split(' ')))
    del str_list[-1]
    route = np.array(str_list, dtype=float).T[:2].T
    return route


def nearest_point(lat_lon):
    """
    Find nearest grid point from decimal degree latitude and longitude tuple.
    1 by 1 grid of SSE database
    :param lat_lon: tuple of float or int (lat, lon)
    :return: tuple of int (lat, lon)
    """
    lat, lon = lat_lon
    lat_coords = np.arange(-90, 90, dtype=int)
    lon_coords = np.arange(-180, 180, dtype=int)

    def find_closest_coordinate(calc_coord, coord_array):
        index = np.abs(coord_array-calc_coord).argmin()
        return coord_array[index]
    lat_co = find_closest_coordinate(lat, lat_coords)
    lon_co = find_closest_coordinate(lon, lon_coords)
    return lat_co, lon_co


SSEradiation = pd.read_table('./renewable/global_radiation.txt',
                             skiprows=13, sep=' ', header=0)
SSEradiation.set_index(['Lat', 'Lon'], inplace=True)

SSEwindspeed = pd.read_table('./renewable/10yr_wspd10arpt.txt',
                             skiprows=7, sep=' ', header=0, na_values='na')
SSEwindspeed.set_index(['Lat', 'Lon'], inplace=True)


def get_sse_solar(lat_lon, annual=None, eff=None):
    """
    Get SSE solar radiation from database
    :param lat_lon: tuple of float (lat, lon)
    :param annual: optional annual=True to return annual averaged data
    :param eff: optional efficiency of solar panel if given it will
            return 12 month solar energy density in W/m^2
    :return: array of 12 months solar radiation at W/m2
    """
    x, y = nearest_point(lat_lon)
    if annual:
        solar = SSEradiation.loc[x, y][12] * 1000 / 24
    # from kW/day to W/hour
    else:
        solar = SSEradiation.loc[x, y][:12].values * 1000 / 24
    if eff:
        solar = eff * solar

    return solar


def get_sse_wind(lat_lon, annual=None, eff=None):
    """
    Get SSE wind speed from database
    :param lat_lon: tuple of float (lat, lon)
    :param annual: optional annual=True to return annual averaged data
    :param eff: optional efficiency of wind turbine if given it will
            return 12 month wind energy density in W/m^2
    :return: array of 12 months wind speed m/s
    """
    x, y = nearest_point(lat_lon)
    if annual:
        wind = SSEwindspeed.loc[x, y][12]
    else:
        wind = SSEwindspeed.loc[x, y][:12].values
    if eff:
        wind = eff * 0.5 * wind ** 3

    return wind


def resource_matrix_handling(A, minimal_resource=None):
    """
    Handle resource matrix with minial resource limitation.
    Resource matrix is matrix A in the linear programming
        min     cT.x
        s.t.    A.x < b
    This function eliminate nan values in the months. With
    optional minimal resource value, it can also eliminate
    and report the number of instance of months under total
    energy.
    :param A: array resource matrix
    :param minimal_resource: optional total least energy
    :return: array processed resource matrix
    """
    A = A[~np.isnan(A).any(axis=1)]
    if minimal_resource is not None:
        break_cases = (A[:, 0] + A[:, 1] < minimal_resource).sum()
        A = A[~((A[:, 0] + A[:, 1]) < minimal_resource)]
        print("Resource matrix break {t} months".format(t=break_cases))
    return A


class Opt:
    """
    Monthly optimization on given route with objective minimize cost of system
    and subjectives in meet flat demand.
    """
    def __init__(self, route, power_demand, wind_eff=0.26,
                 solar_eff=0.12, cost=[1, 1], minimal_resource=None):
        """

        :param route: array of lat,lon with dimension(n, 2)
            [[lat1,lon1], [lat2,lon2], ...]
        :param power_demand: float or int
            power demand in watts
        :param wind_eff: float
            wind generator efficiency
        :param solar_eff: float
            solar panel efficiency
        :param cost: array of cost matrix in
            [solar, wind]
        :param minimal_resource: float optional
            ignore extreme cases of month without total energy under this value
        """
        self.route = route
        self.power_demand = power_demand
        self.wind_eff = wind_eff
        self.solar_eff = solar_eff
        self.cost = cost
        self.minimal_resource = minimal_resource

    def resource_demand_matrix(self):
        A = np.array([])
        for way_point in self.route:
            resources_at_waypoint = (np.vstack(
                (self.solar_eff * get_sse_solar(tuple(way_point)),
                 1 / 2 * self.wind_eff * get_sse_wind(tuple(way_point)) ** 3)).T)
            A = np.append(A, resources_at_waypoint)
        A = A.reshape(-1, 2)
        A = resource_matrix_handling(A, self.minimal_resource)
        b = self.power_demand * np.ones(A.shape[0])
        return A, b

    def route_based_presizing(self):
        A, b = self.resource_demand_matrix()
        res = linprog(self.cost, A_ub= -A, b_ub= -b,
                      options={"tol": 1e-8, "bland": True})
        return res.x




if __name__ == '__main__':
    route1 = coordinates_processing(
        "50.1871,26.1087,0. 52.5835,26.431,0. 54.1412,26.1087,0. 55.9985,26.5919,0. 57.017,25.9472,0. 57.9157,24.8105,0. 60.3721,23.5535,0. 61.091,20.4993,0. 59.054,18.1817,0. 56.6575,16.4085,0. 52.5835,14.445,0. 49.4082,13.0484,0. 45.3941,11.937,0. 43.8364,12.23,0. 41.6197,15.1402,0. 39.8823,18.2955,0. 37.9651,21.7288,0. 36.1677,24.5928,0. 35.0294,26.431,0. 32.8726,29.2385,0. 30.7158,32.7302,0. 26.7616,33.8818,0. 20.7705,34.8216,0. 15.678,35.2631,0. 11.4841,37.3867,0. 6.99077,38.0029,0. 2.55731,37.6243,0. -1.27704,36.6211,0. -7.50785,35.7021,0. -10.9228,35.0672,0. -17.4532,34.8708,0. -24.8223,34.8708,0. -30.9333,34.4272,0. -38.1227,34.1797,0. -46.9297,33.483,0. -52.142,33.0321,0. -58.4927,32.4779,0. -64.4838,32.0727,0. -70.0556,31.8694,0. -75.3278,31.8694,0. -78.6829,31.4614,0. -81.3789,31.4103,0. ")
    route2 = coordinates_processing(
        "-69.4565,20.7236,0. -68.3182,21.8401,0. -67.4195,23.4436,0. -66.0415,25.2989,0. -64.7235,26.7525,0. -63.5851,28.2932,0. -61.6081,29.7599,0. -59.2715,31.2055,0. -56.5156,33.1325,0. -54.4786,34.2788,0. -52.7411,35.0672,0. -50.2248,36.1388,0. -47.5288,37.196,0. -45.4319,37.8139,0. -42.1368,39.2201,0. -38.6619,40.2795,0. -36.8046,41.0972,0. -33.5095,42.2162,0. -30.7536,43.0974,0. -28.5368,43.9229,0. -25.721,44.7372,0. -23.3844,45.624,0. -20.928,46.4144,0. -17.6928,47.1936,0. -14.877,47.7202,0. -12.5404,48.0417,0. -9.84439,48.7181,0. -6.90872,49.1902,0. -3.7334,49.89,0. -1.39685,50.7696,0.")
    route3 = coordinates_processing(
        "-37.7862,52.9074,0. -39.8625,52.1032,0. -40.7119,51.4021,0. -41.8444,50.1489,0. -42.3163,49.0479,0. -43.0713,47.8587,0. -43.732,46.6415,0. -44.2983,45.2637,0. -44.6758,44.1232,0. -44.7702,43.2359,0. -44.7702,41.9857,0. -45.0533,40.7104,0. -45.0533,39.8464,0. -43.3545,39.7014,0. -42.5995,40.3518,0. -41.9388,41.3512,0. -41.2782,42.8911,0. -40.9007,44.1232,0. -39.8625,45.6609,0. -39.2018,46.6415,0. -38.6356,47.4135,0. -38.0693,48.0483,0. -37.503,48.862,0. -36.8424,49.1714,0. -36.0874,49.9064,0. -35.4267,50.4503,0. -34.1054,50.9881,0. -32.4066,51.9871,0. -31.2741,52.5071,0. -29.9528,53.304,0. -27.9709,54.1416,0. -26.272,55.1788,0. -25.517,55.6076,0. -24.5732,56.6074,0. -24.3845,57.2767,0. -24.3845,57.8838,0. -24.2901,58.4316,0. -26.1777,59.4059,0. -28.3484,59.4539,0. -30.3303,59.1165,0. -31.4628,58.3326,0. -32.5954,57.7834,0. -33.256,57.1233,0. -33.8223,56.2945,0. -34.4829,55.3401,0. -34.8605,54.9084,0. -35.238,54.6908,0. -36.5593,53.8643,0. -38.3524,53.5851,0.")


    o = Opt(route2, 1000)
    print(o.route_based_presizing())