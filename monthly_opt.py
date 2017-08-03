import unittest

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pylab as plt


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


solar_radiation = pd.read_table('./renewable/global_radiation.txt',
                             skiprows=13, sep=' ', header=0).drop(
    ['Lat','Lon'], axis=1).as_matrix().reshape(180, 360, 13)



wind_speed = pd.read_table('./renewable/10yr_wspd10arpt.txt',
                             skiprows=7, sep=' ', header=0, na_values='na').drop(
    ['Lat','Lon'], axis=1).as_matrix().reshape(180, 360, 13)



def get_sse_solar(lat_lon, annual=None, eff=None):
    """
    Get SSE solar radiation from database
    :param lat_lon: tuple of float (lat, lon)
    :param annual: optional annual=True to return annual averaged data
    :param eff: optional efficiency of solar panel if given it will
            return 12 month solar energy density in W/m^2
    :return: array of 12 months solar radiation at W/m2
    """
    lat, lon = nearest_point(lat_lon)
    if annual:
        solar = (solar_radiation[lat+90, lon+180, 12]) * 1000 / 24
    # from kW/day to W/hour
    else:
        solar = (solar_radiation[lat+90, lon+180, 0:12])  * 1000 / 24
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
    lat, lon = nearest_point(lat_lon)
    if annual:
        wind = wind_speed[lat+90, lon+180, 12]
    else:
        wind = wind_speed[lat+90, lon+180, 0:12]
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


def mission_resource_matrix(mission, solar_eff=0.12, wind_eff=0.26):
    mission['lat_lon'] = list(map(tuple, mission[['lat','lon']].values))
    month_index = mission.lat_lon.apply(nearest_point).drop_duplicates().index.month
    solar_resource_matrix = mission.lat_lon.apply(nearest_point).drop_duplicates(
        ).apply(lambda x: get_sse_solar(x, eff=solar_eff)).as_matrix()
    wind_resource_matrix = mission.lat_lon.apply(nearest_point).drop_duplicates(
        ).apply(lambda x: get_sse_wind(x, eff=wind_eff)).as_matrix()
    solar_for_month = []
    for way_point_order, month in zip(range((month_index - 1).shape[0]), month_index):
        solar_for_month.append(solar_resource_matrix[way_point_order][month - 1])

    wind_for_month = []
    for way_point_order, month in zip(range((month_index - 1).shape[0]), month_index):
        wind_for_month.append(wind_resource_matrix[way_point_order][month - 1])

    return np.vstack([solar_for_month, wind_for_month]).T

class Opt:
    """
    Monthly optimization on given route with objective minimize cost of system
    and subjectives in meet flat demand.
    """
    def __init__(self, route, wind_eff=0.26, solar_eff=0.12, cost=[1, 1], minimal_resource=None):
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
        self.power_demand = 0
        self.wind_eff = wind_eff
        self.solar_eff = solar_eff
        self.cost = cost
        self.minimal_resource = minimal_resource

    def resource_demand_matrix(self, zero_drop=True):
        A = np.array([])
        for way_point in self.route:
            resources_at_waypoint = (np.vstack(
                (get_sse_solar(tuple(way_point), eff=self.solar_eff),
                get_sse_wind(tuple(way_point), eff=self.wind_eff))).T)
            A = np.append(A, resources_at_waypoint)
        A = A.reshape(-1, 2)
        A = resource_matrix_handling(A, self.minimal_resource)
        if zero_drop:
            violated_cases = np.any(A == 0, axis=1).sum()
            print('Please be aware there are {v} violated cases that energy is 0'.format(v=violated_cases))
        A = A[~np.any(A == 0, axis=1)]
        b = self.power_demand * np.ones(A.shape[0])
        return A, b

    def route_based_presizing(self, power_demand):
        self.power_demand = power_demand
        A, b = self.resource_demand_matrix()
        self.route_resource_matrix = A
        res = linprog(self.cost, A_ub= -A, b_ub= -b,
                      options={"tol": 1e-8, "bland": True})
        return res.x

    def mission_based_preszing(self, start_date, speed, **kwargs):
        position_df = route_manager.get_position_df(start_date, self.route, speed)
        A = mission_resource_matrix(position_df, **kwargs)
        b = self.power_demand * np.ones(A.shape[0])
        self.mission_resource_matrix = A
        res = linprog(self.cost, A_ub=-A, b_ub=-b,
                      options={"tol": 1e-8, "bland": True})
        return res.x



#Utility functions
def print_optimization_result(result):
    """
    (more) Human readable optimization result
    :param result: tuple (As, Aw) area of solar and wind
    :return: none print out stuff
    """
    As, Aw = result
    print("The optimal design is solar panel at {a:.2f} m^2 "
          "and wind turbine swept area of {b:.2f} m^2".format(a=As, b=Aw))

def energy_resource_plot(resource_matrix):
    """
    Plot technical recoverable energy
    :param resource_matrix: np ndarray
     an (n,2) matrix that has solar and wind energy along route
    :return: none
    """
    ax = plt.subplot(111)
    ax.plot(resource_matrix)
    ax.legend(['Solar energy','Wind energy'])
    ax.set(title='Energy resource',xlabel='Travel time', ylabel='Technical recoverable energy $W/m^2$')
    plt.show()


def draw_constraint_lines(resource_matrix, power_demand, xlim=20, ylim=20, solution=None):
    """
    Plot constraint lines for linprog optimization
    :param resoure_matrix: np ndarray
    an (n,2) matrix that has solar and wind energy along route
    :param power_demand: float designed
    :param xlim: float x limits in plot
    :param ylim: float y limits in plot
    :param solution: optional insert the optimization solution as red dot
    :return: none
    """
    A = resource_matrix
    A = A[~np.any(A==0, axis=1)]
    solar_area = np.vstack([np.zeros(A.shape[0]),((power_demand / A) [:,0])]).T
    wind_area = np.vstack([((power_demand / A) [:,1]),np.zeros(A.shape[0])]).T
    for a, b in zip(solar_area,wind_area):
        plt.plot(a,b)
    plt.xlabel('Solar panel area $A_{solar}/m^2$')
    plt.ylabel('Wind generator swept area $A_{wind}/m^2$')
    if solution != None:
        plt.plot(solution[0],solution[1], 'ro')
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.show()


if __name__ == '__main__':
    print(get_sse_solar((10,10)))
    print(get_sse_wind((10, 10)))
