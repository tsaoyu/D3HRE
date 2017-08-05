import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def opencpn_coordinates_processing(coordinates: str) -> np.ndarray:
    """
    Process coordinate copy from OpenCPN xml file to numpy 
    array based route with [lat, lon] pairs.
    :param coordinates: string coordinates from OpenCPN
    :return: (n, 2) array with n way points [lat, lon] 
    """
    str_list = list(map(lambda x: x.split(','), coordinates.split(' ')))
    del str_list[-1]
    route = np.array(str_list, dtype=float).T[:2].T
    return route

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param lon1: longitude of point 1
    :param lat1: latitude of point 1
    :param lon2: longitude of point 2
    :param lat2: latitude of point 2 
    :return: great circle distance between two points in km
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def distance_between_waypoints(way_points):
    """
    Use Haversine method to calculate distance between great circle.
    :param way_points: array a list of way points
    :return: np array Haversine distance between two way points
             in km 
    """
    distance = np.array([haversine(v[0], v[1], w[0], w[1]) for v, w in
                         zip(way_points[:-1], way_points[1:])])
    return distance


def journey_timestamp_generator(start_date, way_points, speed):
    """
    Journey timestamp generator
    is an utility function that takes a route with several way points
    and speed to generate pandas  Timestamp index for
    dataFrame construction
    :param start_date: pandas Timestamp in UTC
    :param way_points: array lik List way points
    :param speed: float or int speed in km/h
    :return: pandas Timestamp index
    """
    distance = distance_between_waypoints(way_points)
    time = distance / speed  # workout voyage time based on constant speed
    cumtime = np.cumsum(time)
    timestamps = [start_date + timedelta(hours=t) for t in cumtime]
    timestamps.insert(0, start_date)
    return timestamps


def position_dataframe(start_date, way_points, speed):
    """
    Generate hourly based position dataframe based on few way points.
    :param start_date: pandas Timestamp in UTC
    :param way_points: way points
    :param speed: float or array speed in km/h
        if it is an array then the size should be one smaller than way points
    :return: pandas dataFrame hourly indexed position
    """
    timeindex = journey_timestamp_generator(start_date, way_points, speed)
    latTS = pd.Series(way_points[:,0], index=timeindex).resample('1H') \
        .mean().interpolate(method='pchip')
    lonTS = pd.Series(way_points[:,1], index=timeindex).resample('1H') \
        .mean().interpolate(method='pchip')
    if isinstance(speed, int) or isinstance(speed, float):
        speedTS = speed
    else:
        speed = np.append(speed, speed[-1])
        speedTS = pd.Series(speed, index=timeindex).resample('1H') \
            .mean().ffill()

    items = [
        ('lat', latTS),
        ('lon', lonTS),
        ('speed', speedTS)
    ]

    mission = pd.DataFrame.from_items(items)
    return mission


def timezone_alignment(mission):
    """
    Convert UTC time into local time zone of a mission dataframe
    :param mission: with time index in UTC and longitude information
    :return: pandas Dataframe with additional local time column
    """
    # get the time index and longitude from the mission
    index = mission.index
    lon = mission.lon
    # map longitude difference into time difference
    timediff = np.floor(lon / 180 * 12)
    # time difference into timedelta
    tdiff = timediff.map(lambda x: timedelta(hours=x))
    local_time = index + tdiff
    mission['local_time'] = local_time
    return mission


def nearest_point(lat_lon):
    """
    Find nearest point in a grid of 1 by 1 degree
    :param lat_lon: tuple (lat, lon)
    :return: nearest coordinates tuple
    """
    lat, lon = lat_lon
    lat_coords = np.arange(-90, 90, dtype=int)
    lon_coords = np.arange(-180, 180, dtype=int)

    def find_closest_coordinate(calc_coord, coord_array):
        index = np.abs(coord_array - calc_coord).argmin()
        return coord_array[index]

    lat_co = find_closest_coordinate(lat, lat_coords)
    lon_co = find_closest_coordinate(lon, lon_coords)
    return lat_co, lon_co

def full_day_cut(df):
    '''
    Cut mission into fully day length mission for simulation
    :param df: pandas data frame
    :return: pandas data frame that end at full day
    '''
    df = df[0:int(np.floor(len(df) / 24)) * 24]
    return df

# MD5 hash checksum is use to link a mission and download file
# The idea is it is hard to give a name of a route or just by name them with
# first way points. We can check the sum of a mission, it will be a unique
# number that help us to track the download file for each mission
def hashFor(data):
    """
    Prepare the project id hash
    :param data: raw data for the hash
    :return: the hash code for the data (full length)
    """
    hashId = hashlib.md5()
    hashId.update(repr(data).encode('utf-8'))
    return hashId.hexdigest()



def get_position_df(start_time, route, speed):
    if type(start_time) == str:
        start_time = pd.Timestamp(start_time)
    else:
        pass
    position_df = position_dataframe(start_time, route, speed).pipe(timezone_alignment)
    return position_df


class Mission():

    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.df = get_position_df(self.start_time, self.route, self.speed)

    def generate_more(self):
        pass


if __name__ == '__main__':

    route1 = opencpn_coordinates_processing(
        "50.1871,26.1087,0. 52.5835,26.431,0. 54.1412,26.1087,0. 55.9985,26.5919,0. "
        "57.017,25.9472,0. 57.9157,24.8105,0. 60.3721,23.5535,0. 61.091,20.4993,0. 59.054,"
        "18.1817,0. 56.6575,16.4085,0. 52.5835,14.445,0. 49.4082,13.0484,0. 45.3941,11.937"
        ",0. 43.8364,12.23,0. 41.6197,15.1402,0. 39.8823,18.2955,0. 37.9651,21.7288,"
        "0. 36.1677,24.5928,0. 35.0294,26.431,0. 32.8726,29.2385,0. 30.7158,32.7302,"
        "0. 26.7616,33.8818,0. 20.7705,34.8216,0. 15.678,35.2631,0. 11.4841,37.3867,"
        "0. 6.99077,38.0029,0. 2.55731,37.6243,0. -1.27704,36.6211,0. -7.50785,35.7021,"
        "0. -10.9228,35.0672,0. -17.4532,34.8708,0. -24.8223,34.8708,0. -30.9333,34.4272,"
        "0. -38.1227,34.1797,0. -46.9297,33.483,0. -52.142,33.0321,0. -58.4927,32.4779,0."
        " -64.4838,32.0727,0. -70.0556,31.8694,0. -75.3278,31.8694,0. -78.6829,31.4614,0. "
        "-81.3789,31.4103,0. ")

    mission1 = get_position_df('2014-01-01', route1, 2)
    print(mission1.head())
    print(hashFor(mission1)[0:7])
