import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd
import nvector as nv
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
    latTS = pd.Series(way_points[:, 0], index=timeindex).resample('1H').mean()
    lonTS = pd.Series(way_points[:, 1], index=timeindex).resample('1H').mean()

    # A custom interpolation to deal with singular angle in 180 lon
    lTs = latTS.copy()
    x = lTs.isnull().reset_index(name='null').reset_index().rename(columns={"level_0": "order"})
    x['block'] = (x['null'].shift(1) != x['null']).astype(int).cumsum()
    block_to_fill = x[x['null']].groupby('block')['order'].apply(np.array)

    def find_start_and_end_location(block_index):
        start_index = block_index[0] - 1
        end_index = block_index[-1] + 1
        lat1 = latTS.iloc[start_index]
        lon1 = lonTS.iloc[start_index]
        lat2 = latTS.iloc[end_index]
        lon2 = lonTS.iloc[end_index]
        n = (end_index - start_index)
        lat_lon1 = lat1, lon1
        lat_lon2 = lat2, lon2
        return [lat_lon1, lat_lon2, n]

    def way_points_interp(location_block):
        lat_lon1 = location_block[0]
        lat_lon2 = location_block[1]
        n = location_block[2]
        wgs84 = nv.FrameE(name='WGS84')
        lat1, lon1 = lat_lon1
        lat2, lon2 = lat_lon2
        n_EB_E_t0 = wgs84.GeoPoint(lat1, lon1, degrees=True).to_nvector()
        n_EB_E_t1 = wgs84.GeoPoint(lat2, lon2, degrees=True).to_nvector()
        path = nv.GeoPath(n_EB_E_t0, n_EB_E_t1)
        interpolate_coor = [[lat1, lon1]]
        piece_fraction = 1 / n
        for n in range(n - 1):
            g_EB_E_ti = path.interpolate(piece_fraction * (n + 1)).to_geo_point()
            interpolate_coor.append([g_EB_E_ti.latitude_deg[0], g_EB_E_ti.longitude_deg[0]])
        return interpolate_coor

    way_interpolated = np.array([])
    for block in block_to_fill:
        way_interp = way_points_interp(find_start_and_end_location(block))
        way_interpolated = np.append(way_interpolated, way_interp)

    way_interpolated = np.append(way_interpolated, [latTS.iloc[-1], lonTS.iloc[-1]])
    locations = way_interpolated.reshape(-1, 2)

    mission = pd.DataFrame(data=locations, index=latTS.index, columns=['lat', 'lon'])

    if isinstance(speed, int) or isinstance(speed, float):
        speedTS = speed
    else:
        speed = np.append(speed, speed[-1])
        speedTS = pd.Series(speed, index=timeindex).resample('1H').mean()

    mission['speed'] = speedTS

    time_diff = np.floor(mission.lon / 180 * 12)
    # time difference into timedelta
    t_diff = time_diff.map(lambda x: timedelta(hours=x))
    local_time = mission.index + t_diff
    mission['local_time'] = local_time

    return mission


def get_position_df(start_time, route, speed):
    if type(start_time) == str:
        start_time = pd.Timestamp(start_time)
    else:
        pass
    position_df = position_dataframe(start_time, route, speed)
    return position_df




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

class Mission():

    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.df = get_position_df(self.start_time, self.route, self.speed)

    def generate_more(self):
        pass

if __name__ == '__main__':
    test_route = np.array(
    [   [9.20628817, 171.58565184],
        [9.48566359, 174.60574911],
        [9.95078073, 176.68206597],
        [10.69358, 178.94713892],
        [11.06430687, -176.90022735],
        [10.87900106, -172.27570342],
        [9.95078073, -168.97247204],
        [9.67178793, -166.89615517],
        [8.92669178, -164.53670418],
        [8.36686209, -163.12103359],
        [7.61917834, -161.61098496],
        [7.05755065, -160.66720457],
        [6.30766145, -159.15715593],
        [5.93230149, -158.1189975],
        [-1.60710319, -156.04268063]])
    test_mission = get_position_df('2014-01-01', test_route, 2)
    print(hashFor(test_mission)[:7])