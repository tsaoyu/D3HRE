import numpy as np
import pandas as pd
import nvector as nv

from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

from D3HRE.core.get_hash import hash_value
from D3HRE.core.dataframe_utility import full_day_cut

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
    Journey timestamp generator is an utility function use way points
    and speed to generate Timestamp index for the construction of Pandas dataFrame

    :param start_date: pandas Timestamp in UTC
    :param way_points: array lik List way points
    :param speed: float or int speed in km/h
    :return: pandas Timestamp index
    """
    distance = distance_between_waypoints(way_points)
    time = distance / speed
    cumtime = np.cumsum(time)
    timestamps = [start_date + timedelta(hours=t) for t in cumtime]
    timestamps.insert(0, start_date)
    return timestamps


def position_dataframe(start_date, way_points, speed):
    """
    Generate position dataFrame at one hour resolution with given way points.

    :param start_date: pandas Timestamp in UTC
    :param way_points: way points
    :param speed: float or array speed in km/h
        if it is an array then the size should be one smaller than way points
    :return: pandas dataFrame with indexed position at one hour resolution
    """
    timeindex = journey_timestamp_generator(start_date, way_points, speed)
    latTS = pd.Series(way_points[:, 0], index=timeindex).resample('1H').mean()
    lonTS = pd.Series(way_points[:, 1], index=timeindex).resample('1H').mean()

    # Custom interpolation calculate latitude and longitude of platform at each hour
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
    mission.fillna(method='bfill', inplace=True)

    # Convert UTC time into local time
    def find_timezone(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx - 12

    lons = np.linspace(-180, 180, 25)

    local_time = []
    for index, row in mission.iterrows():
        local_time.append(index + timedelta(hours = int(find_timezone(lons, row.lon))))
    # time difference into timedelta
    # t_diff = list(map(lambda x: timedelta(hours=x), time_diff))
    #local_time = mission.index + t_diff
    mission['local_time'] = local_time

    return mission


def get_mission(start_time, route, speed):
    """
    Calculate position dataFrame at given start time, route and speed

    :param start_time: str or Pandas Timestamp format YYYY-MM-DD assume 00:00
    :param route: numpy array shape (n,2)  list of way points formatted as [lat, lon]
    :param speed: int, float or (n) list, speed of platform unit in km/h
    :return: Pandas dataFrame
    """
    if type(start_time) == str:
        start_time = pd.Timestamp(start_time)
    else:
        pass
    position_df = full_day_cut(position_dataframe(start_time, route, speed))
    return position_df


def nearest_point(lat_lon):
    """
    Find nearest point in a 1 by 1 degree grid

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


class Mission():
    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.df = get_mission(self.start_time, self.route, self.speed)
        self.get_ID()

    def __str__(self):
        return "This mission {ID} is start from {a} at {b} UTC.".format(
            a=self.route[0], b=self.start_time, ID=self.ID)

    def get_ID(self):
        route_tuple = tuple(self.route.flatten().tolist())
        if isinstance(self.speed, list):
            speed_tuple = tuple(self.speed)
        else:
            speed_tuple = self.speed

        ID_tuple = (self.start_time, route_tuple, speed_tuple)
        self.ID = hash_value(ID_tuple)
        pass




if __name__ == '__main__':
    test_route = np.array(
        [[9.20628817, 171.58565184],
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
    test_mission = get_mission('2014-01-01', test_route, 2)
    speed = np.linspace(2, 5, num=test_route.shape[0]-1)
    variable_speed_mission = get_mission('2014-01-01', test_route, 2)
    print(test_mission.lon.min())
    print(variable_speed_mission.tail())
    print(hash_value(test_mission)[:7])
    print(hash_value(test_route)[:7])
