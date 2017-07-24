import numpy as np
from math import radians, cos, sin, asin, sqrt

def opencpn_coordinates_processing(coordinates):
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
    latTS = pd.Series(way_points.T[1], index=timeindex).resample('1H') \
        .mean().interpolate(method='pchip')
    lonTS = pd.Series(way_points.T[0], index=timeindex).resample('1H') \
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


class

