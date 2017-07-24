import glob
import hashlib
import math
import os.path
from datetime import timedelta
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import xarray as xr

import monthly_opt
from gsee.gsee import brl_model, pv
from opendap_download.multi_processing_download import DownloadManager
from synthetic_solar import synthsolar

# Username and password for MERRA-2 (NASA earthdata portal)
USERNAME = input('Username: ')
PASSWORD = getpass.getpass('Password:')
# The DownloadManager is able to download files. Set number of connections
# based on your internet speed
NUMBER_OF_CONNECTIONS = 6


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
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


def hour_in_the_year(df):
    """
    Find the hour in the year of a time index.
    It suppose the year is 365 days only.
    :param df: pandas dataframe contains local time
    :return: pandas daraframe with additional query_index
    """
    start_of_year = pd.Timestamp(str(df.local_time[0].year) + '-1-1')
    hour_difference = ((df.local_time - start_of_year) / np.timedelta64(1, 'h')
                       ).astype(int)
    query_index = np.mod(hour_difference, 8760)
    df['query_index'] = query_index
    return df


def _get_synthetic_solar_array(mission):
    """
    Generate synthetic solar radiation data for a mission.
    It treat all points within same 1 by 1 grid as same group and simulate solar radiation using Markov
    transition method to calculate hourly radiation.
    :param mission: pandas dataframe A mission contains lat, lon and local time
    :return: pandas dataframe with  Solar radiation trend component G0c, clearness index kt, and
    global horizontal irridiance G0
    """
    synthetic_solar_df = mission.copy()
    # create a copy of processing data to avoid contamination
    synthetic_solar_df = synthetic_solar_df.pipe(hour_in_the_year)
    # Convert local time into hour in the year
    # It then used as query index for synthetic generated solar data
    synthetic_solar_df['lat_lon'] = list(zip(synthetic_solar_df.lat,
                                             synthetic_solar_df.lon))
    synthetic_solar_df['lat_lon'] = synthetic_solar_df.lat_lon.apply(nearest_point)
    synthetic_group = synthetic_solar_df.groupby('lat_lon')

    # Group lat and lon and find the nearest point in the grid

    queries = list(zip(synthetic_group.first().query_index,
                       synthetic_group.last().query_index))

    # The generated synthetic data is whole year long [0:8760]
    # We only need a part of them for each position group

    G0list = np.array([])
    G0clist = np.array([])
    ktlist = np.array([])

    for lat_lon, query in zip(synthetic_group.first().index, queries):
        lat, lon = lat_lon
        start, end = query
        G0, G0c, kt = synthsolar.Aguiar_hourly_G0(synthsolar.monthlyKt(lat, lon), lat)
        if start > end:
            G0_slice = np.concatenate((G0[start:8760], G0[0: end + 1]))
            G0c_slice = np.concatenate((G0c[start:8760], G0c[0: end + 1]))
            kt_slice = np.concatenate((kt[start:8760], kt[0: end + 1]))
        else:
            G0_slice = G0[start:end + 1]
            G0c_slice = G0c[start:end + 1]
            kt_slice = kt[start:end + 1]

        G0list = np.append(G0list, G0_slice)
        G0clist = np.append(G0clist, G0c_slice)
        ktlist = np.append(ktlist, kt_slice)
        solar_data = np.vstack((G0list, G0clist, ktlist)).T

    local_time_index = pd.date_range(synthetic_solar_df.local_time[0],
                                     synthetic_solar_df.local_time[-1], freq='H')

    G0df = pd.DataFrame(solar_data, index=local_time_index,
                        columns=['global_horizontal', 'trend', 'kt']
                        ).reset_index().rename(columns={'index': 'local_time'})

    G0df_merge = pd.merge(G0df, synthetic_solar_df, on='local_time')
    G0df_merge.index = synthetic_solar_df.index
    return G0df_merge


def synthetic_radiation(mission):
    df = _get_synthetic_solar_array(mission)
    del df['query_index']
    del df['lat_lon']
    return df


def get_position_df(start_time, route, speed):
    if type(start_time) == str:
        start_time = pd.Timestamp(start_time)
    else:
        pass
    position_df = position_dataframe(start_time, route, speed).pipe(timezone_alignment)
    return position_df


def get_syntheic_solar_df(start_time, route, speed):
    return synthetic_radiation(get_position_df(start_time, route, speed))


def full_day_cut(df):
    '''
    Cut mission into fully day length mission for simulation
    :param df: pandas dataframe
    :return: pandas dataframe that end at full day
    '''
    df = df[0:int(np.floor(len(df) / 24)) * 24]
    return df


def hashFor(data):
    # Prepare the project id hash
    hashId = hashlib.md5()

    hashId.update(repr(data).encode('utf-8'))

    return hashId.hexdigest()



def generate_single_download_link(start, end, lat_lon, data_set=None):
    """
    Generate download URL at a period with given latitude and longitude.
    It support download link generation of multiple data set including wind, solar,
    pressure and air density.
    :param start: timestamp of start time in UTC
    :param end: timestamp of end time in UTC
    :param lat_lon: tuple of (lat, lon)
    :param data_set: str the dataset to be download support 'solar', 'wind', 'pressure'
                    and 'airdensity'
    :return: string the URL for the download of netCDF4 files
    """
    if data_set == 'solar':
        BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXRAD.5.12.4/'
        dataset_name = 'tavg1_2d_rad_Nx'
        parameters = ['SWGDN', 'SWTDN']
    elif data_set == 'wind':
        BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
        dataset_name = 'tavg1_2d_slv_Nx'
        parameters = ['U2M', 'U10M', 'U50M', 'V2M', 'V10M', 'V50M', 'DISPH', 'T2M']
    elif data_set == 'pressure':
        BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
        dataset_name = 'tavg1_2d_slv_Nx'
        parameters = ['PS']
    elif data_set == 'airdensity':
        BASE_URL = 'http://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXFLX.5.12.4/'
        dataset_name = 'tavg1_2d_flx_Nx'
        parameters = ['RHOA']
    else:
        print('Not supported yet')

    def translate_year_to_file_number(year):
        """
        The file names consist of a number and a meta data string.
        The number changes over the years. 1980 until 1991 it is 100,
        1992 until 2000 it is 200, 2001 until 2010 it is  300
        and from 2011 until now it is 400.
        """
        file_number = ''

        if 1980 <= year < 1992:
            file_number = '100'
        elif 1992 <= year < 2001:
            file_number = '200'
        elif 2001 <= year < 2011:
            file_number = '300'
        elif year >= 2011:
            file_number = '400'
        else:
            raise Exception('The specified year is out of range.')

        return file_number

    date = start

    year = date.year
    month = date.month
    day = date.day
    # get year month and day from date list
    y_str = str(year).zfill(2)
    m_str = str(month).zfill(2)
    d_str = str(day).zfill(2)
    # fill zero to date to match to file name format

    lat, lon = lat_lon
    file_num = translate_year_to_file_number(year)

    start_hour = start.hour
    end_hour = end.hour - 1
    if end_hour == -1:
        end_hour = 23

    hour = '[{s}:{e}]'.format(s=start_hour, e=end_hour)
    file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
        num=file_num, name=dataset_name,
        y=y_str, m=m_str, d=d_str)
    file_folder = '{y}/{m}/'.format(y=y_str, m=m_str)

    location = '[{lat}][{lon}]'.format(lat=lat, lon=lon)

    file_parameter = []
    for parameter in parameters:
        file_parameter.append(parameter + hour + location)

    file_parameter = ",".join(file_parameter)

    url = BASE_URL + file_folder + file_name + '.nc4?' + file_parameter
    return url


def download_URL(mission, data_set='solar'):
    def nearest_point(lat_lon):
        """
        The source for this formula is in the MERRA2
        Variable Details - File specifications for GEOS pdf file.
        The Grid in the documentation has points from 1 to 361 and 1 to 576.
        The MERRA-2 Portal uses 0 to 360 and 0 to 575.
        :param lat_lon: tuple (lat, lon)
        :return: nearest coordinates tuple
        """
        lat, lon = lat_lon
        lat_raw = (lat + 90) / 0.5
        lon_raw = (lon + 180) / 0.625

        lat_coords = np.arange(0, 361, dtype=int)
        lon_coords = np.arange(0, 576, dtype=int)

        def find_closest_coordinate(calc_coord, coord_array):
            index = np.abs(coord_array - calc_coord).argmin()
            return coord_array[index]

        lat_co = find_closest_coordinate(lat_raw, lat_coords)
        lon_co = find_closest_coordinate(lon_raw, lon_coords)
        return lat_co, lon_co

    download_df = mission.copy()
    download_df = download_df.reset_index().rename(columns={'index': 'utc'})
    download_df['lat_lon'] = download_df[['lat', 'lon']].apply(tuple, axis=1).apply(nearest_point)
    a = download_df.groupby('lat_lon').first().reset_index().set_index('utc')
    b = download_df.set_index('utc').resample('1D').ffill()
    c = download_df.groupby('lat_lon').last().reset_index().set_index('utc')
    #
    # a ----> Start time of query at a location
    #    b ----> End of day insert if a->c is over night
    # c ----> End time of query at a location
    #
    download_index = pd.concat([a, b, c], axis=1)['lat_lon'].iloc[:, 0].ffill()
    generated_URLs = []
    for start, end, lat_lon in zip(download_index.index[:-1],
                                   download_index.index[1:],
                                   download_index[1:]):
        generated_URLs.append(generate_single_download_link(start, end, lat_lon, data_set))

    return generated_URLs


def resource_df_download(mission, username=USERNAME, password=PASSWORD, n=NUMBER_OF_CONNECTIONS):
    folder = 'MERRA2data/' + hashFor(mission)[0:7]
    file_name = folder + 'resource.pkl'

    # Check if compact pandas data frame have already processed
    # if so, load the file and skip file download and processing
    if os.path.isfile(file_name):
        resource_df = pd.read_pickle(file_name)

    else:
        # Download when necessary
        download_manager = DownloadManager()
        download_manager.set_username_and_password(username, password)
        download_manager.download_path = folder + '/download_wind'
        download_manager.download_urls = download_URL(mission, data_set='wind')
        if not os.path.exists(download_manager.download_path):
            print('Wind data not found, automatic download starting ...')
            download_manager.start_download(n)

        download_manager.download_path = folder + '/download_solar'
        download_manager.download_urls = download_URL(mission, data_set='solar')
        if not os.path.exists(download_manager.download_path):
            print('Solar data not found, automatic download starting ...')
            download_manager.start_download(n)


        # Process when necessary
        if not os.path.isfile(file_name):
            print('Compact data not found, automatic processing starting ...')
            wind_files = sorted(glob.glob(folder + '/download_wind/MERRA*'))
            # Sort file to make sure time is aligned
            wind_xdata = xr.open_mfdataset(wind_files, concat_dim='time')
            wind_df = wind_xdata.to_dataframe()

            solar_files = sorted(glob.glob(folder + '/download_solar/MERRA*'))
            solar_xdata = xr.open_mfdataset(solar_files, concat_dim='time')
            solar_df = solar_xdata.to_dataframe()

            resource_df = pd.concat([solar_df, wind_df], axis=1)
            resource_df.reset_index(drop=True, inplace=True)
            resource_df['utc'] = mission.index[1:]

            resource_df.set_index('utc', inplace=True)

            resource_df.to_pickle(file_name)

        resource_df = pd.read_pickle(file_name)

    return resource_df


def min_max_model(power, use, battery_capacity):
    power = power.tolist()
    SOC = 0
    SOC_history = []
    for p in power:
        SOC = min(battery_capacity,
                max(0, SOC + (p - use) *1 ))
        SOC_history.append(SOC)

    return SOC_history

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def resource_df_download_and_process(mission):
    resource_df = resource_df_download(mission)
    df = pd.concat([mission, resource_df], axis=1).bfill()
    df['temperature'] = df.T2M - 273
    df['V2'] = np.sqrt(df.V2M ** 2 + df.U2M ** 2)
    df['kt'] = df.SWGDN / df.SWTDN
    location_list = [tuple(x) for x in df[['lat','lon']].values]
    heading = []
    for a, b in zip(location_list[:-1], location_list[1:]):
        heading.append(calculate_initial_compass_bearing(a, b))
    heading.append(heading[-1])
    df['heading'] = heading
    df['Vs'] = df['V2M'] - df['speed']/3.6*np.cos(np.radians(df['heading']))
    df['Us'] = df['U2M'] - df['speed']/3.6*np.sin(np.radians(df['heading']))
    df['apparent_wind_direction'] = np.arctan2(df['Vs'], df['Us'])
    return df


class Simulation:
    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.mission = get_position_df(self.start_time, self.route, self.speed)
        self.resource_df = None
        self.solar_power = 0
        self.wind_power = 0
        self.battery_energy = 0

    @property
    def get_resource_df(self):
        self.resource_df = resource_df_download_and_process(self.mission)
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
        self.position_df = get_position_df(self.start_time, self.route, self.speed)
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
    route1 = monthly_opt.coordinates_processing(
        "50.1871,26.1087,0. 52.5835,26.431,0. 54.1412,26.1087,0. 55.9985,26.5919,0. 57.017,25.9472,0. 57.9157,24.8105,0. 60.3721,23.5535,0. 61.091,20.4993,0. 59.054,18.1817,0. 56.6575,16.4085,0. 52.5835,14.445,0. 49.4082,13.0484,0. 45.3941,11.937,0. 43.8364,12.23,0. 41.6197,15.1402,0. 39.8823,18.2955,0. 37.9651,21.7288,0. 36.1677,24.5928,0. 35.0294,26.431,0. 32.8726,29.2385,0. 30.7158,32.7302,0. 26.7616,33.8818,0. 20.7705,34.8216,0. 15.678,35.2631,0. 11.4841,37.3867,0. 6.99077,38.0029,0. 2.55731,37.6243,0. -1.27704,36.6211,0. -7.50785,35.7021,0. -10.9228,35.0672,0. -17.4532,34.8708,0. -24.8223,34.8708,0. -30.9333,34.4272,0. -38.1227,34.1797,0. -46.9297,33.483,0. -52.142,33.0321,0. -58.4927,32.4779,0. -64.4838,32.0727,0. -70.0556,31.8694,0. -75.3278,31.8694,0. -78.6829,31.4614,0. -81.3789,31.4103,0. ")
    route2 = monthly_opt.coordinates_processing(
        "-69.4565,20.7236,0. -68.3182,21.8401,0. -67.4195,23.4436,0. -66.0415,25.2989,0. -64.7235,26.7525,0. -63.5851,28.2932,0. -61.6081,29.7599,0. -59.2715,31.2055,0. -56.5156,33.1325,0. -54.4786,34.2788,0. -52.7411,35.0672,0. -50.2248,36.1388,0. -47.5288,37.196,0. -45.4319,37.8139,0. -42.1368,39.2201,0. -38.6619,40.2795,0. -36.8046,41.0972,0. -33.5095,42.2162,0. -30.7536,43.0974,0. -28.5368,43.9229,0. -25.721,44.7372,0. -23.3844,45.624,0. -20.928,46.4144,0. -17.6928,47.1936,0. -14.877,47.7202,0. -12.5404,48.0417,0. -9.84439,48.7181,0. -6.90872,49.1902,0. -3.7334,49.89,0. -1.39685,50.7696,0.")
    route3 = monthly_opt.coordinates_processing(
        "-37.7862,52.9074,0. -39.8625,52.1032,0. -40.7119,51.4021,0. -41.8444,50.1489,0. -42.3163,49.0479,0. -43.0713,47.8587,0. -43.732,46.6415,0. -44.2983,45.2637,0. -44.6758,44.1232,0. -44.7702,43.2359,0. -44.7702,41.9857,0. -45.0533,40.7104,0. -45.0533,39.8464,0. -43.3545,39.7014,0. -42.5995,40.3518,0. -41.9388,41.3512,0. -41.2782,42.8911,0. -40.9007,44.1232,0. -39.8625,45.6609,0. -39.2018,46.6415,0. -38.6356,47.4135,0. -38.0693,48.0483,0. -37.503,48.862,0. -36.8424,49.1714,0. -36.0874,49.9064,0. -35.4267,50.4503,0. -34.1054,50.9881,0. -32.4066,51.9871,0. -31.2741,52.5071,0. -29.9528,53.304,0. -27.9709,54.1416,0. -26.272,55.1788,0. -25.517,55.6076,0. -24.5732,56.6074,0. -24.3845,57.2767,0. -24.3845,57.8838,0. -24.2901,58.4316,0. -26.1777,59.4059,0. -28.3484,59.4539,0. -30.3303,59.1165,0. -31.4628,58.3326,0. -32.5954,57.7834,0. -33.256,57.1233,0. -33.8223,56.2945,0. -34.4829,55.3401,0. -34.8605,54.9084,0. -35.238,54.6908,0. -36.5593,53.8643,0. -38.3524,53.5851,0.")


    s = Simulation('2014-12-01', route2, 3)
    s.sim_wind(3)
    s.sim_solar(0, 0, 2, 100)

