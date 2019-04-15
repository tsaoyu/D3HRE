import math
import numpy as np
import xarray as xr
import os.path
import configparser


config = configparser.ConfigParser()
config_file_path = os.path.expanduser('~/.d3hre')
config.read(config_file_path)

OSCAR_DIR = config['OSCAR']['Datadir']


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculate the compass bearing of the mvoing platform on point A and B.

    :param pointA: tuple or list in the format of [lat, lon]
    :param pointB: tuple or list in the format of [lat, lon]
    :return: float degrees compass bearing between A and B
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
    )
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def ocean_current_processing(mission_df, file_dir=OSCAR_DIR):
    """

    Ocean data processing utility.


    :param mission_df: padas dataFrame, contains UTC indexed sptial-temporal information
    :param file_dir: str, direcotry of the OSCAR ocean current file
    :return: pandas dataFrame, unit in m/s the speed of the ocean current and the moving platform
    """
    dataframe = mission_df.copy()
    location_list = [tuple(x) for x in dataframe[['lat', 'lon']].values]

    # Calculate heading with initial compass bearing
    heading = []
    for a, b in zip(location_list[:-1], location_list[1:]):
        heading.append(calculate_initial_compass_bearing(a, b))

    heading.append(heading[-1])
    # As it reach the final way point heading stay unchanged

    dataframe['heading'] = heading

    V_g = dataframe['speed'] / 3.6  # ship ground speed in mission DataFrame unit of km/h

    u_g = V_g * np.sin(np.radians(dataframe['heading']))
    v_g = V_g * np.cos(np.radians(dataframe['heading']))

    def read_and_match_one_year_current(year, df, file_dir):
        dataframe = df[df.index.year == year]
        dataset_file_dir = os.path.expanduser(file_dir + 'oscar_vel{}.nc'.format(year))
        dataset = xr.open_dataset(dataset_file_dir)
        time_index, lat_index, lon_index = (
            dataframe.index,
            dataframe.lat.tolist(),
            (dataframe.lon + 200).tolist(),
        )

        u_speed_list = []
        v_speed_list = []
        for time, lat, lon in zip(time_index, lat_index, lon_index):
            selected_data = dataset.sel(
                time=time, latitude=lat, longitude=lon, depth=15, method='nearest'
            )
            u_speed = selected_data.u.values.tolist()
            v_speed = selected_data.v.values.tolist()
            u_speed_list.append(u_speed)
            v_speed_list.append(v_speed)

        return u_speed_list, v_speed_list

    start_year = dataframe.index[0].year
    end_year = dataframe.index[-1].year

    if start_year != end_year:
        year_range = range(start_year, end_year + 1)
        u_speed_list = []
        v_speed_list = []
        for year in year_range:
            u_speed_one_year, v_speed_one_year = read_and_match_one_year_current(
                year, dataframe, file_dir
            )
            u_speed_list += u_speed_one_year
            v_speed_list += v_speed_one_year

    else:
        u_speed_list, v_speed_list = read_and_match_one_year_current(
            start_year, dataframe, file_dir
        )

    dataframe['current_u'] = u_speed_list
    dataframe['current_v'] = v_speed_list

    dataframe['current_u'].fillna(dataframe['current_u'].mean(), inplace=True)
    dataframe['current_v'].fillna(dataframe['current_v'].mean(), inplace=True)

    u_s = u_g - dataframe['current_u']
    v_s = v_g - dataframe['current_v']

    dataframe['Vs'] = np.sqrt(u_s ** 2 + v_s ** 2)

    return dataframe
