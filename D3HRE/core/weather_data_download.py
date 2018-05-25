import glob
import os.path
import numpy as np
import pandas as pd
import xarray as xr


from data_config import *
from opendap_download.multi_processing_download import DownloadManager
from D3HRE.core.weather_data_processing import resource_df_processing




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


def download_URL(mission, data_set='solar', debug=False):
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
    if debug:
        return download_index.index[0], download_index.index[-1]
    else:
        generated_URLs = []
        for start, end, lat_lon in zip(download_index.index[:-1],
                                     download_index.index[1:],
                                     download_index[1:]):
            generated_URLs.append(generate_single_download_link(start, end, lat_lon, data_set))
        return generated_URLs


def resource_df_download(mission, username=USERNAME, password=PASSWORD, n=NUMBER_OF_CONNECTIONS, data_dir=DATA_DIR):
    """
    Resource dataFrame download function.


    :type mission: object
    :param mission: Mission object
    :param username: username of NASA earthdata portal
    :param password: password of NASA earthdata portal
    :param n: number of concurrent multiprocess download (adjust the number to avoid been banned)
    :return: raw resource dataFrame, time-indexed Pandas dataFrame including all requested field
    """
    mission_df = mission.df
    ID = mission.ID

    folder = os.path.expanduser(data_dir + ID)
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
        download_manager.download_urls = download_URL(mission_df, data_set='wind')
        if not os.path.exists(download_manager.download_path):
            print('Wind data not found, automatic download starting ...')
            download_manager.start_download(n)

        download_manager.download_path = folder + '/download_solar'
        download_manager.download_urls = download_URL(mission_df, data_set='solar')
        if not os.path.exists(download_manager.download_path):
            print('Solar data not found, automatic download starting ...')
            download_manager.start_download(n)


        # Process when necessary
        if not os.path.isfile(file_name):
            print('Compact data not found, automatic processing starting ...')
            wind_files = sorted(glob.glob(folder + '/download_wind/MERRA*'))
            # Sort file to make sure time is aligned
            file_size = np.array([os.path.getsize(wind_file) for wind_file in wind_files])
            corrupted_files = (file_size < 60000).sum()
            if corrupted_files == 0:
                print("Wind data file completeness check pass")
            else:
                print("Some wind data corrupted, redownload start")
                download_manager.download_path = folder + '/download_wind'
                URLS = download_URL(mission_df, data_set='wind')
                url = []
                for url_index in np.where(file_size<60000)[0].tolist():
                    url.append(URLS[url_index])
                download_manager.download_urls = url
                download_manager.start_download(n)
            wind_xdata = xr.open_mfdataset(wind_files, concat_dim='time', autoclose=True)
            # In case of complain on 'Too many files opened', switch autoclose to True,
            # Data processing will be slower with autoclose option on.
            wind_df = wind_xdata.to_dataframe()

            solar_files = sorted(glob.glob(folder + '/download_solar/MERRA*'))
            file_size = np.array([os.path.getsize(solar_file) for solar_file in solar_files])
            corrupted_files = (file_size < 20000).sum()
            if corrupted_files == 0:
                print("Solar data file completeness check pass")
            else:
                print("Some solar data corrupted, redownload start")
                download_manager.download_path = folder + '/download_solar'
                URLS = download_URL(mission_df, data_set='solar')
                url = []
                for url_index in np.where(file_size<20000)[0].tolist():
                    url.append(URLS[url_index])
                download_manager.download_urls = url
                download_manager.start_download(n)
            solar_xdata = xr.open_mfdataset(solar_files, concat_dim='time', autoclose=True)
            # In case of complain on 'Too many files opened', switch autoclose to True,
            # Data processing will be slower with autoclose option on.
            solar_df = solar_xdata.to_dataframe()

            resource_df = pd.concat([solar_df, wind_df], axis=1)
            resource_df.reset_index(drop=True, inplace=True)
            resource_df['utc'] = mission_df.index[1:]

            resource_df.set_index('utc', inplace=True)

            resource_df.to_pickle(file_name)

        resource_df = pd.read_pickle(file_name)
    return resource_df



def resource_df_download_and_process(mission):
    """
    Process downloaded MEERA-2 dataFrame.

    :param mission: Mission object
    :return:  time indexed Pandas dataFrame with additional field in temperature (degree C),
        kt(clearness index), V2 (wind speed at 2 metres height), true_wind_direction (degrees),
        heading (degrees), Va (apparent wind speed), apparent_wind_direction(degrees)
    """
    resource_df = resource_df_download(mission)
    combined_df = pd.concat([mission.df, resource_df], axis=1).bfill()
    # combine mission dataFrame and weather data (resource) dataFrame into a single one
    processed_resource_df = resource_df_processing(combined_df)
    return processed_resource_df


if __name__ == '__main__':
    pass