import ephem
import math
import numpy as np
from D3HRE.core.navigation_utility import calculate_initial_compass_bearing


class SolarSystem():
    """
    A solar system that estimate the zenith angle for arbitary point on the earth.


    """
    def __init__(self):
        sun = ephem.Sun()
        self.sun = sun

    def set_date(self, date):
        self.obs.date = ephem.Date(date)

    def add_observer(self, lat, lon):
        obs = ephem.Observer()
        obs.lat = lat
        obs.lon = lon
        self.obs = obs
        self.lat, self.lon = lat, lon

    def move_observer(self, delta_lat, delta_lon):
        """

        :param delta_lat: difference in latitude
        :param delta_lon: difference in longitude
        :return:
        """
        self.obs.lat = (float(self.obs.lat) - delta_lat)
        self.obs.lon = (float(self.obs.lon) - delta_lon)
        # print(self.obs.lat, self.obs.lon)
        return [math.degrees(self.obs.lat), math.degrees(self.obs.lon)]

    def get_solar_angle(self):
        """

        :return: solar angle
        """

        self.sun.compute(self.obs)
        return [self.sun.az, self.sun.alt]

    def get_zenith_cosine(self):
        """

        :return: zenith angle
        """

        self.sun.compute(self.obs)
        return math.sin(self.sun.alt)



def calculate_zenith_cosine(time, lat, lon, min_zenith_cosine=0.065):
    """
    Wrapper on the solar system.

    :param time: UTC time
    :param lat: latitude
    :param lon: longitude
    :param min_zenith_cosine: minimal zenith angle
    :return: cosine on the zenith angle
    """
    solar_system = SolarSystem()
    solar_system.add_observer(str(lat), str(lon))
    solar_system.set_date(time)
    zenith_cosine = max(min_zenith_cosine,
                       solar_system.get_zenith_cosine())
    return zenith_cosine


def resource_df_processing(dataframe):
    """
    Process downloaded MEERA-2 dataFrame.

    :param dataframe: time indexed Pandas dataFrame contains field of lat, lon, speed,
        local_time, T2M, SWGDN, SWTDN, U2M, V2M
    :return:  time indexed Pandas dataFrame with additional field in temperature (degree C),
        kt(clearness index), V2 (wind speed at 2 metres height), true_wind_direction (degrees),
        heading (degrees), Va (apparent wind speed), apparent_wind_direction(degrees)
    """


    # Clearness index processing
    zenith_cosine_list = []

    for index, row in dataframe.iterrows():
        zenith_cosine = calculate_zenith_cosine(index, row.lat, row.lon)
        zenith_cosine_list.append(zenith_cosine)

    dataframe['zenith_cos'] = zenith_cosine_list
    dataframe['kt'] = dataframe.SWGDN / (dataframe.SWTDN * zenith_cosine_list)
    # Maximum clearness index for hourly data pvlib
    dataframe.loc[dataframe.kt > 0.82, 'kt'] = 0.82


    # Temperature processing
    dataframe['temperature'] = dataframe.T2M - 273.15


    # Wind speed at 2 metres height
    dataframe['V2'] = np.sqrt(dataframe.V2M ** 2 + dataframe.U2M ** 2)

    # Get true wind direction from north ward and east ward wind speed
    dataframe['true_wind_direction'] = (
        np.degrees(np.arctan2(dataframe['U2M'], dataframe['V2M'])) + 360
    ) % 360

    location_list = [tuple(x) for x in dataframe[['lat', 'lon']].values]

    # Calculate heading with initial compass bearing
    heading = []
    for a, b in zip(location_list[:-1], location_list[1:]):
        heading.append(calculate_initial_compass_bearing(a, b))

    heading.append(heading[-1])
    # As it reach the final way point heading stay unchanged

    dataframe['heading'] = heading
    # apparent wind                 V_{app} = [Uapp,   Vapp] :: Va
    # true wind at 2 metres height V_{true} = [U2M,     V2M] :: V2
    # platform speed vector          V_{sp} = [Up,       Vp] :: Vs
    #
    #                 V_{app} = V_{true} + (- V_{s})

    # Get apparent wind vector and scalar apparent wind speed and direction
    V_s = dataframe['speed'] / 3.6  # ship speed in DataFrame unit of km/h

    U_p = V_s * np.sin(np.radians(dataframe['heading']))
    V_p = V_s * np.cos(np.radians(dataframe['heading']))

    U_app = dataframe['U2M'] - U_p
    V_app = dataframe['V2M'] - V_p

    dataframe['Va'] = np.sqrt(U_app ** 2 + V_app ** 2)
    dataframe['apparent_wind_direction'] = (
        np.degrees(np.arctan2(U_app, V_app)) + 360
    ) % 360
    V_a = np.array([U_app, V_app]).T
    V_sp = np.array([U_p, V_p]).T

    wind_cos = []
    for U, V in zip(V_a, V_sp):
        wind_cos.append(np.dot(U, V) / np.linalg.norm(U) / np.linalg.norm(V))

    dataframe['relative_wind_cos'] = wind_cos

    return dataframe
