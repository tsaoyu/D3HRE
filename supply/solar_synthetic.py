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



def get_syntheic_solar_df(start_time, route, speed):
    return synthetic_radiation(get_position_df(start_time, route, speed))

