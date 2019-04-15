import gpxpy
import numpy as np


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
    route[:, [0, 1]] = route[:, [1, 0]]
    # OpenCPN XML copy gives (lon, lat, z) need convert to [lat, lon]
    # Numpy advanced slicing used here
    # ref: https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    return route

