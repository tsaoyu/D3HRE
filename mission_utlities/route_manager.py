import numpy as np


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



