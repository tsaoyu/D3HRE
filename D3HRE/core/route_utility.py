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
    return route

def read_route_from_gpx(file):
    """
    Read route from gpx file

    :param file: str, path to the .gpx file
    :return: list, all routes
    """
    gpx_file = open(file)
    gpx = gpxpy.parse(gpx_file)
    all_routes = []
    for route in gpx.routes:
        route_list = []
        for point in route.points:
            route_list.append([point.latitude, point.longitude])
        all_routes.append(route_list)
    return all_routes