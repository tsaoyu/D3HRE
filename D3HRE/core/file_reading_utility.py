import gpxpy

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
        route_list =[]
        for point in route.points:
            route_list.append([point.latitude, point.longitude])
        all_routes.append(route_list)
    return all_routes

