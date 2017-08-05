import gpxpy

def read_route_from_gpx(file):
    """
    Read route from gpx file
    :param file: str path to the .gpx file
    :return: list contains all routes indexed by order of route first
     all_route[0] is the first route sorted as [lat, lon] with shape (n,2)
    """
    gpx_file = open(file)
    gpx = gpxpy.parse(gpx_file)
    all_route = []
    for route in gpx.routes:
        route_list =[]
        for point in route.points:
            route_list.append([point.latitude, point.longitude])
        all_route.append(route_list)
    return all_route