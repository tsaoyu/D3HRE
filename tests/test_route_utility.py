import os
import pytest
from D3HRE.core.route_utility import *
from D3HRE.core.file_reading_utility import read_route_from_gpx

opencpn_XML='-69.4565,20.7236,0. -68.3182,21.8401,0. '

def test_coordinates_processing():
    route = opencpn_coordinates_processing(opencpn_XML)
    assert route.shape[1] == 2 # return a (n, 2) np array
    assert route[0][0] == 20.7236


def test_gpx_route_reader():
    file_name = os.path.join(os.path.dirname(__file__), 'test.gpx')
    routes = read_route_from_gpx(file_name)
    assert len(routes[0]) == 30  # GPX reader can get right number of way points
    assert routes[0][0][0] == pytest.approx(20.7236)
    # element in first column first row is latitude

