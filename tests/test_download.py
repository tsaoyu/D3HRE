# Test couldn't eliminate all errors in the program automatically,
# but it give programmer more confidence that it can make sure some functions works as expected.

import pytest
import numpy as np
import os, sys
from passwords import *

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))



from D3HRE.core.mission_utility import get_mission
from D3HRE.core.weather_data_download import download_URL, resource_df_download


# Make sure that generated link matches the total length of original mission
test_route =  np.array([[  10.69358 ,  -178.94713892], [  11.06430687, +176.90022735]])
test_mission = get_mission('2014-01-01', test_route, 2)


def test_duration():
    start, end = download_URL(test_mission, debug=True)
    assert end - start == test_mission.index[-1] - test_mission.index[0]
    pass

def test_resource_df_download():
    resource_df_download(test_mission, username=USERNAME, password=PASSWORD, n=1)
    pass

