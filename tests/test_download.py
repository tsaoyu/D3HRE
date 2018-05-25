# Test couldn't eliminate all errors in the program automatically,
# but it give programmer more confidence that it can make sure some functions works as expected.

import pytest

from tests.test_env import *
from D3HRE.core.weather_data_download import resource_df_download



def test_resource_df_download():
    resource_df_download(test_mission)
    pass

