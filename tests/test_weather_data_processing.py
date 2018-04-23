import pytest
import pandas as pd
from D3HRE.core.weather_data_processing import *


test_df = pd.DataFrame([[280, 10, 10, 2, 8, 8, 600, 1000],
                        [285, 10, 10, 2, 9, 9, 600, 1000],
                        [290, 10, 10, 2, 10, 9, 600, 1000],
                        [290, 10, -10, 2, 10, 9, 600, 1000],
                        [290, 0, -10, 2, 10, 9, 600, 1000],
                        [290, -10, -10, 2, 10, 9, 600, 1000],
                        [290, -10, 0, 2, 10, 9, 600, 1000],
                        [290, -10, 10, 2, 10, 9, 600, 1000]],
                       columns=['T2M', 'U2M', 'V2M', 'speed', 'lat', 'lon', 'SWGDN', 'SWTDN'])

resource_df = resource_df_processing(test_df)



def test_temperature_processing():
    assert resource_df.temperature[0] == pytest.approx(6.85)
    assert resource_df.temperature[1] == pytest.approx(11.85)
    assert resource_df.temperature[2] == pytest.approx(16.85)


def test_true_wind_direction():
    assert resource_df.true_wind_direction[0] == 45   # East U= 10, North V= 10
    assert resource_df.true_wind_direction[3] == 135  # East U= 10, North V=-10
    assert resource_df.true_wind_direction[4] == 180  # East U=  0, North V=-10
    assert resource_df.true_wind_direction[5] == 225  # East U=-10, North V=-10
    assert resource_df.true_wind_direction[6] == 270  # East U= 10, North V=  0
    assert resource_df.true_wind_direction[7] == 315  # East U=-10, North V= 10

