import numpy as np
import pandas as pd


class HotelLoad:
    """
    A simple class to generate hotel load as a list (or time series).

    """

    def __init__(self, mission, power_consumption_list, strategy='normal'):
        """


        :param mission: mission object including df (DataFrame with time index)
        :param strategy: string, hotel load planning mode, currently support normal
        and full-power mode.
        """
        self.strategy = strategy
        self.mission = mission

        power_consumption_list = power_consumption_list.copy()
        try:
            np.random.seed(power_consumption_list['seed'])
            del power_consumption_list['seed']
        except KeyError:
            np.random.seed(123)

        self.power_consumption_list = power_consumption_list
        self.components = list(self.power_consumption_list.keys())

    def component_power_consumption(self, component, performance_level):
        """
        Generate power comcumption for one component at given performance level.

        :param component: key,
        :param performance_level:
        :return:
        """
        power_consumption = self.power_consumption_list[component]

        if len(power_consumption['power']) == 1:
            power = power_consumption['power'][0] * performance_level
            return power
        elif len(power_consumption['power']) == 2:
            power = (
                power_consumption['power'][1] - power_consumption['power'][0]
            ) * performance_level + power_consumption['power'][0]
            return power
        else:
            print('Power list could only be range based or fixed. {} was given'.format(power_consumption['power']))


    def generate_power_consumption(self):

        non_critical_hotel_load = 0
        critical_hotel_load_consumption = 0


        if self.strategy == 'full-power':
            performance = 1
            for component in self.components:
                critical_hotel_load_consumption += self.component_power_consumption(
                    component, performance
                )

        elif self.strategy == 'normal':

            for component in self.components:
                if self.power_consumption_list[component]['duty_cycle'] == 1:
                    critical_hotel_load_consumption += self.component_power_consumption(component, 1)
                else:
                    duty_cycle = self.power_consumption_list[component]['duty_cycle']
                    performance = np.clip(duty_cycle + np.random.randn() * 0.1, 0, 1)
                    non_critical_hotel_load += self.component_power_consumption(component, performance)

        else:
            print('This is not supported yet!')

        return non_critical_hotel_load + critical_hotel_load_consumption, critical_hotel_load_consumption

    def generate_power_consumption_timeseries(self):
        duration = len(self.mission.df.index)
        power_consumption_list, critical_hotel_load_list = np.array([
            self.generate_power_consumption() for _ in range(int(duration))
        ]).T
        hotel_load_ts = pd.Series(
            data=power_consumption_list, index=self.mission.df.index
        )
        critical_hotel_load_ts = pd.Series(
            data=critical_hotel_load_list, index=self.mission.df.index
        )
        return hotel_load_ts, critical_hotel_load_ts


if __name__ == '__main__':

    from D3HRE.core.mission_utility import Mission
    import numpy as np

    test_route = np.array([[10.69358, -178.94713892], [11.06430687, +176.90022735]])
    test_mission = Mission('2014-01-01', test_route, 2)

    power_consumption_list = {
        'single_board_computer': {'power': [2, 10], 'duty_cycle': 0.5},
        'webcam': {'power': [0.4, 0.6], 'duty_cycle': 1},
        'gps': {'power': [0.04, 0.4], 'duty_cycle': 0.9},
        'imu': {'power': [0.67, 1.1], 'duty_cycle': 0.9},
        'sonar': {'power': [0.5, 50], 'duty_cycle': 0.5},
        'ph_sensor': {'power': [0.08, 0.1], 'duty_cycle': 0.95},
        'temp_sensor': {'power': [0.04], 'duty_cycle': 1},
        'wind_sensor': {'power': [0.67, 1.1], 'duty_cycle': 0.5},
        'servo_motors': {'power': [0.4, 1.35], 'duty_cycle': 0.5},
        'radio_transmitter': {'power': [0.5, 20], 'duty_cycle': 0.2},
    }
    h_normal = HotelLoad(test_mission, power_consumption_list)
    h_full = HotelLoad(test_mission, power_consumption_list, 'full-power')

    print(h_normal.generate_power_consumption_timeseries())
    h_full_list = h_full.generate_power_consumption_timeseries()
