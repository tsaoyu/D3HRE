__author__ = 'Yu Cao'

from D3HRE.core.hotel_load_model import HotelLoad
from D3HRE.core.navigation_utility import ocean_current_processing
from D3HRE.core.get_hash import hash_value
from D3HRE.core.mission_utility import get_mission

import pandas as pd

class Robot():
    def __init__(self):
        pass

    def __repr__(self):
        return "I am a dummy robot that don't know what to do."

    def prop_load(self):
        pass

    def hotel_load(self):
        pass

    def estimate_demand_load(self, mission):
        pass


class MaritimeRobot(Robot):

    def __init__(self, power_consumption_list, from_pyresis=None, use_ocean_current=False, config={}):
        self.power_consumption_list = power_consumption_list
        if from_pyresis is not None:
            self.set_from_PyResis(from_pyresis)
        self.use_ocean_current = use_ocean_current

        if config != {}:
            self.critical_prop_load_ratio = config['simulation']['critical_prop_load_ratio']
        else:
            self.critical_prop_load_ratio = 1

    def __repr__(self):
        return "I am a maritime robot. "

    def set_from_PyResis(self, vehicle):
        self.vehicle = vehicle
        self.surface_area = vehicle.maximum_deck_area()
        self.beam = vehicle.beam
        self.displacement = vehicle.displacement

    def estimate_demand_load(self, mission):
        self.mission = mission

        if self.use_ocean_current:
            self.ocean_current_df = ocean_current_processing(self.mission.df)
            self.get_propulsion_load(current=True)
        else:
            self.get_propulsion_load(current=False)
        self.get_hotel_load()
        self.critical_prop_load = self.critical_prop_load_ratio * self.prop_load

    def get_ocean_current(self):
        """
        Get ocean current for maritime vehicles.
        :return: None
        """
        self.ocean_current_df = ocean_current_processing(self.mission.df)


    def get_hotel_load(self, strategy='normal'):
        """
        Get hotel load of the vehicles.
        :param strategy: str optinal 'full-power' for continuous power output
        'normal' for duty cycle controlled power consumption generation
        :return: hotel_load dataFrame
        """
        hotel = HotelLoad(self.mission, self.power_consumption_list, strategy)
        self.hotel_load, self.critical_hotel_load = hotel.generate_power_consumption_timeseries()
        return self.hotel_load

    def get_propulsion_load(self, current=True):
        """
        :param current: bool default True, use ocean current for the vehicle
        :return: prop_load dataFrame
        """
        if current == False:
            self.prop_load = self.vehicle.prop_power()
        else:
            try:
                self.get_ocean_current()
            except FileNotFoundError:
                print('No ocean current file found!')
                self.prop_load = self.vehicle.prop_power()
                return self.prop_load
            if self.ocean_current_df.Vs.isnull().values.any():
                print('No current data from database, fallback on no current')
                self.prop_load = self.vehicle.prop_power()
            else:
                prop_power_list = []
                for v in self.ocean_current_df.Vs:
                    self.vehicle.speed = v
                    prop_power_list.append(self.vehicle.prop_power())

                self.prop_load = pd.Series(
                    index=self.mission.df.index, data=prop_power_list
                )
        return self.prop_load

class Task:
    """
    Task is a high level object consist of mission object, vehicle object and
    power consumption list.
    """
    def __init__(self, mission, robot, power_consumption_list={}):
        """

        :param mission: object should have df attribute that contains information on
        :param vehicle: object should have prop_power() method that return
        propulsion power of the vehicle
        :param power_consumption_list: dictionary consist of components with power
        requirements and duty cycle
        """
        self.mission = mission
        self.robot = robot
        self.power_consumption_list = power_consumption_list
        self.estimate_demand_load()


    def estimate_demand_load(self):
        self.robot.estimate_demand_load(self.mission)
        self.hotel_load = self.robot.hotel_load
        self.prop_load = self.robot.prop_load
        self.critical_hotel_load = self.robot.critical_hotel_load
        self.critical_prop_load = self.robot.critical_prop_load

        self.load_demand = self.hotel_load + self.prop_load


class Mission:
    def __init__(self, start_time, route, speed):
        self.start_time = start_time
        self.route = route
        self.speed = speed
        self.df = get_mission(self.start_time, self.route, self.speed)
        self.get_ID()

    def __str__(self):
        return "This mission {ID} is start from {a} at {b} UTC.".format(
            a=self.route[0], b=self.start_time, ID=self.ID
        )

    def get_ID(self):
        route_tuple = tuple(self.route.flatten().tolist())
        if isinstance(self.speed, list):
            speed_tuple = tuple(self.speed)
        else:
            speed_tuple = self.speed

        ID_tuple = (self.start_time, route_tuple, speed_tuple)
        self.ID = hash_value(ID_tuple)
        return self.ID




class Task:
    """
    Task is a high level object that contains the mission and the robot object.
    """
    def __init__(self, mission, robot):
        """

        :param mission: Object mission object
        :param robot: Object robot object
        :param power_consumption_list: optional the list of demand load, only use in legacy mode
        """
        self.mission = mission
        self.robot = robot
        self.estimate_demand_load()


    def estimate_demand_load(self):
        self.hotel_load = self.robot.hotel_load
        self.prop_load = self.robot.prop_load
        self.critical_hotel_load = self.robot.critical_hotel_load
        self.critical_prop_load = self.robot.critical_prop_load
        self.load_demand = self.hotel_load + self.prop_load



def setup():
    from distutils.util import strtobool
    print("This is an interactive setup tool for Dynamic Data Driven Hybrid Renewable Energy (D3HRE).")
    print("I will help you to configuration your system for the future use. ")
    print("================================================================ ")
    import os


    def ask_data_directory(download=True):

        data_directory = input("Where do you want to store you resource file? Enter to use current folder.")
        with open(os.path.expanduser('~/.d3hre'), "a") as f:
            if data_directory == '':
                f.write("[MERRA2]\nDatadir = {}\n".format(os.getcwd()))
            else:
                f.write("[MERRA2]\nDatadir = {}\n".format(data_directory))
        if download == True:
            connections = input("How many connections you want to download at the same time. "
                                "Too much connections may cause ban on IP. Enter to use default value 8.")
            with open(os.path.expanduser('~/.d3hre'), "a") as f:
                if connections == '':
                    f.write("Connections = 8\n")
                else:
                    f.write("Connections = {}\n".format(connections))


    def ask_need_store_password():

        need_store_password = input("Do you want D3HRE store your credential for you? (Y/N)")
        while strtobool(need_download) not in [True, False]:
            need_store_password = input("Please enter 'y/n' for either yes or no")

        if strtobool(need_store_password):
            import getpass
            account = input("Please enter your account:")
            password = getpass.getpass("Please enter your password:")
            with open(os.path.expanduser('~/.d3hre'), "a") as f:
                f.write("Username = {}\nPassword = {}\n".format(account, password))

        elif strtobool(need_store_password) == False:
            print(
                "Don't worry. You can still store your credential as environment variables or enter them when necessary.")

    need_download = input("Do you want D3HRE download the weather reanalysis data for the simulation ? (Y/N)")
    while strtobool(need_download) not in [True, False]:
        need_download = input("Please enter 'y/n' for either yes or no")

    if strtobool(need_download):
        ask_data_directory(download=True)
        ask_need_store_password()

    elif strtobool(need_download) == False:
        print("Please refer to the wiki page for the data preparation.")
        ask_data_directory(download=False)


    print("Setup is done. Enjoy your time with D3HRE!")
    print("================================================================ ")

