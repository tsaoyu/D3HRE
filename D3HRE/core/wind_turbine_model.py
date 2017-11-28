import numpy as np


def power_from_turbine(wind_speed, area, power_coefficient, cut_in_speed, rated_speed):
    """
    Wind turbine power estimation function. Power production from wind turbine is idealised
    and modeled into cut in, rated and cut off speed. When wind speed is lower than cut in
    speed, no power production from wind turbine. AS wind speed increased, power production
    from wind turbine follows cubic rule. Cut off speed is set as 3 times of rated speed,
    before wind speed reach cut-off speed. Wind turbine is assumed to be working under controller
    speed mode that turbine blade rotate as same as rated speed. When speed is higher than
    rated speed, wind turbine will stop operating. Maximum power coefficient tracking is assumed
    in the idealised turbine model.

    :param wind_speed: float m/s wind speed at turbine height
    :param area: float m^2 swept area of wind turbine
    :param power_coefficient: dimensionless power coefficient of wind turbine
    :param cut_in_speed: m/s minimum speed that turbine will generate power
    :param rated_speed: m/s rated speed of wind turbine
    :return: Watts power of wind turbine generation
    """
    Cp = power_coefficient
    A = area
    power = 0
    v = wind_speed
    if v < cut_in_speed:
        power = 0
    elif cut_in_speed < v < rated_speed:
        power = 1 / 2 * Cp * A * v ** 3
    elif rated_speed < v < 3 * rated_speed:
        power = 1 / 2 * Cp * A * rated_speed ** 3
    elif v > 3 * rated_speed:
        power = 0

    return power


def resistance_power(dataframe, area):
    """
    Wind turbine resistance power estimation. When wind blows into wind turbine, it not only
    generate circular moment that can drive electrical power but also axial force. This function
    calculate power consumption due to additional resistance power for wind turbine on platforms.


    :param dataframe: pandas dataframe contains following fields: V2(wind speed at two metres height)
        apparent_wind_direction (apparent wind direction ), speed(speed of platform)
    :param area: m^2 area of wind turbine
    :return: Watts power correction positive for propulsion power contribution
        negative for propulsion power increase due to resistance
    """
    A = area
    power_correction = 1 / 2 * A * 0.6 * dataframe.Va ** 2 * dataframe.relative_wind_cos* dataframe.speed
    return power_correction
