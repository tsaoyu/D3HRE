def power_from_turbine(wind_speed, area, power_coefficient, cut_in_speed, cut_off_speed):
    Cp = power_coefficient
    A = area
    power = 0
    v = wind_speed
    if v < cut_in_speed:
        power = 0
    elif cut_in_speed < v < cut_off_speed:
        power = 1 / 2 * Cp * A * v ** 3
    elif cut_off_speed < v < 3 * cut_off_speed:
        power = 1 / 2 * Cp * A * cut_off_speed ** 3
    elif v > 3 * cut_off_speed:
        power = 0

    return power