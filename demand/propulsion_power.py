import numpy as np
from scipy import interpolate


def frictional_resistance_coef(length, speed, **kwargs):
    Cf = 0.075 / (np.log10(reynolds_number(length, speed, **kwargs)) - 2) ** 2
    return Cf


def reynolds_number(length, speed, temperature=25):
    kinematic_viscosity = interpolate.interp1d([0, 10, 20, 25, 30, 40],
                                               np.array([18.54, 13.60, 10.50, 9.37, 8.42, 6.95]) / 10 ** 7)
    # Data from http://web.mit.edu/seawater/2017_MIT_Seawater_Property_Tables_r2.pdf
    Re = length * speed / kinematic_viscosity(temperature)
    return Re


def froude_number(speed, length):
    g = 9.80665  # conventional standard value m/s^2
    Fr = speed / np.sqrt(g * length)
    return Fr


cr_list = np.loadtxt('./demand/cr.txt')
cr_points = cr_list.T[:3].T
cr_values = cr_list.T[3].T / 1000
cr = interpolate.LinearNDInterpolator(cr_points, cr_values)


def residual_resistance_coef(slenderness, prismatic_coef, froude_number):
    Cr = cr(slenderness, prismatic_coef, froude_number)
    return Cr


class Ship():
    def __init__(self):
        self.total_resistance_coef = 0

    def dimension(self, length, draught, beam, speed,
                 slenderness_coefficient, prismatic_coefficient):
        self.length = length
        self.draught = draught
        self.beam = beam
        self.speed = speed
        self.slenderness_coefficient = slenderness_coefficient
        self.prismatic_coefficient = prismatic_coefficient
        self.displacement = (self.length / self.slenderness_coefficient) ** 3
        self.surface_area = 1.025 * (1.7 * self.length * self.draught +
                                     self.displacement / self.draught)

    def resistance(self):
        self.total_resistance_coef = frictional_resistance_coef(self.length, self.speed) + \
                                residual_resistance_coef(self.slenderness_coefficient,
                                                         self.prismatic_coefficient,
                                                         froude_number(self.speed, self.length))
        RT = 1 / 2 * self.total_resistance_coef * 1025 * self.surface_area * self.speed ** 2
        return RT

    def maximum_deck_area(self, water_plane_coef=0.88):
        AD = self.beam * self.length * water_plane_coef
        return AD

    def get_reynold_number(self):
        return reynolds_number(self.length, self.speed)

    def prop_power(self, propulsion_eff=0.7, sea_margin=0.2):
        PP = (1 + sea_margin) * self.resistance() * self.speed/propulsion_eff
        return PP
