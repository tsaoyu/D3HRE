import cartopy

import numpy as np

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pylab as plt

def plot_route(route, save=False):
    """
    Plot route on global map
    :param route: np ndarry (n,2) [lat, lon] way points
    :param save: optional save the figure or not
    :return: none
    """
    ax1 = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax1.set_global()
    ax1.coastlines()
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.add_feature(cartopy.feature.LAND)
    ax1.add_feature(cartopy.feature.OCEAN)
    ax1.add_feature(cartopy.feature.COASTLINE)
    ax1.plot(route[:,1], route[:,0], 'g')
    plt.show()
    if save:
        plt.savefig('../data/Figures/route' + str(route[0, 0]) + '.pdf', format='pdf')


def plot_solar_map(resource_df):
    pass

