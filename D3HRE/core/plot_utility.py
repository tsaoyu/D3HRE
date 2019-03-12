import cartopy
import os, sys
import numpy as np

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import PreText


def plot_route(route, save=False, save_name=None):
    """
    Plot route on global map
    :param route: np ndarry (n,2) [lat, lon] way points
    :param save: optional save the figure or not
    :return: none
    """
    ax1 = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax1.set_global()
    ax1.stock_img()
    ax1.coastlines()
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.add_feature(cartopy.feature.LAND)
    ax1.add_feature(cartopy.feature.OCEAN)
    ax1.add_feature(cartopy.feature.COASTLINE, edgecolor='gray')
    ax1.add_feature(cartopy.feature.BORDERS, edgecolor='gray')
    ax1.text(route[0][1], route[0][0]-10, 'Start',transform=ccrs.Geodetic())
    ax1.text(route[-1][1], route[-1][0]-10, 'End',transform=ccrs.Geodetic())
    ax1.plot(route[:,1], route[:,0], 'g',transform=ccrs.Geodetic())
    if save:
        plt.savefig('./route' + str(route[0, 0]) + '.pdf', format='pdf')
    if save_name:
        plt.savefig(save_name)
    plt.show()
    return ax1



def wind_dashboard(history):
    d = os.path.dirname(sys.modules["D3HRE.core"].__file__)
    img_dir = os.path.join(d, './map_img.npz')
    img = np.load(img_dir)
    img = img.f.arr_0
    TOOLS = "box_select,lasso_select,save,pan,wheel_zoom,reset"
    source = ColumnDataSource(history)
    source_static = ColumnDataSource(history)
    p1 = figure(x_axis_type="datetime", title="Wind power in simulation", plot_width=900, plot_height=230, tools=TOOLS)
    p1.yaxis.axis_label = 'Power (W)'
    p1.line('index', 'wind_raw', source=source_static, color='#B2DF8A')
    p1.circle('index', 'wind_raw', color='#B2DF8A', source=source, size=1.5, selection_color="orange", alpha=0.6,
              nonselection_alpha=0.1, selection_alpha=0.4)

    p2 = figure(x_axis_type="datetime", title="Wind speed from weather reanalysis", plot_width=900, plot_height=230,
                tools=TOOLS)
    p2.x_range = p1.x_range
    p2.grid.grid_line_alpha = 0.3
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Wind speed (m/s)'
    p2.line('index', 'V2', source=source_static)
    p2.circle('index', 'V2', source=source, size=1.5, selection_color="orange", alpha=0.6, nonselection_alpha=0.1,
              selection_alpha=0.4)

    loc = figure(title="Position", plot_width=450, plot_height=300, x_range=[-180, 180], y_range=[-90, 90],
                 tools='save')
    loc.image_rgba(image=[img], x=-180, y=-90.17, dw=360, dh=181)
    loc.circle('lon', 'lat', source=source, alpha=0.5, size=2, selection_color="orange", nonselection_alpha=0.1,
               selection_alpha=0.4)

    p3 = figure(title="Wind vector", plot_width=300, plot_height=300)
    p3.circle('V2M', 'U2M', source=source, alpha=0.5, size=2, selection_color="orange", nonselection_alpha=0.1,
              selection_alpha=0.4)
    p3.yaxis.axis_label = 'North ward wind speed (m/s)'
    p3.xaxis.axis_label = 'East ward wind speed (m/s)'
    stats = PreText(text='', width=200)

    def update_stats(data):
        stats.text = str(data[['wind_power']].describe().round(2))

    def selection_change(attrname, old, new):
        data = source.data
        selected = source.selected['1d']['indices']
        if selected:
            data = data.iloc[selected, :]
        update_stats(data)

    def update(selected=None):
        data = history
        update_stats(data)

    source.on_change('selected', selection_change)
    update()

    wind_stats = row(p3, stats, loc)
    layout = column(p1, wind_stats, p2)
    show(layout)

def plot_supply_SOC_and_unmet(result, save_name=None, y_lim=None):
    fig, (ax, ax2) =  plt.subplots(2,1, figsize=(10,3), gridspec_kw={'height_ratios':[10,1]}, sharex=True)
    ax1 = ax.twinx()
    result.SOC.resample('24H').mean().plot(color='g', ax=ax1, label='SOC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('State of charge (SOC)')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    result.Unmet.plot(ax=ax2, color='r')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Unmet')
    (result.solar_power + result.wind_power).resample('24H').mean().plot(ax=ax, label='Generation')
    result.Generation.resample('24H').mean().plot(ax=ax, label='Generation')
    ax.legend(loc='upper left')
    ax.set_ylabel('Power generation $(W)$')
    if save_name is not None:
        fig.savefig(save_name)