#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/09/2017 6:46 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : geo.py
# @Software: PyCharm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from taobao.condom.condom import get_city_map
from taobao.utils.geoutils import get_geo

map = Basemap(projection='stere',
              lat_0=35, lon_0=110,
              llcrnrlon=82.33,
              llcrnrlat=3.01,
              urcrnrlon=138.16,
              urcrnrlat=53.123,
              resolution='l',
              area_thresh=10000,
              rsphere=6371200.)

map.readshapefile("../CHN_adm_shp/CHN_adm1", 'states', drawbounds=True)
map.drawcountries()
map.drawcounties()

# draw coastlines.
map.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
# map.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
# map.fillcontinents(color='coral', lake_color='aqua')

# get city shops map
city_map = get_city_map()
nums, lon, lat = get_geo(city_map)
x, y = map(lon, lat)
map.scatter(x, y, s=nums)
for i, j, n in zip(x, y, nums):
    if n > 40:
        plt.text(i, j+50.0, n, fontsize=10, rotation='horizontal', color='red')
    elif n > 20:
        plt.text(i, j + 30.0, n, fontsize=10, rotation='horizontal', color='blue')
    else:
        plt.text(i, j + 10.0, n, fontsize=10, rotation='horizontal', color='green')

parallels = np.arange(0, 90, 10)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)  # 绘制纬线
meridians = np.arange(80, 140, 10)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)  # 绘制经线
plt.title("The Seller Distribution in China")
plt.show()
