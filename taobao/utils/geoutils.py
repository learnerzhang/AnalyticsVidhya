#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/09/2017 8:14 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : geoutils.py
# @Software: PyCharm
import json


def get_geo(city_map):
    shops = []
    lat = []
    lon = []
    with open("../cityGeo.json", 'r') as geo:
        geoJson = json.load(geo)
        # pprint(geoJson)
        for key in geoJson.keys():
            for unit in geoJson[key]:
                k = unit['city']
                v = unit['geo']
                # print k, v[0], v[1]
                for c, s in city_map.items():
                    if k.startswith(c):
                        shops.append(len(s))
                        lat.append(v[1])
                        lon.append(v[0])
    return shops, lon, lat


def get_default_geo():
    loc = []
    lat = []
    lon = []
    with open("../cityGeo.json", 'r') as geo:
        geoJson = json.load(geo)
        # pprint(geoJson)
        for key in geoJson.keys():
            for unit in geoJson[key]:
                k = unit['city']
                v = unit['geo']
                # print k, v[0], v[1]
                loc.append(k)
                lat.append(v[1])
                lon.append(v[0])
    return loc, lon, lat
