#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/09/2017 7:39 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import json
from pprint import pprint


with open("cityGeo.json", 'r') as geo:
    geoJson = json.load(geo)
    # pprint(geoJson)

    for key in geoJson.keys():
        for unit in geoJson[key]:
            k = unit['city']
            v = unit['geo']
            print k, v[0], v[1]
