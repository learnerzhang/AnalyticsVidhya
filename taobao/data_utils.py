#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/09/2017 7:59 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : data_utils.py
# @Software: PyCharm
import pymongo
import json
conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database("flycup")
coll = db.get_collection("comments")
print '原记录数：%d' % coll.count()
cons = coll.distinct("id")


#选择字段
#数据去重

total = 0
for i in cons:
    j = 0
    num = coll.count({"id": i})         #原记录数
    while num > 1:
        # coll.remove({"item_id":i}, 0)        #1-全部删除，0-只删除一个
        # num -= 1
        # j += 1
        print i
        total += 1


print "item total:" % total
