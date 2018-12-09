#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/11/2017 8:31 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : jdData.py
# @Software: PyCharm
import sys;
reload(sys);
sys.setdefaultencoding('utf8')

import json
import pymongo
from pprint import pprint
from collections import defaultdict
conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database("jd")     # other condom jd
questions = db.get_collection("questions")
answers = db.get_collection("answers")

with open('questions.dat', 'w') as f:
    # f.write()
    for index, question in enumerate(questions.find()):
        answerList = question["answerList"]
        content = question["content"]
        questionId = question["id"]
        # print index, "th", content
        f.write(str(index)+"th " + content + "\n")
        # pprint(answers.find({"associatedId": questionId}))
        for answs in answers.find({"associatedId": questionId}):
            f.write("\t"+answs["content"] + "\n")

    f.close()

