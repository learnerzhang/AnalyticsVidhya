#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/09/2017 6:19 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : condom.py
# @Software: PyCharm
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
jieba.load_userdict("user.dict")

import pymongo
from pprint import pprint
from collections import defaultdict
conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database("flycup")
comments = db.get_collection("comments")
items = db.get_collection("items")


def get_city_map():
    """
    主要统计各个城市flycup商店分布的数量
    :return: 
    """
    shop_city = defaultdict(set)
    for item in items.find():
        city = item["item_loc"].split()
        # print city[-1] + "&" + item["nick"]
        shop_city[city[-1]].add(item["nick"])
    return shop_city


def get_raw_titles():
    with open('titles.dat', 'w') as f:
        f.write("编号#title#seg_title#付款#价格#邮费#店家信誉\n")
        for item in items.find():
            raw_title = item["raw_title"]
            view_sales = item["view_sales"]
            comment_count = item["comment_count"]
            view_price = item["view_price"]
            view_fee = item["view_fee"]
            shopcard = item["shopcard"]
            nid = item["nid"]
            print nid, raw_title, view_sales, comment_count, view_price, view_fee, shopcard["sellerCredit"]
            f.write(str(nid)+"#"+raw_title+"#"+" ".join(jieba.cut(raw_title))+"#"+view_sales+"#"+str(comment_count)+"#"+str(view_price)+"#"+str(view_fee)+"#"+str(shopcard["sellerCredit"])+"\n")
        f.close

def count_item_price_nums():
    """统计各个item的价格以及购买数量"""
    max_count = 0
    url = ""
    for item in items.find():
        nid = item["nid"]
        # print "价格", item["view_price"], "邮费",item["view_fee"],"销量",item["view_sales"], "评论数", item["comment_count"],"店家信息",item["shopcard"], item["detail_url"]
        cs = comments.find({"item_id": nid})
        if cs.count() > max_count:
            max_count = cs.count()
            url = item["detail_url"]
    print max_count, url
        

def get_all_comments():
    with open('comments.dat', 'w') as f:
        # f.write()
        for comment in comments.find():
            if len(comment["appendComment"]):
                # print comment["appendComment"]["content"]
                f.write(comment["appendComment"]["content"] + "\n")
            else:
                # print comment["rateContent"]
                f.write(comment["rateContent"] + "\n")
        f.close()


def seg_all_comments():
    stopword = []
    with open('../stopword.txt', 'r') as f:
        for line in f.readlines():
             stopword.append(line.strip())
    # print stopword
    f.close()
    
    lines = set()
    with open('comments.dat', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            lines.add(line)
    f.close()         
    with open('seg_comments.dat', 'w') as f:
        for line in lines:
            seg_list = jieba.cut(line, cut_all=False)
            seg = [w for w in seg_list if w not in stopword]
            if len(seg):
                f.write(" ".join(seg)+"\n")
    f.close()


def gen_word_cloud():
    font = os.path.join(os.path.dirname(__file__), "DroidSansFallbackFull.ttf")
    text = open(u"seg_comments.dat").read().decode('utf-8')
    my_wordcloud = WordCloud(font_path=font).generate(text)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()


def gen_titles_wc():
    font = os.path.join(os.path.dirname(__file__), "DroidSansFallbackFull.ttf")
    rs = ""
    for item in items.find():
        raw_title = item["raw_title"]
        rs += " ".join(jieba.cut(raw_title, cut_all=False))
    my_wordcloud = WordCloud(font_path=font).generate(rs)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()
if __name__ == '__main__':
    # count_item_price_nums()
    # get_all_comments()
    # seg_all_comments()
    # gen_word_cloud()
    # count_item_price_nums()
    get_raw_titles()
    # gen_titles_wc()
    pass






