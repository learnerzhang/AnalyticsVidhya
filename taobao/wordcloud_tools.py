#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 5:30 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : wordcloud_tools.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

font_path = "/System/Library/fonts/PingFang.ttc"

text_from_file_with_apath = open('/Users/zhangzhen/gitRepository/AnalyticsVidhya/taobao/flycup/comments.dat').read()

wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all=True)
wl_space_split = " ".join(wordlist_after_jieba)

my_wordcloud = WordCloud(collocations=False, font_path=font_path, width=1400, height=1400, margin=2).generate(
    wl_space_split)

plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()
my_wordcloud.to_file('flycup.png')  # 把词云保存下来
