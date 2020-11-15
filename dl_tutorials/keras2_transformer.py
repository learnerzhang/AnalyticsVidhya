#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 07:01 PM
# @Author  : zhangzhen
# @Site    :
# @File    : keras2_demo.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function, unicode_literals
# 安装tfds pip install tfds-nightly==1.0.2.dev201904090105
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers

import time
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)