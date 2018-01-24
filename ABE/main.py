# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from __future__ import division

import pandas as pd
import pdb
from utils.bunch import Object
import ABE.subset
import ABE.measures
import ABE.analogies

from utils.kfold import KFoldSplit


def abe_core(settings, train, test):
    # apply case subset selectors
    train = settings.subset(train)

    # apply weighting methods
    # TODO here

    # apply discreatization
    train = settings.discrete(train)
    test = settings.discreate(test)

    # perform prediction (and similarity measures)
    y_predict = list()
    for test_row in test:
        predict = settings.predict(test_row, train, settings.measure)
        y_predict.append(predict)

    # print out errors
    y_actual = test[test.columns[-1]]
    m = 0
    for predict, actual in zip(y_predict, y_actual):
        if predict == actual:
            m += 1
    accuracy = m / (len(y_actual))

    return accuracy


def gen_setting_obj(configurations):
    components = Object()

    # three case subset selectors
    components.subset = ABE.subset.default
    for subset in dir(ABE.subset):
        if subset in configurations:
            components.subset = getattr(ABE.subset, subset)

    # six similarity measures
    components.measures = ABE.measures.default
    for measures in dir(ABE.measures):
        if measures in configurations:
            components.measures = getattr(ABE.measures, measures)

    # six ways to select analogies
    components.analogies = ABE.analogies.default
    for analogies in dir(ABE.analogies):
        if analogies in configurations:
            components.analogies = getattr(ABE.analogies, analogies)

    return components


if __name__ == '__main__':
    settings = gen_setting_obj(['outlier'])
    for meta, train, test in KFoldSplit("data/albrecht.arff", 3):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        ABE.measures.local_weight(testData.iloc[0], trainData)
        pdb.set_trace()
        abe_core(settings=settings, train=trainData, test=testData)
