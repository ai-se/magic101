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

import logging

import pandas as pd

import ABE.adaptation
import ABE.analogies
import ABE.discretization
import ABE.measures
import ABE.normalize
import ABE.subSelector
import ABE.weighting
from utils.bunch import ABE_configures


def abe_execute(S, train, test):
    """
    executing the ABE method
    :param S:
    :param train:
    :param test:
    :return:
    """

    # for convenience, use negative index for test
    test = test.set_index(pd.RangeIndex(start=-1, stop=-test.shape[0] - 1, step=-1))

    logging.debug("Sub selection -- " + S.subSelector.__name__)
    train = S.subSelector(train)
    n_train, n_test = train.shape[0], test.shape[0]
    combined = pd.concat([train, test])

    logging.debug("Normalization")
    combined = ABE.normalize.normalize(combined)

    logging.debug("Discretization -- " + S.discretization.__name__)
    combined = S.discretization(combined)

    logging.debug("Feature weighting -- " + S.weighting.__name__)
    combined = S.weighting(combined)

    # separate train and test
    train = combined.iloc[:n_train, :]
    test = combined.iloc[n_train:, :]

    logging.debug("Predicting " + S.analogies.__name__ + " " + S.adaptation.__name__)
    Y_predict, Y_actual = list(), list()
    for index, test_row in test.iterrows():
        dists = S.measures(test_row, train)
        closest, c_dists = S.analogies(dists, train, measures=S.measures)
        Y_predict.append(S.adaptation(closest, test_row, c_dists))
        Y_actual.append(test_row[-1])

    return Y_predict


def gen_setting_obj(S_str):
    S = ABE_configures()

    # three case subset selectors
    S.subSelector = ABE.subSelector.default
    for subSelector in dir(ABE.subSelector):
        if subSelector in S_str:
            S.subSelector = getattr(ABE.subSelector, subSelector)

    # six similarity measures
    S.measures = ABE.measures.weighted_euclidean
    for measures in dir(ABE.measures):
        if measures in S_str:
            S.measures = getattr(ABE.measures, measures)

    # six ways to select analogies
    S.analogies = ABE.analogies.default
    for analogies in dir(ABE.analogies):
        if analogies in S_str:
            S.analogies = getattr(ABE.analogies, analogies)

    # eight feature weighting methods
    S.weighting = ABE.weighting.default
    for weighting in dir(ABE.weighting):
        if weighting in S_str:
            S.weighting = getattr(ABE.weighting, weighting)

    # five discretization methods
    S.discretization = ABE.discretization.default
    for discretization in dir(ABE.discretization):
        if discretization in S_str:
            S.discretization = getattr(ABE.discretization, discretization)

    # four adaptation methods
    S.adaptation = ABE.adaptation.mean_adaptation
    for adaptation in dir(ABE.adaptation):
        if adaptation in S_str:
            S.adaptation = getattr(ABE.adaptation, adaptation)

    return S

# if __name__ == '__main__':
#     """
#     ABE algorithm Demonstration
#     """
#     logging.basicConfig(stream=sys.stdout,
#                         format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#                         level=logging.DEBUG)
#
#     settings = gen_setting_obj(
#         ['outlier', 'maximum_measure', 'analogy_dynamic', 'weighted_mean'])
#
#     for meta, train, test in KFoldSplit("data/maxwell.arff", folds=10):
#         trainData = pd.DataFrame(data=train)
#         testData = pd.DataFrame(data=test)
#         error = abe_execute(S=settings, train=trainData, test=testData)
