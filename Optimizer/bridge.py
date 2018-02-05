#!/usr/bin/env python
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


from __future__ import division

import logging
import sys

import pandas as pd

from ABE.main import gen_setting_obj, abe_execute
from FeatureModel.Feature_tree import FeatureTree
from utils.kfold import KFoldSplit

"""
Connecting packages "ABE" and "FeatureModel"
"""

fm2S = {
    "Outlier"                            : 'outlier',
    "Prototype"                          : 'prototype',
    "Remove_Nothing"                     : 'rm_noting',
    "Information_Gain"                   : 'gain_rank',
    "Relief"                             : 'relief',
    "Principal_Components"               : 'principal_component',
    "Correlation-based_Feature_Selection": 'cfs',
    "Consistency-Based_Subset_Evaluation": 'consistency_subset',
    "Wrapper_Subset_Evaluation"          : 'wrapper_subset',
    "Feature_Weighted_ABE"               : 'genetic_weighting',
    # "Analogy-X": '',
    "Equal_Frequency"                    : 'equal_frequency',
    "Equal_Width"                        : 'equal_width',
    "Entropy"                            : 'entropy',
    # "PKID": '',
    "Remain_Same"                        : 'remain_same',
    "Weighted_Euclidean"                 : 'weighted_euclidean',
    "Unweighted_Euclidean"               : 'euclidean',
    "Max_Distance"                       : 'maximum_measure',
    "Triangular_Distribution"            : 'local_likelihood',
    "Minkowski"                          : 'minkowski',
    "Mean_of_Ranking"                    : 'mean_adaptation',
    "Median"                             : 'median_adaptation',
    "Weighted_Mean"                      : 'weighted_mean',
    "Unweighted_Mean"                    : 'mean_adaptation',
    "Second_Learner"                     : 'second_learner_adaption',
    "Dynamic"                            : 'analogy_dynamic',
    "Set_K_as_1"                         : 'analogy_fix1',
    "Set_K_as_2"                         : 'analogy_fix2',
    "Set_K_as_3"                         : 'analogy_fix3',
    "Set_K_as_4"                         : 'analogy_fix4',
    "Set_K_as_5"                         : 'analogy_fix5'
}


def ft_dict_to_ABE_setting(d):
    S_str = list()
    for item in d.keys():
        if not d[item]: continue
        if item.name in fm2S:
            S_str.append(fm2S[item.name])
    return gen_setting_obj(S_str)

# if __name__ == '__main__':
#     url = "./FeatureModel/tree_model.xml"
#     ft = FeatureTree()
#     ft.load_ft_from_url(url)
#
#     while True:
#         print('*')
#         X = ft.top_down_random(1024)
#         if ft.check_fulfill_valid(X):
#             break
#
#     settings = ft_dict_to_ABE_setting(X)
#
#     logging.basicConfig(stream=sys.stdout,
#                         format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#                         level=logging.DEBUG)
#
#     for meta, train, test in KFoldSplit("data/maxwell.arff", folds=10):
#         trainData = pd.DataFrame(data=train)
#         testData = pd.DataFrame(data=test)
#         error = abe_execute(S=settings, train=trainData, test=testData)
