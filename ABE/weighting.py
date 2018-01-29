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
from sklearn.decomposition import PCA
import pandas as pd
import math
import numpy as np
import scipy as sc
import pdb
import ABE.measures
import utils.debug

"""
Eight Feature Weighting Methods
note (by jf.chen)- here feature weighting includes feature selection

input is a pd.dataframe
output weights of each feature. type=pandas.core.frame.DataFrame

returned weights are **NOT** necessary to be normalized
"""


def default(df):
    return principal_component(df)


def _ent(data):
    """
    # Input a pandas series
    :param data:
    :return:
    """
    p_data = data.value_counts() / len(data)  # calculates the probabilities
    entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


def gain_rank(df):
    """
    information gain attribute ranking
    reference: sect 2.1 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    requires: discretization
    :param df:
    :return:
    """
    H_C = _ent(df.iloc[:, -1])
    weights = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=df.columns[:-1])

    types_C = set(df.iloc[:, -1])
    for a_i, a in enumerate(df.columns[:-1]):  # for each attribute a
        for typea in set(df.loc[:, a]):  # each class of attribute a
            selected_a = df[df[a] == typea]
            sub = 0
            for typec in types_C:
                p_c_a = selected_a[df.iloc[:, -1] == typec].shape[0] / selected_a.shape[0]
                if p_c_a == 0:
                    continue
                sub += p_c_a * math.log(p_c_a, 2)
            weights.loc[0, a] += -1 * selected_a.shape[0] / df.shape[0] * sub

    weights = H_C - weights
    return weights


def relief(df, measures=ABE.measures.default):
    """
    reference: sect 2.2 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    reference2: Kononenko et al. "Estimating Attributes: Analysis and Extensions of Relief"
    requires: discretization. distance measure provided
    :param measures:
    :param df:
    :return:
    """
    m = 20
    k = 10
    weights = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=df.columns[:-1])

    target = df.columns[-1]

    for i in range(m):
        selected_row = df.sample(1).iloc[0, :]
        dists = measures(selected_row, df)
        df['d_'] = dists
        hits = df[df[target] == df.iloc[0][-2]].iloc[:, :-1][:k]
        miss = df[df[target] != df.iloc[0][-2]].iloc[:, :-1][:k]

        t1 = np.sum(np.abs(hits - selected_row), axis=0) / (hits.shape[0] * m)
        t2 = np.sum(np.abs(miss - selected_row), axis=0) / (miss.shape[0] * m)
        weights = weights - t1 + t2
        df.drop(['d_'], axis=1, inplace=True)  # clear the distance

    weights = weights.drop(df.columns[-1], axis=1)
    weights = np.abs(weights)
    return weights


def principal_component(df):
    """
    THIS METHOD WILL CREATE A NEW DATAFRAME
    :param df:
    :return:
    """

    n_components = int(df.shape[1] * 0.25)
    pca = PCA(n_components=n_components)
    new = pca.fit_transform(df.iloc[:, :-1])

    # recreate a new dataframe
    target = df.columns[-1]
    res = pd.DataFrame(data=new)
    res[target] = df[target]

    return res


def cfs(df):
    """
    CFS = Correlation-based Feature Selection
    reference: sect 2.4 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    :param df:
    :return:
    """
    # TODO here...
    return principal_component(df)
