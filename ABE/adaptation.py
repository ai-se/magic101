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

import numpy as np
import pandas as pd
import warnings
import pdb

from sklearn import linear_model

"""
Four adaptation Methods
Use last column as target

input
- pd.dataframe including closest rows
- row pandas.core.series.Series
- dists: see analogies second return
output is a float
"""


def default(df, row=None, dists=None):
    return median_adaptation(df)


def median_adaptation(df, row=None, dists=None):
    """
    return the median of y-value
    :param df:
    :return:
    """
    Y = df.iloc[:, -1]
    return np.median(Y)


def mean_adaptation(df, row=None, dists=None):
    """
    return the mean of y-value
    :param df:
    :return:
    """
    Y = df.iloc[:, -1]
    return np.mean(Y)


def second_learner_adaption(df, row, dists=None):
    """
    Using linear regression to make prediction
    :param df:
    :return:
    """
    warnings.filterwarnings(action="ignore", module="scipy",
                            message="^internal gelsd")  # https://github.com/scipy/scipy/issues/5998
    clf = linear_model.LinearRegression()
    clf.fit(df.iloc[:, :-1], df.iloc[:, -1])
    res = float(clf.predict([row.tolist()[:-1]])[0])
    return res


def weighted_mean(df, row, dists):
    """
    - Report a weighted mean wehre the nearer analogies are weighted higher than those further away
    - Reference: Mendes et al. A comparative study of cost estimation models for web hypermedia applications
    :param df:
    :return:
    """
    if df.shape[0] <= 2 or sum(dists) == 0:
        return mean_adaptation(df)

    Y = df.iloc[:, -1]
    # normalize
    dists = pd.Series(dists)
    dists = sum(dists) - dists
    dists = dists / sum(dists)

    # weighted sum
    res = sum(dists.tolist() * Y)
    return res
