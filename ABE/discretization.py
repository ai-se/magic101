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
import pandas as pd
import numpy as np
import warnings
import sys
import pdb

"""
Five discretization Methods
Discretization will also perform over last column, i.e. the y-value
Input:
- pd.dataframe
Output:
- pd.dataframe

"""


def default(df):
    return equal_frequency(df)


def do_nothing(df):
    return df


def equal_frequency(df, groupSize=3):
    """
    by default, groupsize=10. i.e. discrete all into 10 groups. each group, value should be median of original values
    :param df:
    :param groupSize:
    :return:
    """
    for c_i in range(df.shape[1]):
        maps = pd.qcut(df.iloc[:, c_i].rank(method='first'), groupSize)
        map_v = np.zeros([df.shape[0], 1])
        for r_i, m in enumerate(maps):
            x, y = m[1:-1].split(',')
            x = float(x)
            y = float(y)
            map_v[r_i] = (y + x) / 2
        df.iloc[:, c_i] = map_v
    return df


def equal_width(df, groupSize=10):
    """
    by default, groupsize=10. i.e. discrete all into 10 groups. each group, value should be median of original values
    :param df:
    :param groupSize:
    :return:
    """
    for c_i in range(df.shape[1]):
        maps = pd.cut(df.iloc[:, c_i], groupSize)
        map_v = np.zeros([df.shape[0], 1])
        for r_i, m in enumerate(maps):
            x, y = m[1:-1].split(',')
            x = float(x)
            y = float(y)
            map_v[r_i] = (y + x) / 2
        df.iloc[:, c_i] = map_v

    return df


def entropy(df):
    """
    referece: Fayyad et al. "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning"
    :param df:
    :return:
    """

    def calc_ent(vect):
        """
        Calculate the differential entropy. reference https://www2.isye.gatech.edu/~yxie77/ece587/Lecture17.pdf
        :param vect: np.array like vector. e.g. pandas.core.series.Series
        :return:
        """
        integers = map(int, vect)

    # TODO here
    warnings.warn("Currently entropy based discretization is NOT available. Will apply equal width")
    return equal_width(df)


def pkid(df):
    """
    reference: Yang et al. "A comparative Study of Discretization Methods for Naive-Bayes Classifiers"
    :param df:
    :return:
    """
    # TODO here
    pdb.set_trace()
