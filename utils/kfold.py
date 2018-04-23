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

import random

import numpy
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def KFoldSplit(arff_file_name, folds=3):
    """
    Split the dataset in cross-validation

    :param arff_file_name:
    :param folds:
    :return:
    """
    data, meta = arff.loadarff(arff_file_name)
    random.shuffle(data)
    indices = range(len(data))
    kf = KFold(n_splits=folds)

    for train, test in kf.split(indices):
        trainData = data[train]
        testData = data[test]
        yield (meta, trainData, testData)


def KFoldSplit_df(df, folds=3):
    kf = KFold(n_splits=folds)

    df = shuffle(df)
    for train, test in kf.split(df.index):
        trainData = df.iloc[train]
        testData = df.iloc[test]
        yield trainData, testData
