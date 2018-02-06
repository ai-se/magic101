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
import pdb

import pandas as pd
import numpy as np
from Optimizer.bridge import ft_dict_to_ABE_setting
from ABE.main import abe_execute
from FeatureModel.Feature_tree import FeatureTree
from utils.kfold import KFoldSplit


def random_config(ft, dataset):
    """
    Randomly generate an ABE hyperparameter.
    To reduce the error, use cross-validation (10 fold).

    :param ft:
    :param dataset:
    :return: average relative error [0,1]
    """

    while True:
        X = ft.top_down_random(1024)
        if ft.check_fulfill_valid(X):
            break
        logging.debug('=== Invalid configuration. Regenerating...')
    # print(X)
    settings = ft_dict_to_ABE_setting(X)

    avg_error = list()
    for meta, train, test in KFoldSplit(dataset, folds=10):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        error = abe_execute(S=settings, train=trainData, test=testData)
        avg_error.append(error)
    return np.mean(avg_error)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                        level=logging.INFO)

    url = "./FeatureModel/tree_model.xml"
    ft = FeatureTree()
    ft.load_ft_from_url(url)

    for _ in range(1):
        print(random_config(ft, "data/maxwell.arff"))
