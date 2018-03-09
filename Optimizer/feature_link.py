import logging
import random
import sys
import pandas as pd
import numpy as np
from scipy.io import arff

from ABE.main import abe_execute, sa_calculate
from ABE.main import gen_setting_obj
from utils.kfold import KFoldSplit, KFoldSplit_df
from data.new_data import data_albrecht, data_china, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki


def transform(x):
    x = x.tolist()
    fm2S = [
        ['rm_noting', 'outlier', 'prototype'],
        ['remain_same', 'genetic_weighting', 'gain_rank', 'relief', 'principal_component', 'cfs', 'consistency_subset',
         'wrapper_subset'],
        ['do_nothing', 'equal_frequency', 'equal_width', 'entropy', 'pkid'],
        ['euclidean', 'weighted_euclidean', 'maximum_measure', 'local_likelihood', 'minkowski', 'feature_mean_dist'],
        ['median_adaptation', 'mean_adaptation', 'second_learner_adaption', 'weighted_mean'],
        ['analogy_fix1', 'analogy_fix2', 'analogy_fix3', 'analogy_fix4', 'analogy_fix5', 'analogy_dynamic']
    ]
    x = map(int, x)
    setting_str = list()

    for i, v in enumerate(x):
        setting_str.append(fm2S[i][v])

    settings_1 = gen_setting_obj(setting_str)

    ERR = list()
    input_data = data_miyazaki()         ##############################

    for train, test in KFoldSplit_df(input_data, folds=3):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        error = abe_execute(S=settings_1, train=trainData, test=testData)
        ERR.append(error)
    return np.mean(ERR),


def convert(x):
    x = x.tolist()
    fm3S = [
        ['rm_noting', 'outlier', 'prototype'],
        ['remain_same', 'genetic_weighting', 'gain_rank', 'relief', 'principal_component', 'cfs', 'consistency_subset',
         'wrapper_subset'],
        ['do_nothing', 'equal_frequency', 'equal_width', 'entropy', 'pkid'],
        ['euclidean', 'weighted_euclidean', 'maximum_measure', 'local_likelihood', 'minkowski', 'feature_mean_dist'],
        ['median_adaptation', 'mean_adaptation', 'second_learner_adaption', 'weighted_mean'],
        ['analogy_fix1', 'analogy_fix2', 'analogy_fix3', 'analogy_fix4', 'analogy_fix5', 'analogy_dynamic']
    ]

    x = map(int, x)
    setting_str = list()
    for i, v in enumerate(x):
        setting_str.append(fm3S[i][v])

    return setting_str


def cov(x=None):
    x = x.tolist()
    fm2S = [
        ['rm_noting', 'outlier', 'prototype'],
        ['remain_same', 'genetic_weighting', 'gain_rank', 'relief', 'principal_component', 'cfs', 'consistency_subset',
         'wrapper_subset'],
        ['do_nothing', 'equal_frequency', 'equal_width', 'entropy', 'pkid'],
        ['euclidean', 'weighted_euclidean', 'maximum_measure', 'local_likelihood', 'minkowski', 'feature_mean_dist'],
        ['median_adaptation', 'mean_adaptation', 'second_learner_adaption', 'weighted_mean'],
        ['analogy_fix1', 'analogy_fix2', 'analogy_fix3', 'analogy_fix4', 'analogy_fix5', 'analogy_dynamic']
    ]
    x = map(int, x)
    setting_str = list()

    for i, v in enumerate(x):
        setting_str.append(fm2S[i][v])

    settings_1 = gen_setting_obj(setting_str)

    mre_list = list()
    sa_list = list()
    input_data = data_miyazaki()            ##############################

    for train, test in KFoldSplit_df(input_data, folds=len(input_data)):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        error = abe_execute(S=settings_1, train=trainData, test=testData)
        mre_list.append(error)
        sa = sa_calculate(S=settings_1, train=trainData, test=testData, inputs=input_data)
        sa_list.append(sa)
    return mre_list, sa_list
