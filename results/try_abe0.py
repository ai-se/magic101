
import random
import array

import numpy as np
import pandas as pd
import sys

from scipy.io import arff

from ABE.main import abe_execute, sa_calculate
from ABE.main import gen_setting_obj
from utils.kfold import KFoldSplit, KFoldSplit_df
from data.new_data import data_albrecht, data_china, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def abe0_list():
    k = [0, 0, 0, 0, 0, 0]
    return k


def transf(x):
    fm4S = [
        ['rm_noting', 'outlier', 'prototype'],
        ['remain_same', 'genetic_weighting', 'gain_rank', 'relief', 'principal_component', 'cfs', 'consistency_subset', 'wrapper_subset'],
        ['do_nothing', 'equal_frequency', 'equal_width', 'entropy', 'pkid'],
        ['euclidean', 'weighted_euclidean', 'maximum_measure', 'local_likelihood', 'minkowski', 'feature_mean_dist'],
        ['median_adaptation', 'mean_adaptation', 'second_learner_adaption', 'weighted_mean'],
        ['analogy_fix1', 'analogy_fix2', 'analogy_fix3', 'analogy_fix4', 'analogy_fix5', 'analogy_dynamic']
    ]
    x = map(int, x)
    setting_str = list()

    for i, v in enumerate(x):
        setting_str.append(fm4S[i][v])

    settings_1 = gen_setting_obj(setting_str)

    ERR = list()
    SA = list()
    input_data = data_miyazaki()         ##############################

    for train, test in KFoldSplit_df(input_data, folds=len(input_data)):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        error = abe_execute(S=settings_1, train=trainData, test=testData)
        ERR.append(error)
        sa = sa_calculate(S=settings_1, train=trainData, test=testData, inputs=input_data)
        SA.append(sa)
    return ERR, SA


def estimate():
    return transf(abe0_list())


if __name__ == "__main__":
    repeats = 10
    temp = estimate()
    mre_list = list()
    for _ in range(repeats):
        mre_list += temp[0]
    print(len(mre_list))
    print(mre_list)

    sa_list = list()
    for _ in range(repeats):
        sa_list += temp[1]
    print(len(sa_list))
    print(sa_list)

    np.savetxt("./data_file/miyazaki/abe0_miyazaki_mre.csv", mre_list, delimiter=",", fmt='%s')
    np.savetxt("./data_file/miyazaki/abe0_miyazaki_sa.csv", sa_list, delimiter=",", fmt='%s')