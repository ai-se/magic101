import numpy as np

from ABE.main import abe_execute
from ABE.main import gen_setting_obj
from utils.kfold import KFoldSplit_df


def get_setting_obj(configurationIndex):
    if not type(configurationIndex) is list:
        configurationIndex = configurationIndex.tolist()
    fm2S = [
        ['rm_noting', 'outlier', 'prototype'],
        ['remain_same', 'genetic_weighting', 'gain_rank', 'relief', 'principal_component', 'cfs', 'consistency_subset',
         'wrapper_subset'],
        ['do_nothing', 'equal_frequency', 'equal_width', 'entropy', 'pkid'],
        ['euclidean', 'weighted_euclidean', 'maximum_measure', 'local_likelihood', 'minkowski', 'feature_mean_dist'],
        ['median_adaptation', 'mean_adaptation', 'second_learner_adaption', 'weighted_mean'],
        ['analogy_fix1', 'analogy_fix2', 'analogy_fix3', 'analogy_fix4', 'analogy_fix5', 'analogy_dynamic']
    ]
    configurationIndex = map(int, configurationIndex)
    setting_str = list()

    for i, v in enumerate(configurationIndex):
        setting_str.append(fm2S[i][v])

    settings_1 = gen_setting_obj(setting_str)
    return settings_1


# def mre_calc(y_predict, y_actual):
#     mre = 0
#     for predict, actual in zip(y_predict, y_actual):
#         mre += abs(predict - actual) / (actual + 0.0001)
#     MRE = mre / (len(y_actual))
#     return MRE


def mre_calc(y_predict, y_actual):
    mre = []
    for predict, actual in zip(y_predict, y_actual):
        mre.append(abs(predict - actual) / (actual))
    MRE = np.median(mre)
    if MRE == 0:
        MRE = np.mean(mre)
    return MRE


def sa_calc(Y_predict, Y_actual):
    ar = 0
    for predict, actual in zip(Y_predict, Y_actual):
        ar += abs(predict - actual)
    mar = ar / (len(Y_predict))
    marr = sum(Y_actual) / len(Y_actual)
    sa_error = (1 - mar / marr)
    return sa_error


def transform(configurationIndex, data):
    """
    Given trainDat, TestData and configuration indices, return the MRE of given test data set.
    :param configurationIndex:
    :param data:
    :return:
    """
    Y_predict, Y_actual = abe_execute(S=get_setting_obj(configurationIndex), data=data)
    return mre_calc(Y_predict, Y_actual),


def calc_error(bestConfigIndex, testData):
    Y_predict, Y_actual = abe_execute(S=get_setting_obj(bestConfigIndex), data=testData)
    return mre_calc(Y_predict, Y_actual), sa_calc(Y_predict, Y_actual)


if __name__ == '__main__':
    from data import new_data

    for _ in range(100):
        for x, y in KFoldSplit_df(new_data.data_desharnais(), 10):
            p = transform(np.array([2, 3, 4, 1, 2, 0]), x, y)
            print(p)
