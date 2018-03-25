#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>, Tianpei Xia <txia4@ncsu.edu>
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
from utils.kfold import KFoldSplit_df
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from deap import base
from deap import creator
from deap import tools
from sklearn.neighbors import KNeighborsRegressor
import warnings
import pandas as pd
import math
import numpy as np
import scipy as sc
import random
import pdb
import time
import ABE.measures

"""
Eight Feature Weighting Methods
note (by jf.chen)- here feature weighting includes feature selection

input is a pd.dataframe
output weighted dataframe. type=pd.dataframe

rows will not shuffled.

returned weights are **NOT** necessary to be normalized
"""


def default(df):
    """
    By default, do nothing
    :param df:
    :return:
    """
    return df


def remain_same(df):
    return df


def _ent(data):
    """
    # Input a pandas series. calculate the entropy of series
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
    target = df.columns[-1]
    for a_i, a in enumerate(df.columns[:-1]):  # for each attribute a
        for typea in set(df.loc[:, a]):  # each class of attribute a
            selected_a = df[df[a] == typea]
            sub = 0
            for typec in types_C:
                p_c_a = selected_a[selected_a[target] == typec].shape[0] / selected_a.shape[0]
                if p_c_a == 0:
                    continue
                sub += p_c_a * math.log(p_c_a, 2)
            weights.loc[0, a] += -1 * selected_a.shape[0] / df.shape[0] * sub

    weights = H_C - weights
    weights[df.columns[-1]] = 1
    weights = weights.append([weights] * (df.shape[0] - 1), ignore_index=False)
    weights.index = df.index

    res = weights * df

    return res[df.columns]


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
    weights[df.columns[-1]] = 1
    weights = weights.append([weights] * (df.shape[0] - 1), ignore_index=True)
    weights.index = df.index

    return (weights * df)[df.columns]
    # return df


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
    res = pd.DataFrame(data=new, index=df.index)
    res[target] = df[target]

    return res


def cfs(df):
    """
    - CFS = Correlation-based Feature Selection
    - reference: sect 2.4 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    reference2: Hall et al. "Correlation-based Feature Selection for Discrete and Numeric Class Machine Learning"
    - Good feature subsets contain features highly corrleated with the calss, yet uncorrelated with each other.
    - random_config search is applied for figure out best feature subsets
    :param df:
    :return:
    """

    features = df.columns[:-1]
    target = df.columns[-1]
    cf = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=features, index=df.columns[-1:])
    ff = pd.DataFrame(data=np.zeros([len(features), len(features)]), index=features, columns=features)

    # fill in cf
    for attr in cf.columns:
        cf.loc[target, attr] = abs(df[attr].corr(df[target], method='pearson'))

    # fill in ff
    for attr1 in ff.index:
        for attr2 in ff.columns:
            if attr1 == attr2: continue
            if ff.loc[attr1, attr2]: continue
            corr = abs(df[attr1].corr(df[attr2], method='pearson'))
            ff.loc[attr1, attr2] = corr
            ff.loc[attr2, attr1] = corr

    def merit_S(fs, cf, ff):
        """
        Calculate the heuristic (to maximize) according to Ghiselli 1964. eq1 in ref2
        :param ff:
        :param cf:
        :param fs: feature_subset names
        :return:
        """
        r_cf = cf[fs].mean().mean()
        r_ff = ff.loc[fs, fs].mean().mean()
        k = len(fs)
        return k * r_cf / math.sqrt(k + (k - 1) * r_ff)

    # use stochastic search algorithm to figure out best subsets
    # features subsets are encoded as [0/1]^F

    hc_starts_at = time.time()
    lst_improve_at = time.time()
    best = [0, None]
    while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
        # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
        selects = [random.choice([0, 1]) for _ in range(len(features))]
        if not sum(selects): continue
        fs = [features[i] for i, v in enumerate(selects) if v]
        score = merit_S(fs, cf, ff)
        if score > best[0]:
            best = [score, fs]
            lst_improve_at = time.time()

    selected_features = best[1] + [target]

    return df[selected_features]


def consistency_subset(df):
    """
    - Consistency-Based Subset Evaluation
    - Subset evaluator use Liu and Setino's consistency metric
    - reference: sect 2.5 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"

    - requires: discreatization
    :param df:
    :return:
    """

    def consistency(sdf, classes):
        """
        Calculate the consistency of feature subset, which will be maximized
        :param sdf: dataframe regrading to a subset feature
        :return:
        """
        sdf = sdf.join(classes)
        uniques = sdf.drop_duplicates()
        target = classes.name

        subsum = 0

        for i in range(uniques.shape[0] - 1):
            row = uniques.iloc[i]
            matches = sdf[sdf == row].dropna()
            if matches.shape[0] <= 1: continue
            D = matches.shape[0]
            M = matches[matches[target] == float(matches.mode()[target])].shape[0]
            subsum += (D - M)

        return 1 - subsum / sdf.shape[0]

    features = df.columns[:-1]
    target = df.columns[-1]

    hc_starts_at = time.time()
    lst_improve_at = time.time()
    best = [0, None]
    while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
        # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
        selects = [random.choice([0, 1]) for _ in range(len(features))]
        if not sum(selects): continue
        fs = [features[i] for i, v in enumerate(selects) if v]
        score = consistency(df[fs], df[target])
        if score > best[0]:
            best = [score, fs]
            lst_improve_at = time.time()

    selected_features = best[1] + [target]
    return df[selected_features]


def wrapper_subset(df):
    """
    - Wrapper Subset Evaluation
    - Reference: sect 2.6 of hall et al. "Benchmarking Attribute Selection Techniques for Discrete Class Data Mining"
    - Here we use linear regression to figure out best feature subsets. why linear regression? simple. effective
    - no need to normalization
    :param df:
    :return:
    """
    warnings.filterwarnings(action="ignore", module="scipy",
                            message="^internal gelsd")  # https://github.com/scipy/scipy/issues/5998

    def predict_error(df):
        """
        - use linear regression to make prediction
        - 5 crossover validation
        - returns errors which to be **minimized**
        :param df: with y-value at the end of that
        :return:
        """

        errors = list()

        for train, test in KFoldSplit_df(df, 5):
            trainX, trainY = train.iloc[:, :-1], train.iloc[:, -1]
            testX, testY = test.iloc[:, :-1], test.iloc[:, -1]
            predicts = LinearRegression().fit(trainX, trainY).predict(testX)
            error = mean_absolute_error(testY, predicts)
            errors.append(error)
        return sum(errors) / len(errors)

    features = df.columns[:-1]
    target = df.columns[-1]

    # use stochastic search algorithm to figure out best subsets
    # features subsets are encoded as [0/1]^F

    hc_starts_at = time.time()
    lst_improve_at = time.time()
    best = [float('inf'), None]
    while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
        # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
        selects = [random.choice([0, 1]) for _ in range(len(features))]
        if not sum(selects): continue
        fs = [features[i] for i, v in enumerate(selects) if v]
        score = predict_error(df[fs + [target]])
        if score < best[0]:
            best = [score, fs]
            lst_improve_at = time.time()

    selected_features = best[1] + [target]

    return df[selected_features]


# def genetic_weighting(df):
#     """
#     THIS METHOD WILL CREATE A NEW DATAFRAME
#     :param df:
#     :return:
#     """
#     n = len(df.columns) - 1
#
#     creator.create("FitnessMax_", base.Fitness, weights=(1.0,))
#     creator.create("Individual_", list, fitness=creator.FitnessMax_)
#
#     toolbox = base.Toolbox()
#     toolbox.register("attr_bool", random.randint, 0, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual_, toolbox.attr_bool, 14 * n)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#     def trans_weights2(popn):
#         popn = popn[0]
#         weights = list()
#
#         tmp = 0
#         for i, v in enumerate(popn):
#             tmp += 2 ** (13 - i % 14) * v
#             if i % 14 == 13:
#                 weights.append(tmp)
#                 tmp = 0
#         weights = [i / float(2 ** 14 - 1) for i in weights]
#         return weights
#
#     def fitness_function(df, w=1):
#         X = df.iloc[:, :-1]
#         Y = df.iloc[:, -1:]
#         X_W = X * w
#
#         clf = KNeighborsRegressor(n_neighbors=5)
#         clf.fit(X_W, Y)
#
#         Y_predict = clf.predict(X_W)
#         Y_predict = [i[0] for i in Y_predict]
#         Y_actual = np.ravel(Y)
#
#         MRE = abs(Y_actual - Y_predict) / (Y_actual + 0.0001)
#         MMRE = sum(MRE) / len(MRE)
#         f = 1 / (MMRE + 0.0001)
#         return f
#
#     def eval_max(individual):
#         w = trans_weights2([individual])
#         f = fitness_function(df, w)
#         return f
#
#     toolbox.register("evaluate", eval_max)
#     toolbox.register("mate", tools.cxOnePoint)
#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     toolbox.register("select", tools.selRoulette)
#     random.seed()
#
#     pop = toolbox.population(n=10)
#     CXPB, MUTPB = 0.7, 0.1
#     temp_list = list(map(toolbox.evaluate, pop))
#     fitnesses = list(zip(temp_list, ))
#
#     for ind, fit in zip(pop, fitnesses):
#         ind.fitness.values = fit
#
#     for g in range(100):
#         offspring = toolbox.select(pop, len(pop))
#         offspring = list(map(toolbox.clone, offspring))
#
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < CXPB:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
#
#         for mutant in offspring:
#             if random.random() < MUTPB:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
#
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         map(toolbox.evaluate, invalid_ind)
#         temp_list = list(map(toolbox.evaluate, invalid_ind))
#
#         fitnesses0 = list(zip(temp_list, ))
#
#         for ind, fit in zip(invalid_ind, fitnesses0):
#             ind.fitness.values = fit
#
#         pop[:] = offspring
#
#     best_ind = tools.selBest(pop, 1)[0]
#
#     final_weights = trans_weights2([best_ind])
#     final_weights.append(1)
#     return (df * final_weights)[df.columns]
def genetic_weighting(df):
    return df