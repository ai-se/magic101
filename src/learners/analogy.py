"""
Analogy Based Estimator in sklearn format

Author name: Rahul Krishna
Author email: i.m.ralk@gmail.com
Date: April 17th, 2018
"""

from sklearn.base import BaseEstimator
from abe_utils import *


class ABE(BaseEstimator):
    """
    My implementation of an analogy based estimator
    """

    def __init__(self,
                 similarity_type=0, subset_selection=0, discrete_type=0,
                 fselect_type=0, adaption_type=0, analogy_type=0):
        """

        :param similarity_type: int [0,5]
            0 = Simple weighted euclidean
            1 = Maximum distance
            2 = Triangular distribution
            3 = Minkowski distance
            4 = Unweighted euclidean
            5 = Mean of ranking

        :param subset_selection: int [0,1]
            0 = Remove Nothing
            1 = Remove outliers

        :param discrete_type: int [0,2]
            0 = No discrete_type
            1 = Equal Weight
            2 = Equal Frequency

        :param fselect_type: int [0,7]
            0 = CFS
            1 = RLF
            2 = PC
            3 = CNS
            4 = WRP
            5 = FWABE
            6 = AX
            7 = IG

        :param adaption_type: int [0,3]
            0 = Median
            1 = Weighted Mean
            2 = Unweighted Mean
            3 = Second Learner (uses CART by default)  # TODO: Add other learners

        :param analogy_type: int [0, 5]
            0 = Dynamic
            1-5 = K=1-5
        """
        self.similarity_type = similarity_type
        self.subset_type = subset_selection
        self.discrete_type = discrete_type
        self.fselect_type = fselect_type
        self.adaption_type = adaption_type
        self.analogy_type = analogy_type

    def fit(self, X, y):
        """
        Fit input data

        :param X: array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        :param y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        :return: self
        """

    def predict(self, X):
        """
        Predict values

        :param X: Array-like of shape = [n_samples, n_features]
                   The input samples.
        :return: array of shape = [n_samples]
        """
