import random
import numpy
import sys

from moead import MOEAD

from deap import base
from deap import creator
from deap import tools

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import numpy as np


class Learner:
    """
    Implement a learner
    """

    def __init__(self, dframe, learner=DecisionTreeRegressor):
        self.data = dframe
        self.learner = learner

    def eval_learner(self, individual):
        """Provide MRE and CI scores as the two objectives"""
        clf = self.learner(
            max_depth=individual[0],
            max_features=individual[1],
            min_samples_leaf=individual[2],
            min_samples_split=individual[3],
            min_impurity_decrease=individual[4],
            min_weight_fraction_leaf=individual[5]
        )

        X = self.data[self.data.columns[:-1]]
        y = self.data[self.data.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mre = [(y_p - y_a) / y_a for y_p, y_a in zip(y_pred, y_test)]
        return np.mean(mre)

    def mutate(self, individual):
        """Perform mutation"""
        for i, item in enumerate(individual):
            if random.random() > 0.5:
                if i == 1:
                    individual[i] = random.randint(1, 7)
                else:
                    individual[i] = random.random()

        return individual


class Tuner:
    def __init__(self, data):
        self.learner = Learner(data)
        self.creator = self._setup_creator()
        self.toolbox = self._setup_toolbox()

    @staticmethod
    def _setup_creator():
        """
        Define a creator method for DEAP
        :return: creator
        :rtype: creator
        """

        "Define a fitness evaluator"
        creator.create("Fitness", base.Fitness, weights=(-1.0,))

        "Define an individual"
        creator.create("Individual", list, fitness=creator.Fitness)

        return creator

    def _setup_toolbox(self):
        INT_MIN, INT_MAX = 1, 7
        N_CYCLES = 1

        toolbox = base.Toolbox()

        """
        Following attributes are being tuned for in CART. 
        For details see: https://goo.gl/rAS7Lr
        
        max_depth: The maximum depth of the tree. Of type int of range [0,#columns]
        max_features: The number of features to consider when looking for the best split. Of type float.
        min_samples_leaf: The minimum number of samples required to be at a leaf node. Of type float.
        min_samples_split: The minimum number of samples required to split an internal node. Of type float.
        min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or 
                               equal to this value. Of type float.
        min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) 
                                  required to be at a leaf node. Of type float.
        """
        toolbox.register("attr_max_depth", random.randint, INT_MIN, INT_MAX)
        toolbox.register("attr_max_features", random.random)
        toolbox.register("attr_min_samples_leaf", random.random)
        toolbox.register("attr_min_samples_split", random.random)
        toolbox.register("attr_min_impurity_decrease", random.random)
        toolbox.register("attr_min_weight_fraction_leaf", random.random)
        toolbox.register("individual", tools.initCycle, self.creator.Individual,
                         (toolbox.attr_max_depth,
                          toolbox.attr_max_features,
                          toolbox.attr_min_samples_leaf,
                          toolbox.attr_min_samples_split,
                          toolbox.attr_min_impurity_decrease,
                          toolbox.attr_min_weight_fraction_leaf),
                         n=N_CYCLES)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.learner.eval_learner)

        toolbox.register("select", tools.selBest, k=10)
        toolbox.register("mutate", self.learner.mutate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)

        return toolbox

    def tune(self):

        NGEN = 50
        MU = 50
        LAMBDA = 2
        CXPB = 0.7
        MUTPB = 0.2

        random.seed(1729)

        pop = self.toolbox.population(n=MU)
        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        ea = MOEAD(pop, self.toolbox, MU, CXPB, MUTPB, ngen=NGEN,
                   stats=stats, halloffame=hof, nr=LAMBDA)

        pop = ea.execute()

        return pop, stats, hof


if __name__ == "__main__":
    pass