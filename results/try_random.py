import array
import random
import sys
import warnings

from deap import base
from deap import creator
from deap import tools

from ABE.main import abe_execute

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from Optimizer.feature_link import transform, get_setting_obj, mre_calc, sa_calc

NDIM = 6


def randlist(a=2, b=7, c=4, d=5, e=3, f=5):
    k = [random.randint(0, a),
         random.randint(0, b),
         random.randint(0, c),
         random.randint(0, d),
         random.randint(0, e),
         random.randint(0, f)]
    return k


def random_strategy(randomTimes, trainData, testData):
    """
    :param randomTimes:
    :param trainData:
    :param testData:
    :return: mre, sa, best configuration
    """

    def evaluateFunc(config):
        return transform(config, trainData, testData)

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=[-1.0], )
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("evaluate", evaluateFunc)

    pop = [creator.Individual(randlist()) for _ in range(randomTimes)]
    hof = tools.HallOfFame(1)
    for ind in pop:
        fitness = toolbox.evaluate(ind)
        ind.fitness.values = fitness

    hof.update(pop)
    best = hof[0]

    y_predict, y_acutal = abe_execute(S=get_setting_obj(best), train=trainData, test=testData)

    return mre_calc(y_predict, y_acutal), sa_calc(y_predict, y_acutal), best
