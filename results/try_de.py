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


def de_estimate(NGEN, trainData, testData):
    """

    :param NGEN:
    :param trainData:
    :param testData:
    :return: mre, sa, best configuration
    """

    def evaluateFunc(config):
        return transform(config, trainData, testData)

    # Differential evolution parameters
    CR = 0.25
    F = 1
    MU = 10

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=[-1.0], )
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", evaluateFunc)

    pop = [creator.Individual(randlist()) for _ in range(MU)]
    hof = tools.HallOfFame(1)
    for ind in pop:
        fitness = toolbox.evaluate(ind)
        ind.fitness.values = fitness

    for g in range(1, NGEN + 1):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            lis = [2, 7, 4, 5, 3, 5]
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = (a[i] + F * (b[i] - c[i])) % lis[i]
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)

    best = hof[0]
    y_predict = abe_execute(S=get_setting_obj(best), train=trainData, test=testData)
    y_acutal = testData.iloc[:, -1].tolist()

    # calculate MRE, SA

    return mre_calc(y_predict, y_acutal), sa_calc(y_predict, y_acutal), best
