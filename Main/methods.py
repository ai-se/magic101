import array
import random
import sys
import warnings
import pdb

from deap import base
from deap import creator
from deap import tools
from numpy import median

from ABE.main import abe_execute

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from Optimizer.feature_link import transform, get_setting_obj, mre_calc, sa_calc

NDIM = 6


def randlist(a=1, b=7, c=2, d=5, e=3, f=5):
    k = [random.randint(0, a),
         random.randint(0, b),
         random.randint(0, c),
         random.randint(0, d),
         random.randint(0, e),
         random.randint(0, f)]
    return k


def de_estimate(NGEN, data):
    """

    :param NGEN: list or int
    :param data:
    :return: mre, sa, best configuration. IF NGEN IS A LIST, RETURN A LIST [[mre,sa, best confi][mre,sa,best config],...]
    """

    def evaluateFunc(config):
        return transform(config, data)

    # Differential evolution parameters
    CR = 0.5
    F = 1
    MU = 3
    LIFE = 5

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
    if type(NGEN) is not list:
        NGEN = [NGEN]

    RES = list()
    count = 0
    g = 1
    fits_old = [ind.fitness.values[0] for ind in pop]

    while count < LIFE and g < max(NGEN) + 1:
    # for g in range(1, max(NGEN) + 1):
        g += 1
        print("count:" + str(count) + " gen:" + str(g))

        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            lis = [1, 7, 2, 5, 3, 5]
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = (a[i] + F * (b[i] - c[i])) % lis[i]
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)

        fits_new = [ind.fitness.values[0] for ind in pop]
        if median(fits_new) >= median(fits_old):
            count += 1
        fits_old = fits_new

        if count == LIFE:
            best = hof[0].tolist()
            best = [int(i) for i in best]
            RES.append(best)

    if len(RES) == 1:
        return RES[0]
    else:
        return RES


def random_strategy(randomTimes, data):
    """
    :param randomTimes:
    :param data:
    :return: best_config
    """

    def evaluateFunc(config):
        return transform(config, data)

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
    best = hof[0].tolist()
    best = [int(i) for i in best]

    return best


def testing(trainData, testData, methodIds):
    def evaluateFunc(config):
        return transform(config, trainData, testData)

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=[-1.0], )
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("evaluate", evaluateFunc)

    pop = [creator.Individual(methodIds)]
    hof = tools.HallOfFame(1)
    fitness = toolbox.evaluate(pop[0])
    pop[0].fitness.values = fitness

    hof.update(pop)
    best = hof[0]

    y_predict, y_acutal = abe_execute(S=get_setting_obj(best), train=trainData, test=testData)
    return mre_calc(y_predict, y_acutal), sa_calc(y_predict, y_acutal), best
