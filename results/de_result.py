
import random
import array

import numpy as np

from deap import base
from deap import creator
from deap import tools
import sys
import pdb
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from Optimizer.feature_link import convert, transform, cov

NDIM = 6

creator.create("FitnessMin", base.Fitness, weights=[-1.0], vioconindex=list())
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


def randlist(a=2, b=7, c=4, d=5, e=3, f=5):
    k = [random.randint(0, a),
         random.randint(0, b),
         random.randint(0, c),
         random.randint(0, d),
         random.randint(0, e),
         random.randint(0, f)]
    return k

toolbox = base.Toolbox()
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", transform)


def de_estimate():
    # Differential evolution parameters
    CR = 0.25
    F = 1
    MU = 10
    NGEN = 4

    pop = [creator.Individual(randlist()) for _ in range(MU)]

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("median", np.median)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max", "median"

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    # print(logbook.stream)

    for g in range(1, NGEN+1):
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
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        # print(logbook.stream)

    # print("Best individual is ", hof[0], hof[0]. fitness.values[0])
    # print("Best configuration is ", convert(hof[0]))
    # print("The error is", hof[0].fitness.values[0])
    # print(record)
    mre_list = cov(hof[0])
    return mre_list


if __name__ == "__main__":
    repeats = 20
    gen_list = list()
    for _ in range(repeats):
        gen_list += de_estimate()
    print(len(gen_list))
    print(gen_list)

    np.savetxt("de4_maxwell.csv", gen_list, delimiter=",", fmt='%s')