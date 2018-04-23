import array
import random
import sys
import warnings

import numpy as np
from deap import base
from deap import creator
from deap import tools
from numpy import median
from ABE.main import abe_execute
from Optimizer.moead import MOEAD

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from Optimizer.feature_link import transform, get_setting_obj, mre_calc, sa_calc, transform2

NDIM = 6


def randlist(a=1, b=7, c=2, d=5, e=3, f=5):
    k = [random.randint(0, a),
         random.randint(0, b),
         random.randint(0, c),
         random.randint(0, d),
         random.randint(0, e),
         random.randint(0, f)]
    return k


def de_estimate(MU, NGEN, data):
    """

    :param MU: int
    :param data:
    :return: mre, sa, best configuration. IF NGEN IS A LIST, RETURN A LIST [[mre,sa, best confi][mre,sa,best config],...]
    """

    def evaluateFunc(config):
        return transform(config, data)

    # Differential evolution parameters
    CR = 0.5
    F = 1
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
    g = 0
    fits_old = [ind.fitness.values[0] for ind in pop]

    while count < LIFE and g < max(NGEN):
        g += 1
        print("count:" + str(count) + " gen:" + str(g))

        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            # lis = [1, 7, 2, 5, 3, 5]
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    # y[i] = (a[i] + F * (b[i] - c[i])) % lis[i]
                    y[i] = random.choice([a[i], random.choice([b[i], c[i]])])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y

        hof.update(pop)
        fits_new = [ind.fitness.values[0] for ind in pop]

        if median(fits_new) >= median(fits_old):
            count += 1
        fits_old = fits_new

        if count == LIFE or g == max(NGEN):
            best = hof[0].tolist()
            best = [int(i) for i in best]
            RES.append(best)

    return RES[0], g


def ga_estimate(NP, NGEN, data):
    def evaluateFunc2(config):
        return transform(config, data)

    # Genetic algorithm parameters
    CX = 0.6
    MUT = 0.1
    LIFE = 5

    toolbox = base.Toolbox()
    creator.create("FitnessMin2", base.Fitness, weights=[-1.0], )
    creator.create("Individual2", array.array, typecode='d', fitness=creator.FitnessMin2)
    toolbox.register("select2", tools.selTournament, tournsize=3)
    toolbox.register("evaluate2", evaluateFunc2)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    pop = [creator.Individual2(randlist()) for _ in range(NP)]
    hof = tools.HallOfFame(1)
    for ind in pop:
        fitness = toolbox.evaluate2(ind)
        ind.fitness.values = fitness
    if type(NGEN) is not list:
        NGEN = [NGEN]

    RES = list()
    count = 0
    g = 0
    fits_old = [ind.fitness.values[0] for ind in pop]

    while count < LIFE and g < max(NGEN):
        g += 1
        print("count:" + str(count) + " gen:" + str(g))

        offspring = toolbox.select2(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CX:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate2, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        hof.update(pop)
        fits_new = [ind.fitness.values[0] for ind in pop]

        if median(fits_new) >= median(fits_old):
            count += 1
        fits_old = fits_new

        if count == LIFE or g == max(NGEN):
            best = hof[0].tolist()
            best = [int(i) for i in best]
            RES.append(best)

    return RES[0], g


def moead_estimate(NP, NGEN, data):
    """

    :param NP:
    :param NGEN:
    :param data:
    :return:
    """
    toolbox = base.Toolbox()
    creator.create("FitnessMin4", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual4", array.array, typecode='d', fitness=creator.FitnessMin4)

    # toolbox.register("select4", tools.selNSGA2)
    toolbox.register("mate3", tools.cxTwoPoint)

    def mutInBound(individual, lower, upper, indpb):
        """ mutation by bit. gurantee to return the integer within [lower_i, upper_i]

        :param individual: Individual to be mutated.
        :param lower: a list defining the lower bound of each dimension in decision space
        :param upper: upper list, see lower
        :param indpb: mutation rate (by bit)
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(lower[i], upper[i])

        return individual,

    toolbox.register("mutate3", mutInBound, lower=[0] * 6, upper=[1, 7, 2, 5, 3, 5], indpb=0.1)

    def tmp_eval(ind):
        ind.fitness.values = transform2(ind, data)

    toolbox.register("evaluate4", tmp_eval)

    MU = 50
    LAMBDA = 2
    CXPB = 0.7
    MUTPB = 0.2

    ea = MOEAD(NP, toolbox, MU, CXPB, MUTPB, ngen=NGEN, nr=LAMBDA)
    pop = ea.execute()

    best, g = pop[np.argmin([ind.fitness.values[0] for ind in pop])]  # type(best) is Individual
    best = [int(i) for i in best]  # type(best) is decision list

    return best, g


def nsga2_estimate(NP, NGEN, data):
    CXPB = 0.9
    LIFE = 5

    toolbox = base.Toolbox()
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)

    toolbox.register("select3", tools.selNSGA2)
    toolbox.register("mate2", tools.cxTwoPoint)

    def mutInBound(individual, lower, upper, indpb):
        """ mutation by bit. gurantee to return the integer within [lower_i, upper_i]

        :param individual: Individual to be mutated.
        :param lower: a list defining the lower bound of each dimension in decision space
        :param upper: upper list, see lower
        :param indpb: mutation rate (by bit)
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(lower[i], upper[i])

        return individual,

    toolbox.register("mutate2", mutInBound, lower=[0] * 6, upper=[1, 7, 2, 5, 3, 5], indpb=0.1)

    pop = [creator.Individual3(randlist()) for _ in range(NP)]

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = transform2(ind, data)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select3(pop, len(pop))

    for ind in pop:
        ind.fitness.values = transform2(ind, data)

    if type(NGEN) is not list:
        NGEN = [NGEN]

    RES = list()
    count = 0
    g = 0
    fits_old = [ind.fitness.values[0] for ind in pop]

    # Begin the generational process
    while count < LIFE and g < max(NGEN):
        # Vary the population
        g += 1
        print("count:" + str(count) + " gen:" + str(g))

        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate2(ind1, ind2)

            toolbox.mutate2(ind1)
            toolbox.mutate2(ind2)
            del ind1.fitness.values
            del ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = transform2(ind, data)

        # Select the next generation population
        pop = toolbox.select3(pop + offspring, NP)

        fits_new = [ind.fitness.values[0] for ind in pop]

        if median(fits_new) >= median(fits_old):
            count += 1
        fits_old = fits_new

        if count == LIFE or g == max(NGEN):
            # find out the best ( with best MRE )
            best = pop[np.argmin([ind.fitness.values[0] for ind in pop])]  # type(best) is Individual
            best = [int(i) for i in best]  # type(best) is decision list
            RES.append(best)

    return RES[0], g


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
