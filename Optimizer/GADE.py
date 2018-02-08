import logging
import random
import sys

from deap import base
from deap import creator
from deap import tools

from Optimizer.feature_link import transform, convert

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.DEBUG)


def randlist(a=2, b=7, c=4, d=5, e=3, f=5):
    k = [random.randint(0, a),
         random.randint(0, b),
         random.randint(0, c),
         random.randint(0, d),
         random.randint(0, e),
         random.randint(0, f)]
    return k


creator.create("FitnessMin", base.Fitness, weights=[-1.0], vioconindex=list())
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", transform)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

pop = [creator.Individual(randlist()) for _ in range(3)]

CXPB, MUTPB = 0.5, 0.2

temp_list = list(map(toolbox.evaluate, pop))
fitnesses = list(zip(temp_list, ))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

fits = [ind.fitness.values[0] for ind in pop]

g = 0

while g < 3:
    g += 1
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))

    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    temp_list = list(map(toolbox.evaluate, invalid_ind))

    fitnesses0 = list(zip(temp_list, ))

    for ind, fit in zip(invalid_ind, fitnesses0):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
# print(best_ind)
# print(convert(best_ind))
print("Best configuration is %s, %s" % (convert(best_ind), best_ind.fitness.values))
