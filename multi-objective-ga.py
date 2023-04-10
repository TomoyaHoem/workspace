# Multi-Objective Evolutionary Algorithm
# to solve 0-1 Knapsack problem
# using DEAP framework

import random
import os

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

print("\n")
print("-" * os.get_terminal_size().columns)
print(f"Multi-Objective Optimization using DEAP-framework")
print("-" * os.get_terminal_size().columns)

print("Solve 0-1 Knapsack problem using EA:")
print("\n")

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20

# set seed for reproducibility
random.seed(128)

# item dictionary (items to fill knapsack with)
## item key -> integer
## item value -> (weight, value) 2-tuple
items = {}
# create random items
# weight between 1-10 and value
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))


# evaluation function, addup weights and values
def evalKnapsack(individual: set) -> tuple[int, float]:
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 10000, 0  # punish invalid knapsacks
    return weight, value


# crossover, intersection and absolute difference of sets
def cxSet(ind1: list, ind2: list) -> tuple[list, list]:
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2


# mutation, randomly add or pop element from set
def mutSet(individual: list) -> tuple[list,]:
    if random.random() < 0.5:
        if len(individual) > 0:
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return (individual,)


# create multi-objective fitness
## 2 objectives, minimize weight maximize value
# individuals are created as sets of unique indices corresponding to items in the dicitonary
creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

# register generators into toolbox
toolbox = base.Toolbox()
# generate random indices
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
# IND_INIT_SIZE times for individuals
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_item,
    IND_INIT_SIZE,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register genetic operators
toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)  # use NSGA-II selection


def main() -> None:
    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()  # keep track of best individuals from paretofront
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # specify axis=0 for multi-objective to compute statistics for each objective independently
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # run predefined algorithm µ+λ algorithm
    algorithms.eaMuPlusLambda(
        pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof
    )


if __name__ == "__main__":
    main()
