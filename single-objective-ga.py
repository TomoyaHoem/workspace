# Single-Objective Evolutionary Algorithm
# to solve Shakespeare Monkey Problem
# using DEAP framework

import random
import string
import os

import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

print("\n")
print("-" * os.get_terminal_size().columns)
print(f"Single-Objective Optimization using DEAP-framework")
print("-" * os.get_terminal_size().columns)

# target to be found
target_sequence = "All the world's a stage, and all the men and women merely players."
target_len = len(target_sequence)

print("Solve Shakespearean Monkey Problem using EA:")
print()
print(f"Target sequence is: {target_sequence}, with length: {target_len}")
print("\n")

# char alphabet (lower and uppercase ascii as well as whitespaces and punctuation)
alphabet = string.ascii_letters + string.whitespace + string.punctuation


# evaluation function (#of incorrect chars / target_len)
def evalFitness(individual: list) -> float:
    correct_count = 0
    for i in range(target_len):
        if individual[i] == target_sequence[i]:
            correct_count += 1

    return (correct_count / target_len,)


# mutation function (randomize some chars depending on prob)
def mutRandChar(individual: list, indpb: float) -> list:
    for i in range(target_len):
        if random.random() < indpb:
            individual[i] = random.choice(alphabet)

    return individual


# toString method for individuals
def individualToString(individual: list) -> string:
    res = ""
    return res.join(individual)


def printBestIndvs(pop, g):
    print(f"Top 10 Individuals for generation {g}:")
    print("")
    best_individuals = tools.selBest(pop, 10)
    for i in range(len(best_individuals)):
        indv_pheno = individualToString(best_individuals[i])
        print(f"{i+1}: {indv_pheno} (Fitness: {best_individuals[i].fitness.values[0]})")
    print("")


def printStats(fits, mean, std):
    print(f"  Min {min(fits)}")
    print(f"  Max {max(fits)}")
    print(f"  Avg {mean}")
    print(f"  Std {std}")
    print("")


def main() -> None:
    # set seed for reproducibility
    random.seed(128)

    # create types for Fitness and Individual
    # Type ~> defines structure of fintess and individual

    ## here we create a single-objective maximizing fitness property
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    ## and individuals using lists and the defined fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # toolbox contains logic on how to create individuals
    # and their properties
    # ~> also create population
    toolbox = base.Toolbox()
    ## here we create individuals as random words -> random char list
    ## first define way to generate random chars
    toolbox.register("attr_char", random.choice, alphabet)
    ## then, individuals from repeating these chars
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_char,
        len(target_sequence),
    )
    ## and finally a population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # genetic operators
    toolbox.register("evaluate", evalFitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutRandChar, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # evolution setup

    ## create population
    pop = toolbox.population(n=100)
    ## evaluate entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    ## constants
    CXPB, MUTPB = 0.5, 0.4

    # statistics object to create statistics data during optimization
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    # register functions with aliases
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # logbook to store statistics in
    logbook = tools.Logbook()

    # extract fitness values of individuals
    fits = [ind.fitness.values[0] for ind in pop]

    # evolution loop
    g = 0

    #  max(fits) < 1 and g < 1000:
    while g < 1000:
        g += 1
        if g % 100 == 0:
            print(f"-- Generation {g} --")

        # select next generation of individuals
        offspring = toolbox.select(pop, len(pop))
        # clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover and mutation on offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # replace current population by offspring
        pop[:] = offspring

        # gather all fitnesses in one list and print stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        if g % 100 == 0:
            print("**Statistics**")
            printStats(fits, mean, std)
            printBestIndvs(pop, g)

        # store statistics
        record = stats.compile(pop)
        logbook.record(gen=g, **record)

    print(f"**Final Statistics after {g} generations**")
    printStats(fits, mean, std)
    printBestIndvs(pop, g)

    print("Finished EA")

    gen, maxFitness, avgFitness = logbook.select("gen", "max", "avg")

    plt.plot(gen, maxFitness, label="Max Fitness")
    plt.plot(gen, avgFitness, color="g", ls="dotted", label="Average Fitness")

    plt.title(f'GA Performance for: "{target_sequence}"', wrap=True)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    main()
