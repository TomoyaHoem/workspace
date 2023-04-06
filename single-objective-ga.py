# Single-Objective Evolutionary Algorithm
# to solve Shakespeare Monkey Problem
# using DEAP - framework

import random
import string
import os

from deap import base
from deap import creator
from deap import tools

print("\n")
print("-" * os.get_terminal_size().columns)
print(f"Single-Objective Optimization using DEAP-framework")
print("-" * os.get_terminal_size().columns)

# target to be found
target_sequence = "To be, or not to be."
target_len = len(target_sequence)

print("Solve Shakespearean Monkey Problem using EA:")
print()
print(f"Target sequence is: {target_sequence}, with length: {target_len}")
print("\n")

# char alphabet (lower and uppercase ascii as well as whitespaces)
alphabet = string.ascii_letters + string.whitespace


# evaluation function (#of incorrect chars / target_len)
def evalFitness(individual: list) -> float:
    correct_count = 0
    for i in range(target_len):
        if individual[i] == target_sequence[i]:
            correct_count += 1

    return correct_count / target_len


# mutation function (randomize some chars depending on prob)
def mutRandChar(individual: list, indpb: float) -> list:
    for i in range(target_len):
        if random.random() < indpb:
            individual[i] = random.choice(alphabet)

    return individual


def main() -> None:
    # create types for Fitness and Individual
    # Types ~> defines structure of fintess and individual

    ## here we create a single-objective maximizing fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    ## and individuals using lists and defined fitness
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
    CXPB, MUTPB = 0.5, 0.2

    fits = [ind.fitness.values[0] for ind in pop]

    # evolution loop
    g = 0

    while max(fits) < 1 and g < 1000:
        g += 1
        print(f"-- Generation {g} --")

        # select next generation of individuals
        offspring = toolbox.select(pop, len(pop))
        # clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover and mutation on offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # gather all fitnesses in one list and print stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {mean}")
        print(f"  Std {std}")

    print("Finished EA")


if __name__ == "__main__":
    main()
