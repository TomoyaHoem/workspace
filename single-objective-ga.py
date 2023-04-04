# Single-Objective Evolutionary Algorithm
# to solve Shakespeare Monkey Problem
# using DEAP - framework

import random
import string

from deap import base
from deap import creator
from deap import tools

target_sequence = "To be, or not to be."


def main():
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
    toolbox.register("attr_char", random.choice(string.ascii_letters))
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

    # evaluation function (#of incorrect chars / target_len)

    # genetic operators

    # evolution loop


if __name__ == "__main":
    main()
