from deap import base, creator, tools, algorithms

import numpy as np

import sys, os

from group_40_controller import group40Controller


sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'group_40'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# DEAP Setup

creator.create("FitnessEvo", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessEvo, strategy=None)
creator.create("Strategy", np.ndarray)

NEURONS=10

INPUTS=20

OUTPUTS=5

POP_SIZE=3

total_weights = (INPUTS + 1) * NEURONS + (NEURONS + 1) * OUTPUTS


def genInd(ind_cls, strat_cls, total_weights, scale):
    ind = ind_cls(scale * (2 * np.random.random((total_weights)) - 1))
    ind.strategy = strat_cls(np.random.random((total_weights)))
    return ind

toolbox = base.Toolbox()

toolbox.register("individual", genInd, creator.Individual, creator.Strategy, total_weights, 1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2, 5, 8],
        level=2,
        playermode="ai",
        player_controller=group40Controller(ind),
        enemymode="static",
        speed="fastest"

    )
    env.play()
    return (env.fitness_single(),)


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.3)
toolbox.register("select", tools.selBest)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(1)

pop = toolbox.population(n=POP_SIZE)

pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=POP_SIZE, lambda_=10,
            cxpb=0.4, mutpb=0.5, ngen=100, stats=stats, verbose=True)
