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

POP_SIZE=10

total_weights = INPUTS * NEURONS + NEURONS + NEURONS * OUTPUTS


def genInd(ind_cls, strat_cls, total_weights):
    ind = ind_cls((2 * np.random.random((total_weights)) - 1))
    ind.strategy = strat_cls(np.random.random((total_weights)))
    return ind

toolbox = base.Toolbox()

toolbox.register("individual", genInd, creator.Individual, creator.Strategy, total_weights)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    env = Environment(
        experiment_name=experiment_name,
        enemies=[1],
        level=2,
        playermode="ai",
        player_controller=group40Controller(ind),
        enemymode="static",
        speed="fastest"
    )
    env.play()
    return (env.fitness_single(),)


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxESBlend, alpha=0.01)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(1)

pop = toolbox.population(n=POP_SIZE)

pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=POP_SIZE, lambda_=100,
            cxpb=0.6, mutpb=0.3, ngen=5, stats=stats, verbose=True)
