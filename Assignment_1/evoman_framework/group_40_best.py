from deap import base, creator, tools, algorithms

import numpy as np
import csv
import sys, os
import pandas as pd
from group_40_controller import group40Controller


sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'group_40c'
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

ENEMY=5

total_weights = (INPUTS + 1) * NEURONS + (NEURONS + 1) * OUTPUTS


def genInd(ind_cls, strat_cls, total_weights, scale):
    ind = ind_cls(scale * (2 * np.random.random((total_weights)) - 1))
    ind.strategy = strat_cls(np.random.random((total_weights)))
    return ind

toolbox = base.Toolbox()

toolbox.register("individual", genInd, creator.Individual, creator.Strategy, total_weights, 5)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    env = Environment(
        experiment_name=experiment_name,
        enemies=[ENEMY],
        level=2,
        playermode="ai",
        player_controller=group40Controller(ind),
        enemymode="static",
        speed="fastest"
    )
    env.play()
    return (env.fitness_single(),)


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxESBlend, alpha=0.3)
toolbox.register("mutate", tools.mutESLogNormal, c=1, indpb=0.5)
toolbox.register("select", tools.selBest)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(1, similar=np.array_equal)

best_runs = []

for i in range(1, 11):

    pop = toolbox.population(n=POP_SIZE)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=POP_SIZE, lambda_=7, halloffame=hof,
                cxpb=0.4, mutpb=0.5, ngen=30, stats=stats, verbose=True)
    tot = 0
    for j in range(5):
        for ind in hof:
            tot += evaluate(ind)[0] / 5

    print("BEST", j, tot)
    best_runs.append(tot)


    df_log = pd.DataFrame(logbook)
    df_log.to_csv(f'results_best_selection/test{ENEMY}{i}.csv', index=False)

best_log = pd.DataFrame()
best_log["best"] = best_runs
best_log.to_csv(f'results_best_selection/best_results_{ENEMY}.csv', index=False)
