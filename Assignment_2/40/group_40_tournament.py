from deap import base, creator, tools, algorithms

import numpy as np
import pandas as pd
import sys, os
import time
from group_40_controller import group40Controller

in_seconds = time.time()  # added sets time marker
in_local_time = time.ctime(in_seconds)  # added local time

sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'group_40_tournament'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# DEAP Setup

creator.create("FitnessEvo", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessEvo, strategy=None)
creator.create("Strategy", np.ndarray)

# set parameters
NEURONS = 10

INPUTS = 20

OUTPUTS = 5

# editable parameters

POP_SIZE = 50
LAMBDA = 65

POP_SIZE = 40
LAMBDA = 160
N_GEN = 30

# testing enemies:
g1 = [1, 5, 7]  # each with different action number
g2 = [2, 6, 8]  # each with different action number
g3 = [5, 6]  # same action number of 3
g4 = [1, 2]  # same action number of 4
g5 = [7, 8]  # same action number of 6
g6 = [1, 5]  # action 4 and action 3
g7 = [2, 5]  # action 4 and action 3
g8 = [6, 7]  # action 3 and action 6
g9 = [5, 8]  # action 3 and action 6
g10 = [4, 6]  # action 4 and action 3
g11 = [3, 5]  # action 4 and action 3
g12 = [3,4,6]           #two of action 4 and one action 3
g13=[1,2,3,4,5,6,7,8]   #all enemies

ENEMY = g1  # manually change this

total_weights = (INPUTS + 1) * NEURONS + (NEURONS + 1) * OUTPUTS


def genInd(ind_cls, strat_cls, total_weights, scale):
    ind = ind_cls(scale * (2 * np.random.random((total_weights)) - 1))
    ind.strategy = strat_cls(np.random.random((total_weights)))
    return ind


toolbox = base.Toolbox()

toolbox.register("individual", genInd, creator.Individual, creator.Strategy, total_weights, 20)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(ind):
    env = Environment(
        experiment_name=experiment_name,
        enemies=ENEMY,
        level=2,
        playermode="ai",
        player_controller=group40Controller(ind),
        enemymode="static",
        speed="fastest",
        multiplemode="yes"
    )
    results = env.play()
    return (results[0],)


def final_evaluate(ind):
    out_arr = []
    for i in range(10):
        out_arr.append({})
        for enemy in range(1, 9):
            env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy],
                level=2,
                playermode="ai",
                player_controller=group40Controller(ind),
                enemymode="static",
                speed="fastest"
            )
            results = env.play()
            out_arr[-1][enemy] = results[0]
    return out_arr


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=4)  # tournament selection

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(1, similar=np.array_equal)

best_runs = []

best_fitness = -1000
best_ind = None

for i in range(1, 11):

    pop = toolbox.population(n=POP_SIZE)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=LAMBDA, halloffame=hof,
                                             cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, stats=stats, verbose=True)

    for ind in hof:
        current_ind = ind
    tot = 0
    for j in range(5):
        tot += evaluate(current_ind)[0] / 5

    if tot > best_fitness:
        best_fitness = tot
        best_ind = current_ind

    print("BEST", j, tot)
    best_runs.append(tot)

    df_log = pd.DataFrame(logbook)
    df_log.to_csv(f'results_tournament_selection/test{ENEMY}{i}.csv', index=False)

best_log = pd.DataFrame()
best_log["best"] = best_runs
best_log.to_csv(f'results_tournament_selection/best_results_{ENEMY}.csv', index=False)

out = final_evaluate(best_ind)
out_df = pd.DataFrame(out)
out_df.to_csv(f'results_tournament_selection/test{ENEMY}_final_results.csv', index=False)

np.savetxt(f'results_tournament_selection/test{ENEMY}_best_weights.txt', best_ind)

#### experiment runtime
end_seconds = time.time()  # print total execution time for experiment
end_local_time = time.ctime(end_seconds)
ex_time = str(round((end_seconds - in_seconds) / 60))
print('\nExecution time: ' + ex_time + ' minutes \n')

# add info of experiment in a csv file with some details
run_data = {'Ending local time': end_local_time,
            'Runtime:': ex_time,
            'N_generations:': N_GEN,
            'Pop_size:': POP_SIZE,
            'Enemy:': [ENEMY]
            }

df_runtime = pd.DataFrame(run_data)
df_runtime.to_csv(f'results_tournament_selection/runtime_t.csv', sep='|', mode='a', index=False,
                  header=False)  # appending data every time we run an experiment to keep track of enemies and runtime
