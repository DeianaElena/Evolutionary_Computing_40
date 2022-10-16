
import sys,os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

from group_40_controller import group40Controller

# imports other libs
import numpy as np

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# sol = np.loadtxt('best_weights_fixed.txt')
sol = np.loadtxt('results_best_selection/Elena_tests/pop25lam150gen30e78t277_BEST/test[7, 8]_best_weights.txt')              #testing with best results
# sol = np.loadtxt('results_best_selection/Elena_tests/pop25lam150gen30e46t260_BEST/test[4, 6]_best_weights.txt')              #testing with our results

# sol = np.loadtxt('results_tournament_selection/pop25_lam150_gen30_46_BEST/test[4, 6]_best_weights.txt')              #testing with our results
# sol = np.loadtxt('results_tournament_selection/pop25_lam150_gen30_78_BEST/test[7, 8]_best_weights.txt')              #testing with our results



controller = player_controller(n_hidden_neurons)

group40Controller(sol)


# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=controller,
		  		  speed="fastest",
				  enemymode="static",
				  level=2)



# tests saved demo solutions for each enemy
for en in range(1, 9):

    #Update the enemy
    env.update_parameter('enemies',[en])

    results = env.play(sol)

    print(results[1] - results[2])
    avg_gain = (results[1] - results[2])/8

print("Average_Gain", avg_gain)
print('\n  \n')

# %%
