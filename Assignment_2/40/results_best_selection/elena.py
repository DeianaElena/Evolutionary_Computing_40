import numpy as np 
import pandas as pd

POP_SIZE=3

N_GEN = 10

#testing enemies:
g1 = [1,5,7]            #each with different action number
g2 = [2,6,8]            #each with different action number
g3 = [5,6]              #same action number of 3
g4 = [1,2]              #same action number of 4
g5 = [7,8]              #same action number of 6

ENEMY=g5  

POP_SIZE = 3

ex_time = 4

run_data = {'Runtime:' : ex_time,
            'N_generations:' : N_GEN,
            'Pop_size:' : POP_SIZE,
            'Enemy:': [ENEMY]
            }

            # 'Enemy1:' : ENEMY[0],
            # 'Enemy2:' : ENEMY[1],
            # 'Enemy3:' : ENEMY[2]
            

df_runtime= pd.DataFrame(run_data)
df_runtime.to_csv(f'results_best_selection/runtime.csv', sep = '|', mode='a', index=True, header=True)   #appending data every time we run an experiment to keep track of enemies and runtime
