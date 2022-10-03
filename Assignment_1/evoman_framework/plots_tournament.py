#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

#algorithm tournament

columns = ["gen", "nevals", "avg", "std", "min", "max"]

#enemy 1

df_product = pd.read_csv("results_tournament_selection/test11.csv", usecols=columns, index_col=None)
df_product1 = pd.read_csv("results_tournament_selection/test12.csv", usecols=columns, index_col=None)
df_product2 = pd.read_csv("results_tournament_selection/test13.csv", usecols=columns, index_col=None)
df_product3 = pd.read_csv("results_tournament_selection/test14.csv", usecols=columns, index_col=None)
df_product4 = pd.read_csv("results_tournament_selection/test15.csv", usecols=columns, index_col=None)
df_product5 = pd.read_csv("results_tournament_selection/test16.csv", usecols=columns, index_col=None)
df_product6 = pd.read_csv("results_tournament_selection/test17.csv", usecols=columns, index_col=None)
df_product7 = pd.read_csv("results_tournament_selection/test18.csv", usecols=columns, index_col=None)
df_product8 = pd.read_csv("results_tournament_selection/test19.csv", usecols=columns, index_col=None)
df_product9 = pd.read_csv("results_tournament_selection/test110.csv", usecols=columns, index_col=None)

#for enemy 5
# df_product = pd.read_csv("results_tournament_selection/test51.csv", usecols=columns, index_col=None)
# df_product1 = pd.read_csv("results_tournament_selection/test52.csv", usecols=columns, index_col=None)
# df_product2 = pd.read_csv("results_tournament_selection/test53.csv", usecols=columns, index_col=None)
# df_product3 = pd.read_csv("results_tournament_selection/test54.csv", usecols=columns, index_col=None)
# df_product4 = pd.read_csv("results_tournament_selection/test55.csv", usecols=columns, index_col=None)
# df_product5 = pd.read_csv("results_tournament_selection/test56.csv", usecols=columns, index_col=None)
# df_product6 = pd.read_csv("results_tournament_selection/test57.csv", usecols=columns, index_col=None)
# df_product7 = pd.read_csv("results_tournament_selection/test58.csv", usecols=columns, index_col=None)
# df_product8 = pd.read_csv("results_tournament_selection/test59.csv", usecols=columns, index_col=None)
# df_product9 = pd.read_csv("results_tournament_selection/test510.csv", usecols=columns, index_col=None)

# #for enemy 7
# df_product = pd.read_csv("results_tournament_selection/test71.csv", usecols=columns, index_col=None)
# df_product1 = pd.read_csv("results_tournament_selection/test72.csv", usecols=columns, index_col=None)
# df_product2 = pd.read_csv("results_tournament_selection/test73.csv", usecols=columns, index_col=None)
# df_product3 = pd.read_csv("results_tournament_selection/test74.csv", usecols=columns, index_col=None)
# df_product4 = pd.read_csv("results_tournament_selection/test75.csv", usecols=columns, index_col=None)
# df_product5 = pd.read_csv("results_tournament_selection/test76.csv", usecols=columns, index_col=None)
# df_product6 = pd.read_csv("results_tournament_selection/test77.csv", usecols=columns, index_col=None)
# df_product7 = pd.read_csv("results_tournament_selection/test78.csv", usecols=columns, index_col=None)
# df_product8 = pd.read_csv("results_tournament_selection/test79.csv", usecols=columns, index_col=None)
# df_product9 = pd.read_csv("results_tournament_selection/test710.csv", usecols=columns, index_col=None)


df_product_final = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"avg": "mean", "std": "mean", "max": "mean"}))

df_product_final["std1"] = df_product_final["std"]
df_product_final["max1"] = df_product_final["max"] #changed this because pandas reads "max" as a command

df_product_final_std = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"max": "std"}))

#print(df_product_final_std["max"])

fig, ax = plt.subplots()
line1, = ax.plot(df_product_final.gen, df_product_final.avg, label='MEAN OF MEANS')
ax.fill_between(df_product_final.gen, (df_product_final.avg-df_product_final['std']), (df_product_final.avg+df_product_final['std']), color='blue', alpha=.1)
line2, = ax.plot(df_product_final.gen, df_product_final.max1, label='MEAN OF MAXES')
#ax.fill_between(df_product_final.gen, (df_product_final.avg-df_product_final['max_std']), (df_product_final.avg+df_product_final['max_std']), color='orange', alpha=.1)
ax.set(xlabel='GENERATION', ylabel='AVERAGE FITNESS',
       title='TOURNAMENT SELECTION ALGORITHM - ENEMY 1')

# ax.set(xlabel='GENERATION', ylabel='AVERAGE FITNESS',
#        title='TOURNAMENT SELECTION ALGORITHM - ENEMY 5')

# ax.set(xlabel='GENERATION', ylabel='AVERAGE FITNESS',
#        title='TOURNAMENT SELECTION ALGORITHM - ENEMY 7')

ax.grid()
ax.legend()
plt.show()
#plt.savefig('Best_selection_fitness_enemy_1.png')


# %%
