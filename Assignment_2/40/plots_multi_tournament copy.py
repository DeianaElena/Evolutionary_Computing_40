#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

#algorithm tournament

columns = ["gen", "nevals", "avg", "std", "min", "max"]

ENEMY=[6,7]         

FOLDER = 'pop50_lam65_gen15'   #or Elena_tests'  #ADD NAME OF PATH/FOLDER MANUALLY

df_product = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}1.csv", usecols=columns, index_col=None)
df_product1 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}2.csv", usecols=columns, index_col=None)
df_product2 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}3.csv", usecols=columns, index_col=None)
df_product3 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}4.csv", usecols=columns, index_col=None)
df_product4 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}5.csv", usecols=columns, index_col=None)
df_product5 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}6.csv", usecols=columns, index_col=None)
df_product6 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}7.csv", usecols=columns, index_col=None)
df_product7 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}8.csv", usecols=columns, index_col=None)
df_product8 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}9.csv", usecols=columns, index_col=None)
df_product9 = pd.read_csv(f"results_tournament_selection/{FOLDER}/test{ENEMY}10.csv", usecols=columns, index_col=None)

df_product_final = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"avg": "mean", "std": "mean", "max": "mean"}))

df_product_final_std = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"max": "mean"}))

df_product_final["std1"] = df_product_final["std"]
df_product_final["max1"] = df_product_final["max"] #changed this because pandas reads "max" as a command

df_product_final_std["max_std"] = df_product_final_std.std(axis=1)
df_product_final_std["max1"] = df_product_final_std["max"] #changed this because pandas reads "max" as a command

print(df_product_final_std)

fig, ax = plt.subplots()
line1, = ax.plot(df_product_final.gen, df_product_final.avg, label='MEAN OF MEANS', marker = '.')
ax.fill_between(df_product_final.gen, (df_product_final.avg-df_product_final['std']), (df_product_final.avg+df_product_final['std']), color='blue', alpha=.1)
line2, = ax.plot(df_product_final.gen, df_product_final.max1, label='MEAN OF MAXES', marker='.')
ax.fill_between(df_product_final.gen, (df_product_final_std.max1-df_product_final_std['max_std']), (df_product_final.max1+df_product_final_std['max_std']), color='orange', alpha=.1)
ax.set(xlabel='GENERATION', ylabel='AVERAGE FITNESS',
       title=f'TOURNAMENT ALGORITHM - ENEMY {ENEMY}')

ax.grid()
ax.legend()
plt.show()
#plt.savefig('Best_selection_fitness_enemy_1.png')


# %%
