#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

columns = ["gen", "nevals", "avg", "std", "min", "max"]

ENEMY=[1,5,7]

FOLDER = 'Elena_tests/pop50lam65gen15e157t110'   #or 'Elena_tests'    #ADD NAME OF PATH/FOLDER MANUALLY

df_product = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}1.csv", usecols=columns, index_col=None)
df_product1 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}2.csv", usecols=columns, index_col=None)
df_product2 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}3.csv", usecols=columns, index_col=None)
df_product3 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}4.csv", usecols=columns, index_col=None)
df_product4 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}5.csv", usecols=columns, index_col=None)
df_product5 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}6.csv", usecols=columns, index_col=None)
df_product6 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}7.csv", usecols=columns, index_col=None)
df_product7 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}8.csv", usecols=columns, index_col=None)
df_product8 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}9.csv", usecols=columns, index_col=None)
df_product9 = pd.read_csv(f"results_best_selection/{FOLDER}/test{ENEMY}10.csv", usecols=columns, index_col=None)

df_product_final = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"avg": "mean", "std": "mean", "max": "mean", "min": "mean"}))

df_product_final_med = (pd.concat([df_product, df_product1, df_product2, df_product3, df_product4, df_product5, df_product6, df_product7, df_product8, df_product9]).groupby('gen', as_index=False).
                      agg({"avg": "median"}))

df_product_final["std1"] = df_product_final["std"]
df_product_final["max1"] = df_product_final["max"] #changed this because pandas reads "max" as a command

df_product_final_med["avg_median"] = df_product_final_med.std(axis=1)

data = np.concatenate((df_product_final['avg'], df_product_final_med["avg_median"], df_product_final["max"], df_product_final["min"]))

fig1, ax1 = plt.subplots()
ax1.set(ylabel='AVERAGE FITNESS',
       title=f'BEST SELECTION ALGORITHM - ENEMY {ENEMY}')
ax1.boxplot(data)

plt.show()

# %%
