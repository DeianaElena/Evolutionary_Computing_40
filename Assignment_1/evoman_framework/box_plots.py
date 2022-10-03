#%%
import pandas as pd
import os

import matplotlib.pyplot as plt

cwd = os.getcwd()
print('current wd', cwd)

df = pd.read_csv("results_best_selection/best_results_7.csv")

df["Enemy 7 Best"] = df["best"]
boxplot = df.boxplot(column="Enemy 7 Best")

boxplot.set_ylabel("Fitness")
boxplot.set_title("Enemy 7 Best Fitness Box Plot")
plt.show()

# %%
