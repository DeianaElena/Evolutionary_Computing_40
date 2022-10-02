import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("best_results_7git.csv")

df["Enemy 7 Best"] = df["best"]
boxplot = df.boxplot(column="Enemy 7 Best")

boxplot.set_ylabel("Fitness")
boxplot.set_title("Enemy 7 Best Fitness Box Plot")
plt.show()
