#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os


best78 = [-70.0, 70, -70.0, -40.0, 57.400000000000325, -30.0, 69.40000000000029, 70.60000000000028]
tour78 = [-70.0, 32, -80.0, -60.0, 57.400000000000325, -70.0, 55.00000000000032, 56.800000000000324]

best46 = [-70.0, 70, -70.0, -40.0, 60.40000000000033, -30.0, 69.40000000000029, 70.60000000000028] 
tour46 = [-60.0, -90.0, -90.0, 40.000000000000284, 79.0000000000002, 43.00000000000029, -90.0, 47.8000000000003]

def box_plot(performance1, performance2, training_group):
    """
    Plots the mean of the "performance" of the best individual from all runs.

    """
    fig, ax = plt.subplots()

    # STEP 1: We already have the data the way we want it...
    # STEP 2: Plot data
    ax.boxplot([performance1, performance2], labels=["Best Sel", "Tournament Sel"])
    plt.ylabel("Gain measure")
    ax.set_title(f"Training group {training_group}")
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=15, titlesize=20)
    plt.savefig(f"BoxTrainingGroup{training_group}.png")

box_plot(best78, tour78,'7_8')
box_plot(best46, tour46,'4_6')


# Performs t-test
def t_test(performance1_alg1, performance2_alg1, performance1_alg2, performance2_alg2):
    A = [performance1_alg1, performance2_alg1]
    B = [performance1_alg2, performance2_alg2]

    C = stats.ttest_ind(A, B)
    print(C)
    file = open('Ttestresult.txt', 'w')
    file.write(str(C))
    file.close()

t_test(best46, best78, tour46, tour78)