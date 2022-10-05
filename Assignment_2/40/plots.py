import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def line_plot(average_fitness_generation, best_solutions_fitness, num_gens, enemy_id, alg_nr):
    """
        Implements the plotting of the generational average fitness value & the highest fitness value
        Is called after each experiment

    """
    # 1. PROCESS THE DATA - GET THE MEAN OF MEANS, MEAN OF MAXES AND THE STANDARD DEVIATION FOR BOTH
    data_points_avg = np.array([])  # create array that will contain the mean of means
    data_points_max = np.array([])  # create array that will contain the mean of maxes
    # create array that will contain calculated standard deviations.
    standard_deviations_average = np.array([])
    standard_deviations_max = np.array([])

    for i in range(num_gens):
        avg_of_a_gen = np.array([])
        max_of_a_gen = np.array([])
        # this has to be able to be done more efficiently... we'll figure out of that later
        for j in range(average_fitness_generation.size - 1):
            if (j % num_gens == i):
                # Using modulo we can sort the generations with each other and calculate the "mean of means" and
                # "mean of maxes"
                avg_of_a_gen = np.append(avg_of_a_gen, average_fitness_generation[j])
                max_of_a_gen = np.append(max_of_a_gen, best_solutions_fitness[j])

        # averaging the generation
        data_points_avg = np.append(data_points_avg, np.average(avg_of_a_gen))
        data_points_max = np.append(data_points_max, np.average(
            max_of_a_gen))  # averages out the best of the generations

        # Calculate standard deviation of the two arrays.
        standard_deviations_average = np.append(
            standard_deviations_average, np.std(avg_of_a_gen))  # same here
        standard_deviations_max = np.append(standard_deviations_max, np.std(
            max_of_a_gen))  # goal here is to add the standard deviation

    # 2. FILL ARRAY WITH VALUES CORRESPONDING TO NUMBER OF EXPERIMENTS
    array_gens = np.array([i + 1 for i in range(num_gens)])  # list comprehension

    # if DEBUG: print(f'Experiments array = {array_experiments, array_experiments.shape}; Average fitnesses array = {
    # data_points, data_points.shape}')

    # 3. PLOT DATAPOINTS AGAINST EXPERIMENT_NUMBER
    # REMOVE LATER MAYBE --- plt.plot(array_experiments, data_points, label="average generation fitness for experiment")
    figure, plot = plt.subplots()
    plot.errorbar(array_gens, data_points_avg, yerr=standard_deviations_average,
                  marker="o", color="blue", label="Means of means")
    plot.errorbar(array_gens, data_points_max, yerr=standard_deviations_max,
                  marker="o", color="red", label="Means of maxes")
    plot.legend(loc='lower right')
    # 4. PLOT PARAMS
    xint = range(num_gens + 1)
    plt.xticks(xint)  # set x-axis "ticks" to only integer values
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=16, titlesize=20)
    plt.xlabel('generation')
    plt.ylabel('average fitness')
    plt.title(f"Group {enemy_id}")

    plt.savefig(f"LinePlotsAlg{alg_nr}Group{enemy_id}.png")


# Take in the results of the best individuals in the form of either: Gain Measure or Fitness
def box_plot(performance1, performance2, training_group):
    """
    Plots the mean of the "performance" of the best individual from all runs.

    """
    fig, ax = plt.subplots()

    # STEP 1: We already have the data the way we want it...
    # STEP 2: Plot data
    ax.boxplot([performance1, performance2], labels=["EA1", "EA2"])
    plt.ylabel("Gain measure")
    ax.set_title(f"Best individuals - Training group {training_group}")
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=15, titlesize=20)
    plt.savefig(f"BestIndividualsBoxTrainingGroup{training_group}.png")


# Performs t-test
def t_test(performance1_alg1, performance2_alg1, performance1_alg2, performance2_alg2):
    A = [performance1_alg1, performance2_alg1]
    B = [performance1_alg2, performance2_alg2]

    C = stats.ttest_ind(A, B)
    print(C)
    file = open('Ttestresult.txt', 'w')
    file.write(str(C))
    file.close()


def table_plot(avg_player_life, avg_enemy_life):
    fig, ax = plt.subplots()
    data = [
        ["Enemy number", "Average player energy points", "Average enemy energy points"]
    ]
    for i in range(1, 9):
        indexnr = i - 1
        data.append([f"Enemy {i}", round(avg_player_life[indexnr], 1), round(avg_enemy_life[indexnr], 1)])

    table = ax.table(cellText=data, loc="center")

    table.set_fontsize(13)
    ax.axis('off')
    plt.savefig(f"BestIndividualPoints2.png")
