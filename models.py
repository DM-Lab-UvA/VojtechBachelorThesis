import numpy as np
import pandas as pd
import MinCompSpin_Python.MinCompSpin as mod
import raster_plots as rplt
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Imports for getting the session data
import utils
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def create_input_file(trialData, startBin, stopBin, filename, path):
    # A function that takes a set of trial data and turns them into
    # a binary data file that MCMs can use

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if the input trial data is a dataframe
    # (i.e. contains more than one trial)
    if isinstance(trialData, pd.DataFrame):
        # Get the trimmed trial spikes data
        trial_spikes, _ = rplt.trim_spikes(trialData)

        data_li = []

        # Turn all of the trials into a single array that can be read into the file
        for _, trial in trial_spikes.items():
            neuron_arr = np.array(list(trial.values()))
            t = np.transpose(neuron_arr[:, startBin:stopBin])
            data_li.append(np.transpose(neuron_arr[:, startBin:stopBin]))

        data_t = tuple(data_li)
        data_arr = np.concatenate(data_t)

    # Otherwise, handle the trialData as a series
    else:
        neuron_arr = np.array(list(trialData["binSpikes"].values()))
        data_arr = np.transpose(neuron_arr[:, startBin:stopBin])

    # Create and open the file in write mode
    with open(os.path.join(path, f"{filename}.dat"), "w") as file:

        # Write the contents of the data array into the file,
        # in the structure necessary for the MCM module
        for row in data_arr:
            row_string = ''.join(map(str, row))
            file.write(row_string + '\n')


def generate_coocurrance_matrix(MCM_partitions, n):
    matrix = np.zeros((n, n))

    communities = []
    for array in MCM_partitions:
        if n <= 64:
            component = bin(array[1])[2:].zfill(n)
        else:
            comp1 = bin(array[1])[2:].zfill(64)
            comp2 = bin(array[2])[2:].zfill(n-64)
            component = comp1 + comp2

        communities.append(component)

    for i in range(n):
        for j in range(n):
            for component in communities:
                if (component[i] == "1") & (component[j] == "1"):
                    matrix[i,j] = 1

    return matrix


def count_component_size(MCM_partitions, n, exclude_singles):
    count_li = []

    for array in MCM_partitions:
        if n <= 64:
            component = bin(array[1])[2:].zfill(n)
        else:
            comp1 = bin(array[1])[2:].zfill(64)
            comp2 = bin(array[2])[2:].zfill(n-64)
            component = comp1 + comp2

        count = 0
        for i in range(n):
            if (component[i] == "1"):
                count += 1

        if (exclude_singles & (count == 1)):
            continue

        count_li.append(count)

    return np.array(count_li)

# A function to plot a heatmap based on a co-occurance frequency matrix
def plot_heatmap(data, neuron_series, spikeData, save_dir, filename, trial_comb):
    # Plot the resulting superimposition of the matrices
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', cmap='OrRd', interpolation='nearest')
    plt.colorbar(label='Frequency of co-occurence in the same component')

    plt.title(f'A heatmap showing the superimposed co-occurence matrices\n{trial_comb}')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')
    plt.xticks(ticks=range(len(neuron_series)), labels=neuron_series, rotation="vertical")
    plt.yticks(ticks=range(len(neuron_series)), labels=neuron_series)

    # Color the y-axis labels based on which brain area they belong to
    tick_labels = [plt.gca().get_xticklabels(), plt.gca().get_yticklabels()]
    label_colors = ["blue", "green", "magenta"]
    for labels in tick_labels:
        for tick_label in labels:
            neuron_ID = tick_label.get_text()
            row_index = spikeData.index[spikeData["cell_ID"] == neuron_ID].to_list()
            area = spikeData.at[row_index[0], "area"]
            if area == "V1":
                tick_label.set_color(label_colors[0])
            elif area == "CG1":
                tick_label.set_color(label_colors[1])
            else:
                tick_label.set_color(label_colors[2])

    # Add an additional legend to explain the different brain areas
    area_labels = ["V1", "CG1", "PPC"]
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(label_colors, area_labels)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='best', title="Brain Area")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    plt.savefig(f"{save_dir}/{filename}.png")


# Getting the saved binarized data
time_bin = "10"
save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
spike_file = f"binSpikeTrials_{time_bin}ms.pkl"
trialBinData = pd.read_pickle(f"{save_dir}/{spike_file}")


# Getting the session data
min_fire = 0.5
quality = 'good'
path_root = Path("/Users/vojtamazur/Documents/Capstone_code")
experiment = ["ChangeDetectionConflict"]

trialData, sessionData, spikeData = utils.load_data(path_root, experiment)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)


# Setting up some variables
data_dir = "./binData"
heatmap_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/{time_bin}ms"

logE_list = []
logL_list = []
partition_li = []
n_list = []
r_list = []
session_list = []
trial_comb_list = []
comp_size_list = []


# # Create an MCM for all trials in a certain stimulus combination before change
# # and save them all in a dataframe (in a file)
# for index, session in sessionData.iterrows():
#     # get the trials from this session
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     # find the best MCM for all trials with one stimulus combination
#     for visGroup in ses_trials.visGroupPreChange.unique():
#         for audioGroup in ses_trials.audioGroupPreChange.unique():
#             comb_trials = ses_trials[
#                 (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#             ]

#             # generate the data file for MCM
#             filename = f"visual{visGroup}_audio{audioGroup}"
#             create_input_file(comb_trials, 99, 299, filename, data_dir)

#             # finding the best MCM
#             data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)
#             MCM_best = mod.MCM_GreedySearch(data, n, False)

#             # Calculate the Log evidence of the MCM and add it to the list
#             LogE = mod.LogE_MCM(data, MCM_best, MCM_best.r)
#             logE_list.append(LogE)

#             # Calculate the log likelihood of the MCM and add it to the list
#             LogL = mod.LogL_MCM(data, MCM_best, MCM_best.r)
#             logL_list.append(LogL)

#             # Add other relevant values to the columns
#             partition_li.append(MCM_best.array)
#             n_list.append(n)
#             r_list.append(MCM_best.r)
#             session_list.append(ses_ID)
#             trial_comb_list.append([visGroup, audioGroup])
#             comp_size_list.append(count_component_size(MCM_best.array, n))

#             print(f"{visGroup} and {audioGroup} combination finished")


# data_dict = {
#     "Session_ID": session_list,
#     "Stimulus combination": trial_comb_list,
#     "Partition array": partition_li,
#     "No. of variables": n_list,
#     "r": r_list,
#     "Log evidence": logE_list,
#     "Log likelihood": logL_list,
#     "Component sizes": comp_size_list
# }

# beforeChMCM_data = pd.DataFrame(data_dict)
# save_dir = "/Users/vojtamazur/Documents/Capstone_code/MCM_results"
# beforeChMCM_data.to_pickle(f"{save_dir}/before_change_{time_bin}ms.pkl")

########################################################################################################################################################################


# beforeChMCM_data = pd.read_pickle(f"/Users/vojtamazur/Documents/Capstone_code/MCM_results/before_change_{time_bin}ms.pkl")
# for _, row in beforeChMCM_data.iterrows():
#     # get the session neurons and spike data
#     ses_ID = row["Session_ID"]
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]

#     save_dir = f"/Users/vojtamazur/Documents/Capstone_code/MCM_results/{time_bin}ms/plots_ses_{ses_ID}"
#     visGroup, audioGroup = row["Stimulus combination"]
#     n = row["No. of variables"]

#     neuron_matrix = generate_coocurrance_matrix(row["Partition array"], n)
#     for i in range(n):
#         neuron_matrix[i][i] = 0

#     plot_heatmap(
#         neuron_matrix,
#         neuron_series,
#         ses_neurons,
#         save_dir,
#         f"stim_comb_{visGroup}-{audioGroup}_co-occurence",
#         f"A visual representation of the co-occurence matrix for\nthe stimulus combination of {visGroup} degree line and {audioGroup} Hz frequency\nin session {ses_ID} ({time_bin}ms time bins)"
#         )

# for index, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_MCMs = beforeChMCM_data[beforeChMCM_data["Session_ID"] == ses_ID]

#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     superimposed_matrix = np.zeros((n, n))

#     for _, row in ses_MCMs.iterrows():
#         neuron_matrix = generate_coocurrance_matrix(row["Partition array"], row["No. of variables"])
#         superimposed_matrix += neuron_matrix

#     for i in range(n):
#         superimposed_matrix[i][i] = 0

#     save_dir = f"/Users/vojtamazur/Documents/Capstone_code/MCM_results/plots_superimposed_{time_bin}ms"

#     plot_heatmap(
#         superimposed_matrix,
#         neuron_series,
#         ses_neurons,
#         save_dir,
#         f"session_{index+1}-{ses_ID}_co-occurence_heatmap",
#         f"for all stimulus combinations in session {index+1} ({ses_ID})\nfor the duration before the stimulus change ({time_bin}ms time bins)"
#         )


# for _, row in beforeChMCM_data.iterrows():
#     sizes_full = row["Component sizes"]
#     comp_sizes = count_component_size(row["Partition array"], row["No. of variables"], True)

#     print(row["Session_ID"])
#     print(row["Stimulus combination"])
#     print(np.mean(comp_sizes))
#     # print(len(comp_sizes)/len(sizes_full))
#     # print(np.max(comp_sizes))
#     # print(np.min(comp_sizes))
#     print("\n")


########################################################################################################################################################################

# # Create and plot a superimposed co-occurrence matrix
# # for every combination of stimuli before change in each session
# for index, session in sessionData.iterrows():
#     # get the trials from this session
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     for visGroup in ses_trials.visGroupPreChange.unique():
#         for audioGroup in ses_trials.audioGroupPreChange.unique():
#             comb_trials = ses_trials[
#                 (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#             ]
#             superimposed_matrix = np.zeros((n, n))

#             # Iterating through every trial and generating the best
#             # MCM for each trial
#             for _, trial in comb_trials.iterrows():

#                 # Converting the data into a format usable by the MCM
#                 filename = f"session{ses_ID}_trial{trial['trialNum']}"
#                 create_input_file(trial, 99, 299, filename, data_dir)

#                 data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

#                 # Creating the MCM
#                 MCM_best = mod.MCM_GreedySearch(data, n, False)

#                 # Calculate the Log evidence of the MCM and add it to the list
#                 LogE = mod.LogE_MCM(data, MCM_best, MCM_best.r)
#                 logE_list.append(LogE)

#                 # Calculate the log likelihood of the MCM and add it to the list
#                 LogL = mod.LogL_MCM(data, MCM_best, MCM_best.r)
#                 logL_list.append(LogL)

#                 # Generate the co-ocurrence matrix for the model
#                 co_matrix = generate_coocurrance_matrix(MCM_best.array, n)

#                 # print(np.array2string(co_matrix, threshold=np.inf))
#                 superimposed_matrix += co_matrix

#             for i in range(n):
#                 superimposed_matrix[i][i] = 0

#             plot_heatmap(
#                 superimposed_matrix,
#                 neuron_series,
#                 ses_neurons,
#                 f"{heatmap_dir}/session_{index+1}-{ses_ID}",
#                 f"stim{visGroup}_{audioGroup}",
#                 f"session {index+1} trials ({time_bin}ms time bin), stimulus combination before change:\nvisual {visGroup} degrees, auditory {audioGroup} ({len(comb_trials)} trials)"
#             )


########################################################################################################################################################################

plt_save_dir = f"/Users/vojtamazur/Documents/Capstone_code/comp_sizes/{time_bin}ms"

# Calculate how the average component size increases with the number of trials included
for index, session in sessionData.iterrows():
    # get the trials from this session
    ses_ID = session["session_ID"]
    ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

    # get the neurons from the session and their number
    ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
    neuron_series = ses_neurons["cell_ID"]
    n = len(neuron_series)

    # Iterating through each stimulus combination for valid comparisons
    for visGroup in ses_trials.visGroupPreChange.unique():
        for audioGroup in ses_trials.audioGroupPreChange.unique():
            # Get the trials in this stimulus comnbination
            comb_trials = ses_trials[
                (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
            ]

            comp_size_averages = []
            trial_num = len(comb_trials)

            # Loop through each number of possible trials and calculate the average component size
            for trial_n in range(1, trial_num):
                MCM_trials = comb_trials.iloc[0:trial_n, :]

                # Converting the data into a format usable by the MCM
                filename = f"session{ses_ID}_trial{trial_n}"
                create_input_file(MCM_trials, 99, 299, filename, data_dir)
                data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

                # Creating the MCM
                MCM_best = mod.MCM_GreedySearch(data, n, False)

                # Calculate the average component size
                avg_comp_size_full = np.mean(count_component_size(MCM_best.array, n, False))
                avg_comp_size_excl = np.mean(count_component_size(MCM_best.array, n, True))
                comp_size_averages.append((avg_comp_size_full, avg_comp_size_excl))

            if not os.path.exists(plt_save_dir):
                os.makedirs(plt_save_dir)

            plt.figure(figsize=(6, 6))
            plt.scatter(range(1, trial_num), [x[0] for x in comp_size_averages])
            plt.title(f"Mean component size of the MCM plotted against the\nnumber of trials used in training\nStimulus combination: visual {visGroup}°, auditory {audioGroup}Hz")
            plt.xlabel('No. of trials used for the MCM')
            plt.ylabel('Mean component size')
            plt.savefig(f"{plt_save_dir}/vis{visGroup}_aud{audioGroup}_comp_sizes_plot.png")


