import numpy as np
import pandas as pd
import MinCompSpin_Python.MinCompSpin as mod
import raster_plots as rplt
import os
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from collections import defaultdict

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
                if (((component[i] == "1") & (component[j] == "1")) & (i != j)):
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
def plot_heatmap(data, neuron_series, spikeData, save_dir, filename, trial_comb=None):
    # Plot the resulting superimposition of the matrices
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', cmap='OrRd', interpolation='nearest')
    plt.colorbar(label='Frequency of co-occurence in the same component')

    if trial_comb:
        plt.title(
            f'A heatmap showing the superimposed co-occurence matrices\n{trial_comb}',
            fontsize="16"
            )
    else:
        plt.title(
            f'A heatmap showing the superimposed co-occurence matrices',
            fontsize="16"
            )

    plt.xlabel('Neuron Index', fontsize="12")
    plt.ylabel('Neuron Index', fontsize="12")
    plt.xticks(ticks=range(len(neuron_series)), labels=neuron_series, rotation="vertical", fontsize="6")
    plt.yticks(ticks=range(len(neuron_series)), labels=neuron_series, fontsize="6")

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
    area_labels = ["V1", "AC1", "PPC"]
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(label_colors, area_labels)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='best', title="Brain Area", fontsize="12")

    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(left=0.2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    plt.savefig(f"{save_dir}/{filename}.png")



def get_linkage_matrix(co_occurence_matrix):
    # Convert to distance matrix
    max_similarity = np.max(co_occurence_matrix)
    distance_matrix = max_similarity - co_occurence_matrix

    for i in range(n):
        distance_matrix[i][i] = 0

    # Scaling the distance matrix so that its values are between 1 and 0
    scaler = MinMaxScaler()
    distance_matrix_norm = scaler.fit_transform(distance_matrix)

    # Compute linkage matrix
    linkage_matrix = sch.linkage(sch.distance.squareform(distance_matrix_norm), method='ward')

    return linkage_matrix


def plot_dendrogram(linkage_matrix, neuron_ids, spikeData, save_dir, filename):
    # Plot dendrogram
    plt.figure(figsize=(6, 4))
    dendrogram = sch.dendrogram(linkage_matrix, color_threshold=1.25, labels=neuron_ids, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Neuron Index')
    plt.ylabel('Distance')

    # Extract the neuron labels into a list so that they can be exactly clustered in the re-arranged matrix
    neuron_list = []

    # Color the y-axis labels based on which brain area they belong to
    tick_labels = plt.gca().get_xticklabels()
    label_colors = ["blue", "green", "magenta"]
    for tick_label in tick_labels:
        neuron_ID = tick_label.get_text()
        row_index = spikeData.index[spikeData["cell_ID"] == neuron_ID].to_list()
        area = spikeData.at[row_index[0], "area"]
        if area == "V1":
            tick_label.set_color(label_colors[0])
        elif area == "CG1":
            tick_label.set_color(label_colors[1])
        else:
            tick_label.set_color(label_colors[2])

        neuron_list.append(neuron_ID)

    # Add an additional legend to explain the different brain areas
    area_labels = ["V1", "AC1", "PPC"]
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(label_colors, area_labels)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc='upper left', title="Brain Area")

    plt.subplots_adjust(bottom=0.3)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    # plt.savefig(f"{save_dir}/{filename}.png")

    return pd.Series(neuron_list)


# A function to plot a heatmap based on a co-occurance frequency matrix
# with a transformed data basis
def plot_trans_basis(data, save_dir, filename, trial_comb):
    # Plot the resulting superimposition of the matrices
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', cmap='OrRd', interpolation='nearest')
    plt.colorbar(label='Frequency of co-occurence in the same component')

    plt.title(f'A heatmap showing the superimposed co-occurence matrices\n{trial_comb}')
    plt.xlabel('Basis element')
    plt.ylabel('Basis element')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    plt.savefig(f"{save_dir}/{filename}.png")

def reorder_matrix(reordered_neurons, original_neurons, co_occurrence_matrix):
    # Map original neuron IDs to their indices
    neuron_index_map = {neuron_id: idx for idx, neuron_id in enumerate(original_neurons)}

    # Find the indices in the original matrix that correspond to the reordered neurons
    reordered_indices = [neuron_index_map[neuron] for neuron in reordered_neurons]

    # Reorder the co-occurrence matrix accordingly
    reordered_matrix = co_occurrence_matrix[np.ix_(reordered_indices, reordered_indices)]

    return reordered_matrix







# Getting the saved binarized data
time_bin = 50
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
# heatmap_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/{time_bin}ms"

logE_list = []
logL_list = []
partition_li = []
n_list = []
r_list = []
session_list = []
trial_comb_list = []
comp_size_list = []


# for index, session in sessionData.iterrows():
#     # get the trials from this session
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     # generate the data file for MCM
#     filename = f"session{ses_ID}_full_50ms"
#     create_input_file(ses_trials, 0, 199, filename, data_dir)



########################################################################################################################################################################

########################################## MCMs of CONCATENATED TRIALS for entire sessions #########################################################################################################

# # Initializing the variables for a dataframe:
# session_list = []
# reordered_series_list = []
# partition_list = []
# linkage_list = []
# bin_list = []

# # Bootstrap concatenations of enough trials to get 1000 data points
# time_bins = [5, 10, 15, 20, 25, 30]
# min_data_size = 1500
# sample_count = 30

# for time_bin in time_bins:
#     # Getting the saved binarized data
#     spike_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
#     spike_file = f"binSpikeTrials_{time_bin}ms.pkl"
#     trialBinData = pd.read_pickle(f"{spike_dir}/{spike_file}")

#     # Calculate the number of concatenated trials necessary for a data size of 1000
#     sample_size = math.ceil(time_bin*(min_data_size/2000))

#     for index, session in sessionData.iterrows():
#         # get the trials from this session
#         ses_ID = session["session_ID"]
#         ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#         # get the neurons from the session and their number
#         ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#         neuron_series = ses_neurons["cell_ID"]
#         n = len(neuron_series)
#         neuron_ids = neuron_series.to_list()

#         superimposed_matrix = np.zeros((n, n))
#         partitions = []
#         for i in range(sample_count):
#             sampleTrials = pd.DataFrame(columns=trialBinData.columns)
#             for visGroup in ses_trials.visGroupPreChange.unique():
#                 for audioGroup in ses_trials.audioGroupPreChange.unique():
#                     combTrials = ses_trials[
#                         (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#                     ]
#                     combSample = combTrials.sample(n=sample_size)

#                     sampleTrials = pd.concat([sampleTrials, combSample])

#             # Converting the data into a format usable by the MCM
#             filename = f"session{ses_ID}_concat_{time_bin}ms"
#             create_input_file(sampleTrials, 0, int(2000/time_bin)-1, filename, data_dir)

#             data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

#             # Creating the MCM
#             MCM_best = mod.MCM_GreedySearch(data, n, False)

#             # Generate the co-ocurrence matrix for the model
#             co_matrix = generate_coocurrance_matrix(MCM_best.array, n)
#             superimposed_matrix += co_matrix

#             partitions.append(MCM_best.array)

#         # Setting up variables necessary for the heatmap + dendrogram creation
#         heatmap_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/concat_trials/{min_data_size}_data_points/{time_bin}ms"
#         dendrogram_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/dendrograms/{min_data_size}_data_points/{time_bin}ms"
#         description = f"Session {index + 1} ({ses_ID}), ({time_bin}ms time bin)"
#         fname = f"session{index+1}-{ses_ID}_full"

#         # Hierarchically cluster the data and plot the dendrogram
#         linkage_matrix = get_linkage_matrix(superimposed_matrix)
#         reordered_series = plot_dendrogram(
#             linkage_matrix,
#             neuron_ids,
#             spikeData,
#             dendrogram_dir,
#             f"{fname}_concat_trials"
#         )

#         # Plot the superimposed co-occurence matrices
#         reordered_matrix = reorder_matrix(reordered_series, neuron_series, superimposed_matrix)

#         plot_heatmap(
#             reordered_matrix,
#             reordered_series,
#             ses_neurons,
#             heatmap_dir,
#             f"{fname}_reordered",
#             description
#         )

#         plot_heatmap(
#             superimposed_matrix,
#             neuron_series,
#             ses_neurons,
#             heatmap_dir,
#             fname,
#             description
#         )

#         session_list.append(ses_ID)
#         bin_list.append(time_bin)
#         reordered_series_list.append(reordered_series.to_list())
#         partition_list.append(partitions)
#         linkage_list.append(linkage_matrix)


# data = {
#     "session_ID": session_list,
#     "time_Bin": bin_list,
#     "reordered_Series": reordered_series_list,
#     "MCM_Partition": partition_list,
#     "linkage_Matrix": linkage_list
# }

# save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data"
# clusterDF = pd.DataFrame(data)
# print(clusterDF)
# clusterDF.to_pickle(f"{save_dir}/bc_clusters_full_ses.pkl")

########################################################################################################################################################################

############################# MCMs of CONCATENATED TRIALS for unique stimulus combinations #########################################################################################################

# # Initializing the variables for a dataframe:
# session_list = []
# reordered_series_list = []
# partition_list = []
# linkage_list = []
# bin_list = []
# vis_group_list = []
# audio_group_list = []


# # Bootstrap concatenations of enough trials to get 1000 data points
# time_bins = [5, 10, 15, 20, 25, 30]
# min_data_size = 1500
# sample_count = 30

# for time_bin in time_bins:
#     # Getting the saved binarized data
#     spike_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
#     spike_file = f"binSpikeTrials_{time_bin}ms.pkl"
#     trialBinData = pd.read_pickle(f"{spike_dir}/{spike_file}")

#     # Calculate the number of concatenated trials necessary for a data size of 1000
#     sample_size = math.ceil(time_bin*(min_data_size/2000))

#     for index, session in sessionData.iterrows():
#         # get the trials from this session
#         ses_ID = session["session_ID"]
#         ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#         # get the neurons from the session and their number
#         ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#         neuron_series = ses_neurons["cell_ID"]
#         n = len(neuron_series)
#         neuron_ids = neuron_series.to_list()

#         # Iterate through every combination of visual and audio stimuli
#         for visGroup in ses_trials.visGroupPreChange.unique():
#             for audioGroup in ses_trials.audioGroupPreChange.unique():
#                 combTrials = ses_trials[
#                     (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#                 ]

#                 # Set up the superimposed matrix and partitions list
#                 superimposed_matrix = np.zeros((n, n))
#                 partitions = []

#                 for i in range(sample_count):
#                     # Obtaing a sample of the trials
#                     sampleTrials = combTrials.sample(n=sample_size)

#                     # Converting the data into a format usable by the MCM
#                     filename = f"session{ses_ID}_concat_{time_bin}ms"
#                     create_input_file(sampleTrials, 0, int(2000/time_bin)-1, filename, data_dir)

#                     data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

#                     # Creating the MCM
#                     MCM_best = mod.MCM_GreedySearch(data, n, False)

#                     # Generate the co-ocurrence matrix for the model
#                     co_matrix = generate_coocurrance_matrix(MCM_best.array, n)
#                     superimposed_matrix += co_matrix

#                     partitions.append(MCM_best.array)

#                 # Setting up variables necessary for the heatmap + dendrogram creation
#                 dendrogram_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/dendrograms/{min_data_size}_data_points/{time_bin}ms"
#                 description = f"Session {index + 1} ({ses_ID}), ({time_bin}ms time bin)\nStimuli before change: {visGroup}° and {audioGroup}Hz"
#                 fname = f"session{index+1}-{ses_ID}_full"

#                 # Hierarchically cluster the data and plot the dendrogram
#                 linkage_matrix = get_linkage_matrix(superimposed_matrix)
#                 reordered_series = plot_dendrogram(
#                     linkage_matrix,
#                     neuron_ids,
#                     spikeData,
#                     dendrogram_dir,
#                     f"{fname}_concat_trials"
#                 )

#                 session_list.append(ses_ID)
#                 bin_list.append(time_bin)
#                 reordered_series_list.append(reordered_series.to_list())
#                 partition_list.append(partitions)
#                 linkage_list.append(linkage_matrix)
#                 vis_group_list.append(visGroup)
#                 audio_group_list.append(audioGroup)

# # Save the data in a dataframe as a pickle file
# data = {
#     "session_ID": session_list,
#     "time_Bin": bin_list,
#     "reordered_Series": reordered_series_list,
#     "MCM_Partition": partition_list,
#     "linkage_Matrix": linkage_list,
#     "visGroup": vis_group_list,
#     "audioGroup": audio_group_list
# }

# save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data"
# clusterDF = pd.DataFrame(data)
# print(clusterDF)

# clusterDF.to_pickle(f"{save_dir}/bc_clusters_trial_combs.pkl")
# print(clusterDF["MCM_Partition"])
# print(clusterDF["MCM_Partition"][0])


########################################################################################################################################################################

##################### Calculate how the average component size increases with the number of trials included ################################################################################

# plt_save_dir = f"/Users/vojtamazur/Documents/Capstone_code/comp_sizes/{time_bin}ms"


# for index, session in sessionData.iterrows():
#     # get the trials from this session
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     # Iterating through each stimulus combination for valid comparisons
#     for visGroup in ses_trials.visGroupPreChange.unique():
#         for audioGroup in ses_trials.audioGroupPreChange.unique():
#             # Get the trials in this stimulus comnbination
#             comb_trials = ses_trials[
#                 (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#             ]

#             comp_size_averages = []
#             trial_num = len(comb_trials)

            # # Loop through each number of possible trials and calculate the average component size
            # for trial_n in range(1, trial_num):
            #     MCM_trials = comb_trials.iloc[0:trial_n, :]

            #     # Converting the data into a format usable by the MCM
            #     filename = f"session{ses_ID}_trial{trial_n}"
            #     create_input_file(MCM_trials, 99, 299, filename, data_dir)
            #     data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

            #     # Creating the MCM
            #     MCM_best = mod.MCM_GreedySearch(data, n, False)

            #     # Calculate the average component size
            #     avg_comp_size_full = np.mean(count_component_size(MCM_best.array, n, False))
            #     avg_comp_size_excl = np.mean(count_component_size(MCM_best.array, n, True))
            #     comp_size_averages.append((avg_comp_size_full, avg_comp_size_excl))

            # if not os.path.exists(plt_save_dir):
            #     os.makedirs(plt_save_dir)

            # plt.figure(figsize=(6, 6))
            # plt.scatter(range(1, trial_num), [x[0] for x in comp_size_averages])
            # plt.title(f"Mean component size of the MCM plotted against the\nnumber of trials used in training\nStimulus combination: visual {visGroup}°, auditory {audioGroup}Hz")
            # plt.xlabel('No. of trials used for the MCM')
            # plt.ylabel('Mean component size')
            # plt.savefig(f"{plt_save_dir}/vis{visGroup}_aud{audioGroup}_comp_sizes_plot.png")

########################################################################################################################################################################


# # Create and visualize the MCMs in the best basis

# directory = "./bestBasisData/transformedData"

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)

#     # Get the number of variables from the file
#     with open(f, "r") as file:
#         first_line = file.readline()
#         n = len(first_line)

#     # Find the best MCM for the data
#     data = mod.read_datafile(f, n)
#     MCM_best = mod.MCM_GreedySearch(data, n, False)

#     # Generate the co-ocurrence matrix for the model
#     co_matrix = generate_coocurrance_matrix(MCM_best.array, n)

#     save_dir = "/Users/vojtamazur/Documents/Capstone_code/bestBasis/10ms"
#     trial_comb = filename[:(len(filename)-26)]

#     # Plot the co-occurence matrix and save it
#     plot_trans_basis(co_matrix, save_dir, trial_comb, trial_comb)




########################################################################################################################################################################

############################# MCMs of CONCATENATED TRIALS for 20ms for LogE-LogE plots #########################################################################################################

time_bin = 20

# Getting the saved binarized data
spike_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
spike_file = f"binSpikeTrials_{time_bin}ms.pkl"
trialBinData = pd.read_pickle(f"{spike_dir}/{spike_file}")

# Calculate the number of concatenated trials necessary for a data size of 1000
sample_size = 15
sample_count = 30

session_list = []
Tset_list = []
partition_list = []
linkage_list = []
vis_group_list = []
audio_group_list = []
log_evidences = []
log_likelihoods =[]

for index, session in sessionData.iterrows():
    # get the trials from this session
    ses_ID = session["session_ID"]
    ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

    # get the neurons from the session and their number
    ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
    neuron_series = ses_neurons["cell_ID"]
    n = len(neuron_series)

     # Iterate through every combination of visual and audio stimuli
    for visGroup in ses_trials.visGroupPreChange.unique():
        for audioGroup in ses_trials.audioGroupPreChange.unique():
            combTrials = ses_trials[
                (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
            ]

            # Split the combination trial data in 2 equal parts
            midway_point = len(combTrials) // 2
            combTrials1 = combTrials.iloc[:midway_point]
            combTrials2 = combTrials.iloc[midway_point:]

            # Check if after the split there are still enough trials for bootstrap sampling
            if len(combTrials1) >= 20:
                for t_set, trials in zip([1, 2], [combTrials1, combTrials2]):

                    # Set up the superimposed matrix and partitions list
                    superimposed_matrix = np.zeros((n, n))
                    partitions = []
                    logE_list = []
                    logL_list = []

                    for i in range(sample_count):
                        # Obtaing a sample of the trials
                        sampleTrials = trials.sample(n=sample_size)

                        # Converting the data into a format usable by the MCM
                        filename = f"session{ses_ID}_concat_{time_bin}ms"
                        create_input_file(sampleTrials, 0, int(2000/time_bin)-1, filename, data_dir)

                        data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

                        # Creating the MCM
                        MCM_best = mod.MCM_GreedySearch(data, n, False)

                        logE = mod.LogE_MCM(data, MCM_best, MCM_best.r)
                        logL = mod.LogL_MCM(data, MCM_best, MCM_best.r)

                        # Generate the co-ocurrence matrix for the model
                        co_matrix = generate_coocurrance_matrix(MCM_best.array, n)
                        superimposed_matrix += co_matrix

                        partitions.append(MCM_best.array)
                        logE_list.append(logE)
                        logL_list.append(logL)

                    # Hierarchically cluster the data and plot the dendrogram
                    linkage_matrix = get_linkage_matrix(superimposed_matrix)
                    reordered_series = plot_dendrogram(
                        linkage_matrix,
                        neuron_ids,
                        spikeData,
                        "",
                        ""
                    )

                    session_list.append(ses_ID)
                    Tset_list.append(t_set)
                    partition_list.append(partitions)
                    linkage_list.append(linkage_matrix)
                    vis_group_list.append(visGroup)
                    audio_group_list.append(audioGroup)
                    log_evidences.append(np.array(logE_list))
                    log_likelihoods.append(np.array(logL_list))


# Save the data in a dataframe as a pickle file
data = {
    "session_ID": session_list,
    "trial_set": Tset_list,
    "MCM_partition": partition_list,
    "linkage_matrix": linkage_list,
    "visGroup": vis_group_list,
    "audioGroup": audio_group_list,
    "logE_array": log_evidences,
    "logL_array": log_likelihoods
}

save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data"
clusterDF = pd.DataFrame(data)
print(clusterDF)

clusterDF.to_pickle(f"{save_dir}/bc_20ms_2set_MCM_data.pkl")






#######################################################################################################################################################################

# # Plotting the heatmaps to be used in the Capstone text


# visGroup, audioGroup = ["225-230", "13000-13020"]
# comb_trials = trialBinData[(trialBinData["visGroupPreChange"] == visGroup) & (trialBinData["audioGroupPreChange"] == audioGroup)]

# ses_neurons = spikeData[spikeData["session_ID"] == "20122018081421"]
# print(ses_neurons)
# neuron_series = ses_neurons["cell_ID"]
# n = len(neuron_series)

# superimposed_matrix = np.zeros((n, n))


# min_data_size = 1500
# sample_size = math.ceil(time_bin*(min_data_size/2000))
# sample_count = 20

# for i in range(sample_count):
#     trial_sample = comb_trials.sample(n=sample_size)

#     # Converting the data into a format usable by the MCM
#     filename = f"session20122018081421_concat_{time_bin}ms"
#     create_input_file(trial_sample, 0, int(2000/time_bin)-1, filename, data_dir)

#     data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

#     # Creating the MCM
#     MCM_best = mod.MCM_GreedySearch(data, n, False)

#     # Generate the co-ocurrence matrix for the model
#     co_matrix = generate_coocurrance_matrix(MCM_best.array, n)
#     superimposed_matrix += co_matrix

# # Cluster the MCM results by hierarchical clustering and re-order the
# # neuron series and co-occurence matrix accordingly
# linkage_matrix = get_linkage_matrix(superimposed_matrix)
# threshold = 1
# clusters = fcluster(linkage_matrix, threshold, criterion="distance")

# reordered_series, reordered_matrix = group_clusters(neuron_series, superimposed_matrix, clusters)

# heatmap_dir = f"/Users/vojtamazur/Documents/Capstone_code/0_main_text_plots"
# fname = f"superimposed_{time_bin}ms"

# plot_heatmap(
#     reordered_matrix,
#     reordered_series,
#     ses_neurons,
#     heatmap_dir,
#     f"{fname}_reordered",
# )

# plot_heatmap(
#     superimposed_matrix,
#     neuron_series,
#     ses_neurons,
#     heatmap_dir,
#     fname
#                 )

# ########################################################################################################################################################################
