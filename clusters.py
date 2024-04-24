import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import models
import MinCompSpin_Python.MinCompSpin as mod
import matplotlib.pyplot as plt

# Imports for getting the session data
import utils
from pathlib import Path
import warnings

# Getting the session data
min_fire = 0.5
quality = 'good'
path_root = Path("/Users/vojtamazur/Documents/Capstone_code")
experiment = ["ChangeDetectionConflict"]

trialData, sessionData, spikeData = utils.load_data(path_root, experiment)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)


save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/"
# clusterDF = pd.read_pickle(f"{save_dir}bc_clusters_concat_trials.pkl")
data_dir = "./binData"

def calc_cluster_comparison(cluster_ser, comp_metric):
    result_arr = np.zeros([len(cluster_ser), len(cluster_ser)])
    reset_indexDF = cluster_ser.reset_index()
    cluster_ser = reset_indexDF["clusters"]

    for i, clusters1 in cluster_ser.items():
        for j, clusters2 in cluster_ser.items():
            result_arr[int(i)][int(j)] = comp_metric(clusters1, clusters2)

    return result_arr


means_ari = []
means_nmi = []

# print(clusterDF.loc[0, "clusters"])
# for visGroup in clusterDF.visGroup.unique():
#     for audioGroup in clusterDF.audioGroup.unique():
#         comb_clusterDF = clusterDF[(clusterDF["visGroup"] == visGroup) & (clusterDF["audioGroup"] == audioGroup)]
#         cluster_ser = comb_clusterDF["clusters"]


#         # calculate comparison indices
#         ari_score = calc_cluster_comparison(cluster_ser, adjusted_rand_score)
#         nmi_score = calc_cluster_comparison(cluster_ser, normalized_mutual_info_score)

#         where_matrix = np.full(ari_score.shape, True)
#         for i in range(len(ari_score)):
#             for j in range(len(ari_score)):
#                 if i == j:
#                     where_matrix[i][j] = False
#         print(where_matrix)

#         # print the scores
#         # print(f"For the stimulus combination {visGroup} degrees and {audioGroup} Hz:")
#         # print(f"ARI score: {np.mean(ari_score)}")
#         # print(f"NMI score: {np.mean(nmi_score)}\n")

#         ari_mean = np.mean(ari_score, where=where_matrix)
#         nmi_mean = np.mean(nmi_score, where=where_matrix)


#         if not math.isnan(ari_mean):
#             means_ari.append(ari_mean)
#             means_nmi.append(nmi_mean)


# for session in clusterDF.session_ID.unique():
#     means_ari = []
#     means_nmi = []

#     for t_bin in clusterDF.time_bin.unique():
#         ses_clusterDF = clusterDF[(clusterDF["session_ID"] == session) & (clusterDF["time_bin"] == t_bin)]
#         cluster_ser = ses_clusterDF["clusters"]

#         # calculate comparison indices
#         ari_score = calc_cluster_comparison(cluster_ser, adjusted_rand_score)
#         nmi_score = calc_cluster_comparison(cluster_ser, normalized_mutual_info_score)

#         where_matrix = np.full(ari_score.shape, True)
#         for i in range(len(ari_score)):
#             for j in range(len(ari_score)):
#                 if i == j:
#                     where_matrix[i][j] = False

#         # print the scores
#         # print(f"For session {session} with {t_bin}s time bins:")

#         ari_mean = np.mean(ari_score, where=where_matrix)
#         nmi_mean = np.mean(nmi_score, where=where_matrix)
#         # print(f"ARI score: {ari_mean}")
#         # print(f"NMI score: {nmi_mean}\n")

#         if not math.isnan(ari_mean):
#             means_ari.append(ari_mean)
#             means_nmi.append(nmi_mean)

#     print(f"Session {session}:")
#     print(f"Mean ARI score: {np.mean(means_ari)}")
#     print(f"Mean NMI score: {np.mean(means_nmi)}\n")


# print(f"Mean ARI score: {np.mean(means_ari)}")
# print(f"Mean NMI score: {np.mean(means_nmi)}")


########################################################################################################################################################################
######################################## Plotting stimulus combination dengrograms reordered based on session-wide culstering #################################################

# time_bins = [5, 10, 15, 20, 25, 30]

# # Bootstrap concatenations of enough trials to get 1000 data points
# time_bins = [5, 10, 15, 20, 25, 30]
# min_data_size = 1500
# sample_count = 30

# fname = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data/bc_clusters_full_ses.pkl"
# linkageDF = pd.read_pickle(fname)

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

#         ses_linkageDF = linkageDF[(linkageDF["session_ID"] == ses_ID) & (linkageDF["time_Bin"] == time_bin)]

#         reordered_series = ses_linkageDF["reordered_Series"].item()

#         for visGroup in ses_trials.visGroupPreChange.unique():
#             for audioGroup in ses_trials.audioGroupPreChange.unique():
#                 comb_trials = ses_trials[
#                     (ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)
#                 ]
#                 superimposed_matrix = np.zeros((n, n))

#                 for i in range(sample_count):
#                     trial_sample = comb_trials.sample(n=sample_size)

#                     # Converting the data into a format usable by the MCM
#                     filename = f"session{ses_ID}_concat_{time_bin}ms"
#                     models.create_input_file(trial_sample, 0, int(2000/time_bin)-1, filename, data_dir)

#                     data = mod.read_datafile(f"{data_dir}/{filename}.dat", n)

#                     # Creating the MCM
#                     MCM_best = mod.MCM_GreedySearch(data, n, False)

#                     # Generate the co-ocurrence matrix for the model
#                     co_matrix = models.generate_coocurrance_matrix(MCM_best.array, n)
#                     superimposed_matrix += co_matrix

#                 heatmap_dir = f"/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/full_ses_cluster_reorder/{time_bin}ms/{index+1}_{ses_ID}"
#                 description = f"Session {index + 1} ({ses_ID}), ({time_bin}ms time bin)\nStimulus combination {visGroup}° and {audioGroup}Hz"
#                 fname = f"{min_data_size}vis{visGroup}_audio{audioGroup}"

#                 reordered_matrix = models.reorder_matrix(reordered_series, neuron_series, superimposed_matrix)
#                 models.plot_heatmap(
#                     reordered_matrix,
#                     reordered_series,
#                     ses_neurons,
#                     heatmap_dir,
#                     f"{fname}_reordered",
#                     f"{description}\nReordered based on hierarchical clustering from the full session data"
#                 )

#                 models.plot_heatmap(
#                     superimposed_matrix,
#                     neuron_series,
#                     ses_neurons,
#                     heatmap_dir,
#                     fname,
#                     description
#                 )


########################################################################################################################################################################
######################################## Plotting how the average cluster size varies with the time bin ####################################################################

time_bins = [5, 10, 15, 20, 25, 30]
mean_component_sizes = []
mean_comp_st_deviations = []

mean_cluster_sizes = []

fname = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data/bc_clusters_full_ses.pkl"
linkageDF = pd.read_pickle(fname)


for time_bin in time_bins:

    component_means = []
    component_st_devs = []
    for index, session in sessionData.iterrows():
        ses_ID = session["session_ID"]

        # get the neurons from the session and their number
        ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
        neuron_series = ses_neurons["cell_ID"]
        n = len(neuron_series)

        ses_linkageDF = linkageDF[(linkageDF["session_ID"] == ses_ID) & (linkageDF["time_Bin"] == time_bin)]
        comp_sizes = [models.count_component_size(x, n, True) for x in ses_linkageDF["MCM_Partition"].item()]
        means = [np.mean(x) for x in comp_sizes]
        stdevs = [np.std(x) for x in comp_sizes]
        component_means.append(np.mean(means))
        [component_st_devs.append(x) for x in stdevs]

    mean_cluster_sizes.append(component_means)
    mean_comp_st_deviations.append(np.mean(component_st_devs))


# plt.scatter(time_bins, mean_cluster_sizes)
# plt.errorbar(time_bins, mean_cluster_sizes, yerr=mean_comp_st_deviations, fmt="o")

fig = plt.figure(figsize=(10,7))

# Creating an axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(mean_cluster_sizes)

plt.title("Relationship between the time bin size and the average MCM\n component size (after excluding components of size 1)")
plt.xlabel("Time bin (ms)")
plt.ylabel("Mean component size")
# plt.show()

plt.savefig("/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/comp_size_box_plt_1_excluded.png")
