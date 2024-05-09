import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import models
import MinCompSpin_Python.MinCompSpin as mod
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import fcluster
from collections import Counter

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


save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data"
clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_trial_combs.pkl")

data_dir = "./binData"

def calc_cluster_comparison(cluster_ser, comp_metric):
    n = len(cluster_ser)
    result_arr = np.zeros([n, n])

    # Reset the index if it's a Pandas series
    if isinstance(cluster_ser, pd.Series):
        reset_indexDF = cluster_ser.reset_index()
        cluster_ser = reset_indexDF["clusters"]

        for i, clusters1 in cluster_ser.items():
            for j, clusters2 in cluster_ser.items():
                result_arr[int(i)][int(j)] = comp_metric(clusters1, clusters2)
    else:
        for i, clusters1 in enumerate(cluster_ser):
            for j, clusters2 in enumerate(cluster_ser):
                result_arr[int(i)][int(j)] = comp_metric(clusters1, clusters2)

    return result_arr

def count_unique_instances(input_list):
    # Count instances using Counter
    count_map = Counter(input_list)

    # Extract counts for each unique element
    unique_counts = list(count_map.values())

    return unique_counts


means_ari = []
means_nmi = []

# for ses_ID in sessionData["session_ID"]:
#     ses_clusterDF = clusterDF[clusterDF["session_ID"] == ses_ID]
#     for visGroup in ses_clusterDF.visGroup.unique():
#         for audioGroup in ses_clusterDF.audioGroup.unique():
#             comb_clusterDF = ses_clusterDF[(ses_clusterDF["visGroup"] == visGroup) & (ses_clusterDF["audioGroup"] == audioGroup)]

#             linkage_matrices = comb_clusterDF["linkage_Matrix"]
#             cluster_list = [fcluster(x, 0.7*max(x[:, 2]), criterion="distance") for x in linkage_matrices]

#             # calculate comparison indices
#             ari_score = calc_cluster_comparison(cluster_list, adjusted_rand_score)
#             nmi_score = calc_cluster_comparison(cluster_list, normalized_mutual_info_score)
#             print(f"\nFor the combination {visGroup}° and {audioGroup}Hz:")
#             print("ARI score:")
#             print(ari_score)
#             print("NMI score:")
#             print(nmi_score)

            # where_matrix = np.full(ari_score.shape, True)
            # for i in range(len(ari_score)):
            #     for j in range(len(ari_score)):
            #         if i == j:
            #             where_matrix[i][j] = False
            # print(where_matrix)

            # # print the scores
            # # print(f"For the stimulus combination {visGroup} degrees and {audioGroup} Hz:")
            # # print(f"ARI score: {np.mean(ari_score)}")
            # # print(f"NMI score: {np.mean(nmi_score)}\n")

            # ari_mean = np.mean(ari_score, where=where_matrix)
            # nmi_mean = np.mean(nmi_score, where=where_matrix)


            # if not math.isnan(ari_mean):
            #     means_ari.append(ari_mean)
            #     means_nmi.append(nmi_mean)


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
################################# Plotting stimulus combination dengrograms reordered based on session-wide culstering #################################################

time_bins = [5, 10, 15, 20, 25, 30]

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
################################## Plotting how the average cluster size varies with the time bin ####################################################################

# time_bins = [5, 10, 15, 20, 25, 30]
# mean_component_sizes = []
# mean_comp_st_deviations = []

# mean_cluster_sizes = []

# fname = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data/bc_clusters_full_ses.pkl"
# linkageDF = pd.read_pickle(fname)


# for time_bin in time_bins:

#     component_means = []
#     component_st_devs = []

#     cluster_sizes = []
#     for index, session in sessionData.iterrows():
#         ses_ID = session["session_ID"]

#         # get the neurons from the session and their number
#         ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#         neuron_series = ses_neurons["cell_ID"]
#         n = len(neuron_series)

#         ### Get the average component size for each MCM
#         ses_linkageDF = linkageDF[(linkageDF["session_ID"] == ses_ID) & (linkageDF["time_Bin"] == time_bin)]
#         comp_sizes = [models.count_component_size(x, n, True) for x in ses_linkageDF["MCM_Partition"].item()]

#         # for mcm in comp_sizes:
#         #     for component in mcm:
#         #         cluster_sizes.append(component)

#         means = [np.mean(x) for x in comp_sizes]
#         stdevs = [np.std(x) for x in comp_sizes]
#         cluster_sizes.append(np.mean(means))
#         [component_st_devs.append(x) for x in stdevs]

#     # mean_cluster_sizes.append(component_means)
#     # mean_comp_st_deviations.append(np.mean(component_st_devs))
#     mean_cluster_sizes.append(cluster_sizes)

# # plt.scatter(time_bins, mean_cluster_sizes)
# # plt.errorbar(time_bins, mean_cluster_sizes, yerr=mean_comp_st_deviations, fmt="o")

# fig = plt.figure(figsize=(10,7))

# # Creating an axes instance
# ax = fig.add_subplot(111)

# # Creating plot
# bp = ax.boxplot(mean_cluster_sizes)

# plt.title("Relationship between the time bin size and the average hierarchical clustering cluster size")
# plt.xlabel("Time bin (ms)")
# plt.ylabel("Mean cluster size")
# plt.xticks(ticks=range(7), labels=["", 5, 10, 15, 20, 25, 30])
# # plt.show()

# plt.savefig("/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/mean_comp_size_box_plt_1_excluded.png")


########################################################################################################################################################################
################################## Calculating the within and across area connections ####################################################################

# clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_trial_combs.pkl")

# within_count_series = pd.Series(0, index=clusterDF.index)
# between_count_series = pd.Series(0, index=clusterDF.index)

# # Iterate through all stimulus combinations of all sessions
# for _, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_clusterDF = clusterDF[clusterDF["session_ID"] == session["session_ID"]]

#     # get the neurons from the session and their ID number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     neuron_arr = np.array(neuron_series)
#     n = len(neuron_series)

#     for visGroup in ses_clusterDF["visGroup"].unique():
#         for audioGroup in ses_clusterDF["audioGroup"].unique():
#             for time_bin in time_bins:
#                 comb_DF = ses_clusterDF[((ses_clusterDF["visGroup"] == visGroup) & (ses_clusterDF["audioGroup"] == audioGroup)) & (ses_clusterDF["time_Bin"] == time_bin)]
#                 comb_index = comb_DF.index.item()
#                 print(comb_index)

#                 within_area_count = 0
#                 between_area_count = 0

#                 # Get the partitions and check which neurons are grouped
#                 # with others in the same or different brain area
#                 MCM_partitions = comb_DF["MCM_Partition"].item()
#                 for partition in MCM_partitions:
#                     co_matrix = models.generate_coocurrance_matrix(partition, n)
#                     for i in range(len(co_matrix)):
#                         for j in range(len(co_matrix)):
#                             if (co_matrix[i][j] == 1) & (i != j):
#                                 neuron_i = neuron_arr[i]
#                                 neuron_j = neuron_arr[j]

#                                 area_i = spikeData[spikeData["cell_ID"] == neuron_i].loc[:,"area"].item()
#                                 area_j = spikeData[spikeData["cell_ID"] == neuron_j].loc[:,"area"].item()

#                                 if area_i == area_j:
#                                     within_area_count += 1
#                                 else:
#                                     between_area_count += 1

#                 within_count_series[comb_index] = int(within_area_count/2)
#                 between_count_series[comb_index] = int(between_area_count/2)

# clusterDF["within_area_count"] = within_count_series
# clusterDF["between_area_count"] = between_count_series

# clusterDF.to_pickle(f"{save_dir}/bc_clusters_area_counts.pkl")


########################################################################################################################################################################
################################## Plotting the within and cross-area connections as a fraction of total connections ####################################################################

# clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_area_counts.pkl")

# for index, ses in zip(range(1, 5), sessionData["session_ID"]):
#     within_fractions = []
#     between_fractions = []

#     # Count the porportion of between and within area connections for each time bin
#     # in the session
#     for time_bin in time_bins:
#         bin_clusterDF = clusterDF[(clusterDF["time_Bin"] == time_bin) & (clusterDF["session_ID"] == ses)]
#         within_sum = np.sum(bin_clusterDF["within_area_count"])
#         between_sum = np.sum(bin_clusterDF["between_area_count"])

#         total_connections = within_sum + between_sum

#         within_fractions.append((within_sum)/total_connections)
#         between_fractions.append((between_sum)/total_connections)

#     # Save the results in a scatter plot
#     plt.scatter(time_bins, within_fractions, c="blue")
#     plt.scatter(time_bins, between_fractions, c="red")

#     plt.title(f"Proportion of connections that are within and between brain areas\nplotted against time bin size, for session {index} ({ses})")
#     plt.xlabel("Time bin (sec)")
#     plt.ylabel("Proportion of total connections in MCM")
#     plt.yticks(ticks=(np.array(range(5))/5))

#     legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(["blue", "red"], ["Within areas", "Between areas"])]
#     plt.legend(handles=legend_patches, bbox_to_anchor=(1, 0.3), loc='best', title="Proportion of connections:")
#     # plt.margins(x=1)

#     plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/within-between_connections_{ses}.png")
#     plt.clf()



########################################################################################################################################################################

############################# Creating LogE-LogE plots for the same stimulus types #########################################################################################################


mcmDF = pd.read_pickle(f"{save_dir}/bc_20ms_2set_MCM_data.pkl")

ses_ID = sessionData.loc[1, "session_ID"]
# get the neurons from the session and their number
ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
neuron_series = ses_neurons["cell_ID"]
n = len(neuron_series)

# ses_mcmDF = mcmDF[mcmDF["session_ID"] == ses_ID]

# same_stim_dir = "/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/log-log_same_stimulus"
# different_stim_dir = "/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/log-log_different_stimulus"

# finished_combinations = []

# for visGroup1 in ses_mcmDF.visGroup.unique():
#     for audioGroup1 in ses_mcmDF.audioGroup.unique():
#         for visGroup2 in ses_mcmDF.visGroup.unique():
#             for audioGroup2 in ses_mcmDF.audioGroup.unique():
#                 combination = np.sort([visGroup1, visGroup2, audioGroup1, audioGroup2])
#                 combination =  "".join([str(elem) for elem in combination])
#                 if combination in finished_combinations:
#                     continue

#                 stim_mcmDF1 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup1) & (ses_mcmDF["audioGroup"] == audioGroup1)) & (mcmDF["trial_set"] == 1)]
#                 stim_mcmDF2 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup2) & (ses_mcmDF["audioGroup"] == audioGroup2)) & (mcmDF["trial_set"] == 2)]

#                 logE_arr1 = np.sort(stim_mcmDF1["logE_array"].item())
#                 logE_arr2 = np.sort(stim_mcmDF2["logE_array"].item())

#                 plt.scatter(logE_arr1, logE_arr2, marker="x", c="black")

#                 if not ((visGroup1 == visGroup2) & (audioGroup1 == audioGroup2)):
#                     plt.title("A scatter plot comparing the sorted log-evidence\nof MCMs from two stimulus groups")
#                     plt.xlabel(f"Log-evidence, stimulus: {visGroup1}°, {audioGroup1}Hz")
#                     plt.ylabel(f"Log-evidence, stimulus: {visGroup2}°, {audioGroup2}Hz")

#                     plt.subplots_adjust(left=0.2)
#                     plt.savefig(f"{different_stim_dir}/v1{visGroup1}-a1{audioGroup1}_v2{visGroup2}-a2{audioGroup2}.png")
#                     plt.clf()
#                     print(f"v1{visGroup1}-a1{audioGroup1}_v2{visGroup2}-a2{audioGroup2} done")
#                 else:
#                     plt.title(f"A scatter plot comparing the sorted log-evidence\nof MCMs of different trial concatenations of the\nstimulus group {visGroup1}° and {audioGroup1}°")
#                     plt.xlabel(f"Log-evidence")
#                     plt.ylabel(f"Log-evidence")

#                     plt.subplots_adjust(left=0.2)
#                     plt.savefig(f"{same_stim_dir}/vis{visGroup1}-aud{audioGroup1}.png")
#                     plt.clf()
#                     print(f"/vis{visGroup1}-aud{audioGroup1}.png done")

#                 finished_combinations.append(combination)



########################################################################################################################################################################

############################# Creating LogE-LogE plots for the same stimulus types #########################################################################################################

ari_scores = []
nmi_scores = []

for ses_ID in sessionData["session_ID"]:
    ses_mcmDF = mcmDF[mcmDF["session_ID"] == ses_ID]

    for visGroup in ses_mcmDF.visGroup.unique():
        for audioGroup in ses_mcmDF.audioGroup.unique():
            stim_mcmDF1 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup) & (ses_mcmDF["audioGroup"] == audioGroup)) & (mcmDF["trial_set"] == 1)]
            stim_mcmDF2 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup) & (ses_mcmDF["audioGroup"] == audioGroup)) & (mcmDF["trial_set"] == 2)]

            if not stim_mcmDF1.empty:
                linkage_m1 = np.array(stim_mcmDF1["linkage_matrix"])[0]
                linkage_m2 = np.array(stim_mcmDF2["linkage_matrix"])[0]

                clusters1 = fcluster(linkage_m1, 0.7*max(linkage_m1[:, 2]), criterion="distance")
                clusters2 = fcluster(linkage_m2, 0.7*max(linkage_m2[:, 2]), criterion="distance")

                ari_score = adjusted_rand_score(clusters1, clusters2)
                nmi_score = normalized_mutual_info_score(clusters1, clusters2)

                ari_scores.append(ari_score)
                nmi_scores.append(nmi_score)

plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams["figure.autolayout"] = True

fig, (ax1, ax2) = plt.subplots(1, 2)

hist_bins = np.array(range(20))/20

ax1.hist(ari_scores, bins=hist_bins)
ax1.set_title("ARI scores")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 5])

ax2.hist(nmi_scores, bins=hist_bins)
ax2.set_title("NMI scores")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 5])

plt.savefig("/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/ARI-NMI_score_histograms_20ms.png")
