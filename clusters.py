import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import models
import MinCompSpin_Python.MinCompSpin as mod
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#     for visGroup in ses_clusterDF.visGroup.unique():
#         for audioGroup in ses_clusterDF.audioGroup.unique():
#             comb_clusterDF = ses_clusterDF[(ses_clusterDF["visGroup"] == visGroup) & (ses_clusterDF["audioGroup"] == audioGroup)]
#             partitions = comb_clusterDF["MCM_Partition"]

#             clusterings = []

#             for t_bin_partitions in partitions:
#                 for partition in t_bin_partitions:
#                     clusters = np.zeros(n)

#                     for community_index, community in enumerate(partition):
#                         if n <= 64:
#                             component = bin(community[1])[2:].zfill(n)
#                         else:
#                             comp1 = bin(community[1])[2:].zfill(64)
#                             comp2 = bin(community[2])[2:].zfill(n-64)
#                             component = comp1 + comp2

#                         for variable_index, belongs in enumerate(component):
#                             if belongs == '1':
#                                 clusters[variable_index] = community_index
#                     clusterings.append(clusters)
#             print(len(clusterings))

#             # calculate comparison indices
#             ari_score = calc_cluster_comparison(clusterings, adjusted_rand_score)
#             nmi_score = calc_cluster_comparison(clusterings, normalized_mutual_info_score)
#             # print(f"\nFor the combination {visGroup}° and {audioGroup}Hz:")
#             # print("ARI score:")
#             # print(ari_score)
#             # print("NMI score:")
#             # print(nmi_score)

#             where_matrix = np.full(ari_score.shape, True)
#             for i in range(len(ari_score)):
#                 for j in range(len(ari_score)):
#                     if i == j:
#                         where_matrix[i][j] = False
#             print(where_matrix)

#             # # print the scores
#             # # print(f"For the stimulus combination {visGroup} degrees and {audioGroup} Hz:")
#             # # print(f"ARI score: {np.mean(ari_score)}")
#             # # print(f"NMI score: {np.mean(nmi_score)}\n")

#             ari_mean = np.mean(ari_score, where=where_matrix)
#             nmi_mean = np.mean(nmi_score, where=where_matrix)

#             print(ari_mean)
#             print(nmi_mean)

#             if not math.isnan(ari_mean):
#                 means_ari.append(ari_mean)
#                 means_nmi.append(nmi_mean)


for session in clusterDF.session_ID.unique():
    ses_neurons = spikeData[spikeData["session_ID"] == session]
    neuron_series = ses_neurons["cell_ID"]
    n = len(neuron_series)


    for t_bin in clusterDF.time_Bin.unique():
        ses_clusterDF = clusterDF[(clusterDF["session_ID"] == session) & (clusterDF["time_Bin"] == t_bin)]
        # cluster_ser = ses_clusterDF["clusters"]
        partitions = ses_clusterDF["MCM_Partition"]

        clusterings = []

        for stim_comb_partitions in partitions:
            for partition in stim_comb_partitions:
                clusters = np.zeros(n)

                for community_index, community in enumerate(partition):
                    if n <= 64:
                        component = bin(community[1])[2:].zfill(n)
                    else:
                        comp1 = bin(community[1])[2:].zfill(64)
                        comp2 = bin(community[2])[2:].zfill(n-64)
                        component = comp1 + comp2

                    for variable_index, belongs in enumerate(component):
                        if belongs == '1':
                            clusters[variable_index] = community_index
                clusterings.append(clusters)

        print(len(clusterings))

         # calculate comparison indices
        ari_score = calc_cluster_comparison(clusterings, adjusted_rand_score)
        nmi_score = calc_cluster_comparison(clusterings, normalized_mutual_info_score)

        where_matrix = np.full(ari_score.shape, True)
        for i in range(len(ari_score)):
            for j in range(len(ari_score)):
                if i == j:
                    where_matrix[i][j] = False

        # print the scores
        # print(f"For session {session} with {t_bin}s time bins:")

        ari_mean = np.mean(ari_score, where=where_matrix)
        nmi_mean = np.mean(nmi_score, where=where_matrix)
        # print(f"ARI score: {ari_mean}")
        # print(f"NMI score: {nmi_mean}\n")

        if not math.isnan(ari_mean):
            means_ari.append(ari_mean)
            means_nmi.append(nmi_mean)

    # print(f"Session {session}:")
    # print(f"Mean ARI score: {np.mean(means_ari)}")
    # print(f"Mean NMI score: {np.mean(means_nmi)}\n")


print(f"Mean ARI score: {np.mean(means_ari)}")
print(f"Mean NMI score: {np.mean(means_nmi)}")


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
# V1_CG1_series = pd.Series(0, index=clusterDF.index)
# V1_PPC_series = pd.Series(0, index=clusterDF.index)
# PPC_CG1_series = pd.Series(0, index=clusterDF.index)

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
#                 V1_CG1_count = 0
#                 V1_PPC_count = 0
#                 PPC_CG1_count = 0

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
#                                 elif ((area_i == "V1") and (area_j == "CG1")) or ((area_i == "CG1") and (area_j == "V1")):
#                                     V1_CG1_count += 1
#                                     between_area_count += 1
#                                 elif ((area_i == "V1") and (area_j == "PPC")) or ((area_i == "PPC") and (area_j == "V1")):
#                                     V1_PPC_count += 1
#                                     between_area_count += 1
#                                 else:
#                                     PPC_CG1_count += 1
#                                     between_area_count += 1

#                 within_count_series[comb_index] = int(within_area_count/2)
#                 between_count_series[comb_index] = int(between_area_count/2)
#                 V1_CG1_series[comb_index] = int(V1_CG1_count/2)
#                 V1_PPC_series[comb_index] = int(V1_PPC_count/2)
#                 PPC_CG1_series[comb_index] = int(PPC_CG1_count/2)

# clusterDF["within_area_count"] = within_count_series
# clusterDF["between_area_count"] = between_count_series
# clusterDF["V1_CG1_count"] = V1_CG1_series
# clusterDF["V1_PPC_count"] = V1_PPC_series
# clusterDF["PPC_CG1_count"] = PPC_CG1_series

# clusterDF.to_pickle(f"{save_dir}/bc_clusters_area_counts.pkl")


########################################################################################################################################################################
################################## Plotting the within and cross-area connections as a fraction of total connections ####################################################################

# clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_area_counts.pkl")

# for index, ses in zip(range(1, 5), sessionData["session_ID"]):
#     within_fractions = []
#     between_fractions = []
#     V1_CG1_fractions = []
#     V1_PPC_fractions = []
#     PPC_CG1_fractions = []
#     all_3_fractions = []

#     # Count the porportion of between and within area connections for each time bin
#     # in the session
#     for time_bin in time_bins:
#         bin_clusterDF = clusterDF[(clusterDF["time_Bin"] == time_bin) & (clusterDF["session_ID"] == ses)]
#         within_sum = np.sum(bin_clusterDF["within_area_count"])
#         between_sum = np.sum(bin_clusterDF["between_area_count"])
#         V1_CG1_sum = np.sum(bin_clusterDF["V1_CG1_count"])
#         V1_PPC_sum = np.sum(bin_clusterDF["V1_PPC_count"])
#         PPC_CG1_sum = np.sum(bin_clusterDF["PPC_CG1_count"])

#         total_connections = within_sum + between_sum

#         within_fractions.append((within_sum)/total_connections)
#         between_fractions.append((between_sum)/total_connections)
#         V1_CG1_fractions.append((V1_CG1_sum)/total_connections)
#         V1_PPC_fractions.append((V1_PPC_sum)/total_connections)
#         PPC_CG1_fractions.append((PPC_CG1_sum)/total_connections)
#         all_3_fractions.append((between_sum - V1_CG1_sum - V1_PPC_sum - PPC_CG1_sum)/total_connections)

#     # Save the results in a scatter plot
#     plt.scatter(time_bins, within_fractions, c="blue")
#     plt.scatter(time_bins, between_fractions, c="red")
#     if not (np.sum(V1_CG1_fractions) == 0):
#         plt.scatter(time_bins, V1_CG1_fractions, c="teal")
#     plt.scatter(time_bins, V1_PPC_fractions, c="purple")
#     if not (np.sum(V1_CG1_fractions) == 0):
#         plt.scatter(time_bins, PPC_CG1_fractions, c ="orange")
#     # plt.scatter(time_bins, all_3_fractions, c="black")


#     # plt.title(f"Proportion of connections that are within and between brain areas\nplotted against time bin size, for session {index} ({ses})")
#     plt.title(f"Session {index}")
#     plt.xlabel("Time bin (sec)")
#     plt.xlim([3, 32])
#     plt.ylabel("Proportion of total connections in MCM")
#     plt.yticks(ticks=(np.array(range(5))/5))

#     # legend_colors = ["blue", "red", "teal", "purple", "orange"]
#     # legend_labels = ["Within areas", "Between areas", "Between V1 and CG1", "Between V1 and PPC", "Between PPC and CG1"]
#     # legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
#     # plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, -0.15), loc='best', title="Proportion of connections:")

#     # plt.subplots_adjust(bottom=0.4)

#     plt.show()

#     # plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/within-between_connections_{ses}.png")
#     # plt.clf()



########################################################################################################################################################################

############################# Creating LogE-LogE plots for the same stimulus types #########################################################################################################


mcmDF = pd.read_pickle(f"{save_dir}/bc_30ms_2set_MCM_data.pkl")

# ses_ID = sessionData.loc[1, "session_ID"]
# # get the neurons from the session and their number
# ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
# neuron_series = ses_neurons["cell_ID"]
# n = len(neuron_series)

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

############################# Creating histograms of ARI and NMI scores #########################################################################################################

# ari_scores = []
# nmi_scores = []

# full_ses_clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_full_ses.pkl")

# for ses_ID in sessionData["session_ID"]:
#     ses_mcmDF = mcmDF[mcmDF["session_ID"] == ses_ID]
#     ses_clusterDF = full_ses_clusterDF[(full_ses_clusterDF["session_ID"] == ses_ID) & (full_ses_clusterDF["time_Bin"] == 20)]

#     if ses_ID == "20122018081524":
#         neuron_thresh = 42
#         n = 86
#     if ses_ID == "20122018081421":
#         neuron_thresh = 33
#         n = 68
#     if ses_ID == "20122018081320":
#         neuron_thresh = 28
#         n = 37
#     if ses_ID == "20122018081628":
#         neuron_thresh = 18
#         n = 31

#     reordered_series = ses_clusterDF["reordered_Series"].item()

#     # Get the neurons from the session and their ID number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_arr = np.array(ses_neurons["cell_ID"])

#     # Map original neuron IDs to their indices
#     neuron_index_map = {neuron_id: idx for idx, neuron_id in enumerate(neuron_arr)}

#     # Find the indices in the original matrix that correspond to the reordered neurons
#     reordered_indices = [neuron_index_map[neuron] for neuron in reordered_series]
#     relevant_indices = reordered_indices[:neuron_thresh]

#     for visGroup in ses_mcmDF.visGroup.unique():
#         for audioGroup in ses_mcmDF.audioGroup.unique():
#             stim_mcmDF1 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup) & (ses_mcmDF["audioGroup"] == audioGroup)) & (mcmDF["trial_set"] == 1)]
#             stim_mcmDF2 = ses_mcmDF[((ses_mcmDF["visGroup"] == visGroup) & (ses_mcmDF["audioGroup"] == audioGroup)) & (mcmDF["trial_set"] == 2)]

#             if not stim_mcmDF1.empty:
#                 # linkage_m1 = np.array(stim_mcmDF1["linkage_matrix"])[0]
#                 # linkage_m2 = np.array(stim_mcmDF2["linkage_matrix"])[0]
#                 partitions1 = stim_mcmDF1["MCM_partition"].item()
#                 partitions2 = stim_mcmDF2["MCM_partition"].item()

#                 clusterings = []

#                 for partition in partitions1:
#                     clusters = np.zeros(n)

#                     for community_index, community in enumerate(partition):
#                         if n <= 64:
#                             component = bin(community[1])[2:].zfill(n)
#                         else:
#                             comp1 = bin(community[1])[2:].zfill(64)
#                             comp2 = bin(community[2])[2:].zfill(n-64)
#                             component = comp1 + comp2

#                         for variable_index, belongs in enumerate(component):
#                             if belongs == '1':
#                                 clusters[variable_index] = community_index
#                     clusterings.append(clusters)

#                 for partition in partitions2:
#                     clusters = np.zeros(n)

#                     for community_index, community in enumerate(partition):
#                         if n <= 64:
#                             component = bin(community[1])[2:].zfill(n)
#                         else:
#                             comp1 = bin(community[1])[2:].zfill(64)
#                             comp2 = bin(community[2])[2:].zfill(n-64)
#                             component = comp1 + comp2

#                         for variable_index, belongs in enumerate(component):
#                             if belongs == '1':
#                                 clusters[variable_index] = community_index
#                     clusterings.append(clusters)

#                 ari_score_m = calc_cluster_comparison(clusterings, adjusted_rand_score)
#                 nmi_score_m = calc_cluster_comparison(clusterings, normalized_mutual_info_score)

#                 for i in range(len(ari_score_m)):
#                     for j in range(len(ari_score_m)):
#                         if i != j:
#                             ari_scores.append(ari_score_m[i][j])
#                             nmi_scores.append(nmi_score_m[i][j])

#                 # clusters1 = fcluster(linkage_m1, 0.7*max(linkage_m1[:, 2]), criterion="distance")
#                 # clusters2 = fcluster(linkage_m2, 0.7*max(linkage_m2[:, 2]), criterion="distance")

#                 # good_clusters1 = [neuron for index, neuron in enumerate(clusters1) if index in relevant_indices]
#                 # good_clusters2 = [neuron for index, neuron in enumerate(clusters2) if index in relevant_indices]

#                 # ari_score = adjusted_rand_score(good_clusters1, good_clusters2)
#                 # nmi_score = normalized_mutual_info_score(good_clusters1, good_clusters2)
#                 # # ari_score = adjusted_rand_score(clusters1, clusters2)
#                 # # nmi_score = normalized_mutual_info_score(clusters1, clusters2)

#                 # ari_scores.append(ari_score)
#                 # nmi_scores.append(nmi_score)


# plt.rcParams["figure.figsize"] = [6,3]
# plt.rcParams["figure.autolayout"] = True

# fig, (ax1, ax2) = plt.subplots(1, 2)

# hist_bins = np.array(range(40))/40

# ax1.hist(ari_scores, bins=hist_bins)
# ax1.set_title("ARI scores")
# # ax1.set_xlim([0, 1])
# # ax1.set_ylim([0, 5])

# ax2.hist(nmi_scores, bins=hist_bins)
# ax2.set_title("NMI scores")
# # ax2.set_xlim([0, 1])
# # ax2.set_ylim([0, 5])


# plt.savefig("/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/score_hists_good_clusters_20ms.png")

# print(np.mean(ari_scores))
# print(np.mean(nmi_scores))











# ########################################################################################################################################################################

# ############################# Comparing the sizes of clusters with connections within brain areas and between brain areas ###################################################

# time_bins = [5, 10, 15, 20, 25, 30]

# clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_area_counts.pkl")

# mean_sizes_within = []
# mean_sizes_between = []

# for index, ses_ID in sessionData["session_ID"].items():
#     ses_clusterDF = clusterDF[clusterDF["session_ID"] == ses_ID]

#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]

#     neuron_arr = np.array(ses_neurons["cell_ID"])
#     n = len(neuron_arr)

#     for time_bin in time_bins:
#         t_clusterDF = ses_clusterDF[ses_clusterDF["time_Bin"] == time_bin]

#         # linkage_matrices = t_clusterDF["linkage_Matrix"].to_list()
#         # cluster_list = [fcluster(x, 0.7*max(x[:, 2]), criterion="distance") for x in linkage_matrices]

#         sizes_within = []
#         sizes_between = []

#         # for clustering in cluster_list:
#         #     for clust in np.unique(clustering):
#         #         var_indeces = [index for index, value in enumerate(clustering) if value == clust]
#         #         clust_size = len(var_indeces)
#         #         print(len(var_indeces))

#         #         # if len(var_indeces) < (0.2 * n):
#         #         neuron_ids = neuron_arr[var_indeces]
#         #         area_li = []

#         #         for neuron in neuron_ids:
#         #             neuron_data = spikeData[spikeData["cell_ID"] == neuron]
#         #             area_li.append(neuron_data["area"].item())

#         #         same_area = all(x == area_li[0] for x in area_li)
#         #         if same_area:
#         #             sizes_within.append(clust_size)
#         #         else:
#         #             sizes_between.append(clust_size)

#         MCM_partitions = t_clusterDF["MCM_Partition"]
#         for comb in MCM_partitions:
#             for indx, sample in enumerate(comb):
#                 communities = []

#                 for array in sample:
#                     if n <= 64:
#                         component = bin(array[1])[2:].zfill(n)
#                     else:
#                         comp1 = bin(array[1])[2:].zfill(64)
#                         comp2 = bin(array[2])[2:].zfill(n-64)
#                         component = comp1 + comp2

#                     communities.append(component)

#                 for community in communities:
#                     var_indeces = [index for index, value in enumerate(community) if value == "1"]

#                     clust_size = len(var_indeces)
#                     if clust_size > 1:
#                         neuron_ids = neuron_arr[var_indeces]
#                         area_li = []

#                         for neuron in neuron_ids:
#                             neuron_data = spikeData[spikeData["cell_ID"] == neuron]
#                             area_li.append(neuron_data["area"].item())

#                         if all(x == area_li[0] for x in area_li):
#                             sizes_within.append(clust_size)
#                         else:
#                             sizes_between.append(clust_size)

#                 print(f"Sample {indx} done")

#         mean_sizes_within.append([np.mean(sizes_within), index])
#         mean_sizes_between.append([np.mean(sizes_between), index])

# # PLot the results in a figure
# fig, ax = plt.subplots()

# plt.scatter(time_bins, [x[0] for x in mean_sizes_within if x[1] == 0], c="blue", marker="o")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_within if x[1] == 1], c="blue", marker="x")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_within if x[1] == 2], c="blue", marker="*")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_within if x[1] == 3], c="blue", marker="v")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_between if x[1] == 0], c="red", marker="o")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_between if x[1] == 1], c="red", marker="x")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_between if x[1] == 2], c="red", marker="*")
# plt.scatter(time_bins, [x[0] for x in mean_sizes_between if x[1] == 3], c="red", marker="v")
# # plt.scatter(time_bins, mean_sizes_between, c="red")
# # plt.title(f"The size of components plotted against time bins for session: {ses_ID}")

# # # Create custom legends for markers and connections
# # sessions = ["Session 1", "Session 2", "Session 3", "Session 4"]
# # labels = ["Within areas", "Between areas"]
# # markers = ["o", "X", "*", "v"]
# # colors = ["blue", "red"]

# # # Create custom legend for markers
# # marker_legend = [Line2D([0], [0], marker=m, color='w', label=sessions[i],
# #                         markerfacecolor='black', markersize=10) for i, m in enumerate(markers)]
# # # Create custom legend for colors
# # color_legend = [Line2D([0], [0], marker='o', color=colors[j], label=labels[j],
# #                        markerfacecolor=colors[j], markersize=10) for j in range(len(colors))]

# # legend1 = ax.legend(handles=marker_legend, title="Sessions", loc='upper right', bbox_to_anchor=(1,-0.1))
# # legend2 = ax.legend(handles=color_legend, title="Connections", loc='upper left', bbox_to_anchor=(0,-0.2))
# # ax.add_artist(legend1)

# plt.xlabel("Time bin (ms)")
# plt.ylabel("Average cluster size")

# # # plt.legend(handles=legend_patches, bbox_to_anchor=(1, -0.1), loc='best', title="Components with connections:")
# # plt.subplots_adjust(bottom=0.35)

# plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/clust_size/bet-with_all_ses.png")






