import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/"
clusterDF = pd.read_pickle(f"{save_dir}/bc_clusters_all_bins.pkl")

def calc_cluster_comparison(cluster_ser, comp_metric):
    result_arr = np.zeros([len(cluster_ser), len(cluster_ser)])
    reset_indexDF = cluster_ser.reset_index()
    cluster_ser = reset_indexDF["clusters"]

    for i, clusters1 in cluster_ser.items():
        for j, clusters2 in cluster_ser.items():
            result_arr[int(i)][int(j)] = comp_metric(clusters1, clusters2)


    return result_arr


ses3_clusterDF = clusterDF[clusterDF["session_ID"] == "20122018081320"]

means_ari = []
means_nmi = []

for visGroup in clusterDF.visGroup.unique():
    for audioGroup in clusterDF.audioGroup.unique():
        comb_clusterDF = clusterDF[(clusterDF["visGroup"] == visGroup) & (clusterDF["audioGroup"] == audioGroup)]
        cluster_ser = comb_clusterDF["clusters"]


        # calculate comparison indices
        ari_score = calc_cluster_comparison(cluster_ser, adjusted_rand_score)
        nmi_score = calc_cluster_comparison(cluster_ser, normalized_mutual_info_score)

        where_matrix = np.full(ari_score.shape, True)
        for i in range(len(ari_score)):
            for j in range(len(ari_score)):
                if i == j:
                    where_matrix[i][j] = False
        print(where_matrix)

        # print the scores
        # print(f"For the stimulus combination {visGroup} degrees and {audioGroup} Hz:")
        # print(f"ARI score: {np.mean(ari_score)}")
        # print(f"NMI score: {np.mean(nmi_score)}\n")

        ari_mean = np.mean(ari_score, where=where_matrix)
        nmi_mean = np.mean(nmi_score, where=where_matrix)


        if not math.isnan(ari_mean):
            means_ari.append(ari_mean)
            means_nmi.append(nmi_mean)


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


print(f"Mean ARI score: {np.mean(means_ari)}")
print(f"Mean NMI score: {np.mean(means_nmi)}")



