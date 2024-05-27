from pathlib import Path
import utils
import raster_plots as rplt
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import math
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

min_fire = 0.5
quality = 'good'
path_root = Path("/Users/vojtamazur/Documents/Capstone_code")
experiment = ["ChangeDetectionConflict"]

trialData, sessionData, spikeData, videoData = utils.load_data(path_root, experiment)
videoData["session_ID"] = sessionData["session_ID"]
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)


# # trialData['visGroupPreChange'] = trialData['visualOriPreChange'].apply(utils.assign_group_visual)
# # trialData['visGroupPostChange'] = trialData['visualOriPostChange'].apply(utils.assign_group_visual)
# # trialData['audioGroupPostChange'] = trialData['audioFreqPostChange'].apply(utils.assign_group_auditory)
# # trialData['audioGroupPreChange'] = trialData['audioFreqPreChange'].apply(utils.assign_group_auditory)

# # fTrialData = trialData.dropna(how="any", subset = ["trialStart", "trialEnd", "stimChange"])

# ########################################################################################################################################################################

# # # Obtain the trial counts for each combination of stimuli groupings for each session
# # trial_count_path = "/Users/vojtamazur/Documents/Capstone_code/trial_counts"

# # for session in sessionData["session_ID"]:
# #     trialData_ses = fTrialData[fTrialData["session_ID"] == session]
# #     countsTotal, countsPre, countsPost = utils.get_trial_counts(trialData_ses)
# #     countsTotal.to_csv(f"{trial_count_path}/{session}_all_counts.csv")
# #     countsPre.to_csv(f"{trial_count_path}/{session}_pre_counts.csv")
# #     countsPost.to_csv(f"{trial_count_path}/{session}_post_counts.csv")


# ########################################################################################################################################################################


# # The following code can be used to obtain the unique combinations of all trial stimuli:

# # Combinations before the stimulus change:
# # unique_combinations_pre = trialData.groupby(['visualOriPreChange', 'audioFreqPreChange']).size().reset_index(name='Count')
# # # Combinations after the stimulus change:
# # unique_combinations_post = trialData.groupby(['visualOriPostChange', 'audioFreqPostChange']).size().reset_index(name='Count')
# # # Combinations both before and after the change:
# # unique_combinations = trialData.groupby(['visualOriPreChange', 'visualOriPostChange', 'audioFreqPreChange', 'audioFreqPostChange']).size().reset_index(name='Count')

# # # Visual change combinations
# # change_combinations_visual = trialData.groupby(['visualOriPreChange', 'visualOriPostChange']).size().reset_index(name='Count')
# # # Auditory change combinations
# # change_combinations_audio = trialData.groupby(['audioFreqPreChange', 'audioFreqPostChange']).size().reset_index(name='Count')

# # #Print the results:
# # print(f"Cominations before: {unique_combinations_pre}")
# # print(f"Cominations after: {unique_combinations_post}")
# # print(f"Cominations before and after: {unique_combinations}")

# # # Save the results of combinations before and after change in a CSV file
# # unique_combinations_pre.to_csv("stimulus_comb_pre_change.csv")
# # unique_combinations_post.to_csv("stimulus_comb_post_change.csv")

# ########################################################################################################################################################################

# # Get the spike data from each trial
# # trialSpikesData = utils.get_trial_spikes(sessionData, fTrialData, spikeData)
# # save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
# # trialSpikesData.to_pickle(f'{save_dir}/3s-1sTrialSpikes.pkl')

# # # #######################################################################################################################################################################

# # # Binarizing intervals and saving them as pickle files
# interval = 10000
# # spike_file = "3s-1sTrialSpikes.pkl"
# # save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"

# # trialSpikesData = pd.read_pickle(f"{save_dir}/{spike_file}")
# # trialBinData = utils.binarize_neurons_in_trial(trialSpikesData, interval)

# # filename = f"binSpikeTrials_{int(interval/1000)}ms"
# # trialBinData.to_pickle(f'{save_dir}/{filename}.pkl')

# # ########################################################################################################################################################################

# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
# plot_dir = "/Users/vojtamazur/Documents/Capstone_code/raster_plots/"
# spike_file = "binSpikeTrials_10ms.pkl"

# # Getting the saved binarized data
# trialBinData = pd.read_pickle(f"{save_dir}/{spike_file}")

# # # Plotting superimposed Raster plots for each session
# # for index, session in sessionData.iterrows():
# #     ses_ID = session["session_ID"]
# #     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]
# #     rplt.raster_plot_superimposed(ses_trials, spikeData, ses_ID, interval, f"Session {index+1} trials", plot_dir)

# # Plotting the binarized Raster plots for each combination of trials in each session
# for index, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     for visGroup in ses_trials.visGroupPreChange.unique():
#         for audioGroup in ses_trials.audioGroupPreChange.unique():
#             comb_trials = ses_trials[(ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)]
#             plt_save_dir = f"{plot_dir}/Session_{index+1}-{ses_ID}_trial_groups"
#             rplt.raster_plot_superimposed(comb_trials, spikeData, ses_ID, interval, f"Pre-change visual{visGroup} audio{audioGroup}", plt_save_dir)

# # # Plotting the binarized Raster plots for each combination of trials before and after change in session 3
# # for index, session in sessionData.iterrows():
# #     ses_ID = session["session_ID"]
# #     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

# #     for visGroupPre in ses_trials.visGroupPreChange.unique():
# #         for audioGroupPre in ses_trials.audioGroupPreChange.unique():
# #             for visGroupPost in ses_trials.visGroupPostChange.unique():
# #                 for audioGroupPost in ses_trials.audioGroupPostChange.unique():
# #                     comb_trials = ses_trials[(ses_trials["visGroupPreChange"] == visGroupPre) & (ses_trials["audioGroupPreChange"] == audioGroupPre) & (ses_trials["visGroupPostChange"] == visGroupPost) & (ses_trials["audioGroupPostChange"] == audioGroupPost)]
# #                     plt_save_dir = f"{plot_dir}/Session_{index+1}-{ses_ID}_trial_groups"
# #                     rplt.raster_plot_superimposed(comb_trials, spikeData, ses_ID, interval, f"Pre visual-{visGroupPre} audio-{audioGroupPost}_Post visual-{visGroupPost} audio-{audioGroupPost}", plt_save_dir)


# # # Plotting each trial in session 3
# # ses_ID = sessionData.loc[2, "session_ID"]
# # ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

# # for _, trial in ses_trials.iterrows():
# #     plt_save_dir = f"{plot_dir}/Session_3-{ses_ID}_individual_test"
# #     rplt.raster_plot_individual(trial, spikeData, ses_ID, interval, plt_save_dir)


# # # Plotting the Raster plots to be used in the Capstone text
# # visGroup, audioGroup = ["225-230", "13000-13020"]
# # comb_trials = trialBinData[(trialBinData["visGroupPreChange"] == visGroup) & (trialBinData["audioGroupPreChange"] == audioGroup)]
# # trial1 = comb_trials.iloc[0, :]

# # rplt.raster_plot_superimposed(
# #     comb_trials,
# #     spikeData,
# #     trial1["session_ID"],
# #     interval,
# #     f"visual {visGroup} audio {audioGroup} text plot", f"{plot_dir}/main_text",
# #     title=False)



# ########################################################################################################################################################################
# ############ Count the average number of individual firings in a time bin ######################################################


# # time_bins = [5, 10, 15, 20, 25, 30, 40, 50]
# # # time_bins = [10, 20]
# # spike_file = "3s-1sTrialSpikes.pkl"
# # save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"


# # trialSpikesData = pd.read_pickle(f"{save_dir}/{spike_file}")

# # avg_bin_counts = {}

# # for time_bin in time_bins:
# #     interval = time_bin * 1000
# #     t_spike_counts = utils.count_bin_spikes(trialSpikesData, interval)

# #     counts = []

# #     for trial in t_spike_counts:
# #         for bin_counts in trial.values():
# #             # mean_count = np.mean(bin_counts)
# #             non0counts = bin_counts[bin_counts != 0]
# #             mean_count = np.mean(non0counts)
# #             counts.append(mean_count)

# #     counts = np.array(counts)
# #     avg_bin_counts[str(time_bin)] = np.mean(counts[~np.isnan(counts)])

# # print(avg_bin_counts)



# # print(trialSpikesData.columns)

# # filename = f"binSpikeCounts_{int(interval/1000)}ms"
# # trialBinCountData.to_pickle(f'{save_dir}/{filename}.pkl')





# ########################################################################################################################################################################

################################ Obtaining the pupil size data in each trial ###############################################################################################

# time_bin = 20
# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
# spike_file = f"binSpikeTrials_{time_bin}ms.pkl"

# trialBinData = pd.read_pickle(f"{save_dir}/{spike_file}")
# trial_area_ser = pd.Series()

# for ses in sessionData["session_ID"]:
#     ses_trialData = trialBinData[trialBinData["session_ID"] == ses]
#     ses_videoData = videoData[videoData["session_ID"] == ses]

#     video_areas = ses_videoData["area"].item()[0]
#     video_ts = ses_videoData["ts"].item()[0]

#     # Initialize a pandas Series to store the results
#     result_series = pd.Series(index=ses_trialData.index, dtype=object)

#     # Iterate through each row in trialData
#     for idx, stim_time in ses_trialData['stimChange'].items():
#         if not math.isnan(stim_time):
#             # Define the time window of 2 seconds before the stimChange
#             start_time = stim_time - 2_000_000  # 2 seconds before, since timestamps are in microseconds
#             end_time = stim_time

#             # Extract the area measurements in the defined time window
#             relevant_areas = video_areas[(video_ts >= start_time) & (video_ts < end_time)].tolist()

#             # Store the result in the series
#             result_series.at[idx] = relevant_areas
#         else:
#             result_series.at[idx] = []

#     trial_area_ser = pd.concat([trial_area_ser, result_series])

# trialBinData["pupilAreas"] = trial_area_ser

# trialBinData.to_pickle(f"{save_dir}/pupilBinSpikeTrials_{time_bin}ms.pkl")




# ########################################################################################################################################################################

################################ Correlating pupil size and log-evidence ###############################################################################################

time_bin = 20

# get the dataframe with log-evidence data
save_dir = "/Users/vojtamazur/Documents/Capstone_code/superimposed_matrices/data"
clusterData = pd.read_pickle(f"{save_dir}/bc_30ms_trial_logE_data.pkl")

# get the dataframe with the pupil area data
pupil_file = f"/Users/vojtamazur/Documents/Capstone_code/spike_data/pupilBinSpikeTrials_{time_bin}ms.pkl"
pupilTrialData = pd.read_pickle(pupil_file)

min_data_size = 1500
sample_size = math.ceil(time_bin*(min_data_size/2000))
print(sample_size)

for ses in clusterData.session_ID.unique():
    # Get the data from trials in only this session
    ses_clustData = clusterData[clusterData["session_ID"] == ses]
    sliced_clustData = ses_clustData.iloc[4:8]
    # print(sliced_clustData.columns)

    ses_pupilTrialData = pupilTrialData[pupilTrialData["session_ID"] == ses]

    # Set up a variables to track the mean area and log-evidence in a trial concatenation
    ses_mean_areas = []
    ses_std_areas = []
    ses_logE = []


    # Since the clusterData, which contain log-evidence is organized stimulus combination-wise:
    for visGroup in ses_clustData.visGroup.unique():
        for audioGroup in ses_clustData.audioGroup.unique():
            comb_clustData = sliced_clustData[(sliced_clustData["visGroup"] == visGroup) & (sliced_clustData["audioGroup"] == audioGroup)]
            comb_pupilTrialData = ses_pupilTrialData[(ses_pupilTrialData["visGroupPreChange"] == visGroup) & (ses_pupilTrialData["audioGroupPreChange"] == audioGroup)]
            logE_array = comb_clustData["logE_arrays"].item()

            areas = []

            # Average the pupil size over the trial concatenation
            for i in range(len(comb_pupilTrialData)-sample_size):
                # Get the average pupil size for the concatenation
                end_index = i+sample_size
                sampleTrials = comb_pupilTrialData.iloc[i:end_index, :]
                mean_areas = [np.mean(areas) for areas in sampleTrials["pupilAreas"]]

                # Append the log-evidence and the concatenation to the results
                ses_mean_areas.append(np.mean(mean_areas))
                ses_logE.append(logE_array[i])
                ses_std_areas.append(np.std(mean_areas))

                areas.append(np.mean(mean_areas))

            x = range(len(logE_array))

            fig, ax1 = plt.subplots()

            color = 'blue'
            ax1.set_xlabel('trial number')
            ax1.set_ylabel('log evidence', color=color)
            ax1.scatter(x, logE_array, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

            color = 'red'
            ax2.set_ylabel('pupil size', color=color)  # we already handled the x-label with ax1
            ax2.scatter(x, areas, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f"Log evidence (blue) plotted alongside average pupil size (red)\nfor the session {ses}, stimulus {visGroup}° and {audioGroup}Hz")

            fig.tight_layout()

            plt_save = f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/logE_vs_pupil/stimulus_specific_{ses}"
            # Check if the save directory exists, and if not, create it
            if not os.path.exists(plt_save):
                os.makedirs(plt_save)

            plt.savefig(f"{plt_save}/log_E_stim_{visGroup}-{audioGroup}.png")
            plt.clf()

    correlation = np.corrcoef([ses_mean_areas, ses_logE])[0, 1]
    # print(ses_std_areas)
    # print(correlation)
    print(np.std(areas))
    print("\n")




    # plt.scatter(range(len(ses_logE)), ses_logE)
    # plt.show()
    # plt.scatter(range(len(ses_logE)), ses_mean_areas)
    # plt.show()




# logEData = pd.read_pickle(f"{save_dir}/bc_clusters_ind_15ms.pkl")

# for ses in clusterData.session_ID.unique():
#     # Get the data from trials in only this session
#     ses_logEData = logEData[logEData["session_ID"] == ses]

#     ses_pupilTrialData = pupilTrialData[pupilTrialData["session_ID"] == ses]

#     # Set up a variables to track the mean area and log-evidence in a trial concatenation
#     ses_mean_areas = [np.mean(areas) for areas in ses_pupilTrialData["pupilAreas"]]
#     ses_area_std = [np.std(areas) for areas in ses_pupilTrialData["pupilAreas"]]
#     ses_logE = []

#     for logE_array in ses_logEData["logE"]:
#         for logE in logE_array:
#             ses_logE.append(logE)

#     correlation = np.corrcoef([ses_mean_areas, ses_logE])[0, 1]
#     # print(correlation)
#     print(np.mean(ses_area_std))

#     x = range(len(ses_logE))

#     # fig, ax1 = plt.subplots()

#     # color = 'blue'
#     # ax1.set_xlabel('trial number')
#     # ax1.set_ylabel('log evidence', color=color)
#     # ax1.scatter(x, ses_logE, color=color)
#     # ax1.tick_params(axis='y', labelcolor=color)

#     # ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

#     # color = 'red'
#     # ax2.set_ylabel('pupil size', color=color)  # we already handled the x-label with ax1
#     # ax2.scatter(x, ses_mean_areas, color=color)
#     # ax2.tick_params(axis='y', labelcolor=color)

#     # fig.tight_layout()
#     # plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/logE_vs_pupil/session_{ses}_single_trials.png")

#     # plt.scatter(range(len(ses_logE)), ses_logE)
#     plt.title("Progression of log-evidence over time (MCMs from single trials)")
#     plt.plot(ses_logE)
#     plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/logE_vs_pupil/log_E_over_time_{ses}.png")
#     plt.clf()



# ################################ Getting plots of just the pupil size over time

# for _, ses in videoData.iterrows():
#     pupil_sizes = ses["area"][0]


#     plt.plot(pupil_sizes)
#     plt.savefig(f"/Users/vojtamazur/Documents/Capstone_code/clustering_analysis/logE_vs_pupil/p_size_over_time_{ses['session_ID']}.png")
#     plt.clf()

