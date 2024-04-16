from pathlib import Path
import utils
import raster_plots as rplt
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

min_fire = 0.5
quality = 'good'
path_root = Path("/Users/vojtamazur/Documents/Capstone_code")
experiment = ["ChangeDetectionConflict"]

trialData, sessionData, spikeData = utils.load_data(path_root, experiment)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)

trialData['visGroupPreChange'] = trialData['visualOriPreChange'].apply(utils.assign_group_visual)
trialData['visGroupPostChange'] = trialData['visualOriPostChange'].apply(utils.assign_group_visual)
trialData['audioGroupPostChange'] = trialData['audioFreqPostChange'].apply(utils.assign_group_auditory)
trialData['audioGroupPreChange'] = trialData['audioFreqPreChange'].apply(utils.assign_group_auditory)

fTrialData = trialData.dropna(how="any", subset = ["trialStart", "trialEnd", "stimChange"])

########################################################################################################################################################################

# # Obtain the trial counts for each combination of stimuli groupings for each session
# trial_count_path = "/Users/vojtamazur/Documents/Capstone_code/trial_counts"

# for session in sessionData["session_ID"]:
#     trialData_ses = fTrialData[fTrialData["session_ID"] == session]
#     countsTotal, countsPre, countsPost = utils.get_trial_counts(trialData_ses)
#     countsTotal.to_csv(f"{trial_count_path}/{session}_all_counts.csv")
#     countsPre.to_csv(f"{trial_count_path}/{session}_pre_counts.csv")
#     countsPost.to_csv(f"{trial_count_path}/{session}_post_counts.csv")


########################################################################################################################################################################


# The following code can be used to obtain the unique combinations of all trial stimuli:

# Combinations before the stimulus change:
# unique_combinations_pre = trialData.groupby(['visualOriPreChange', 'audioFreqPreChange']).size().reset_index(name='Count')
# # Combinations after the stimulus change:
# unique_combinations_post = trialData.groupby(['visualOriPostChange', 'audioFreqPostChange']).size().reset_index(name='Count')
# # Combinations both before and after the change:
# unique_combinations = trialData.groupby(['visualOriPreChange', 'visualOriPostChange', 'audioFreqPreChange', 'audioFreqPostChange']).size().reset_index(name='Count')

# # Visual change combinations
# change_combinations_visual = trialData.groupby(['visualOriPreChange', 'visualOriPostChange']).size().reset_index(name='Count')
# # Auditory change combinations
# change_combinations_audio = trialData.groupby(['audioFreqPreChange', 'audioFreqPostChange']).size().reset_index(name='Count')

# #Print the results:
# print(f"Cominations before: {unique_combinations_pre}")
# print(f"Cominations after: {unique_combinations_post}")
# print(f"Cominations before and after: {unique_combinations}")

# # Save the results of combinations before and after change in a CSV file
# unique_combinations_pre.to_csv("stimulus_comb_pre_change.csv")
# unique_combinations_post.to_csv("stimulus_comb_post_change.csv")

########################################################################################################################################################################

# Get the spike data from each trial
# trialSpikesData = utils.get_trial_spikes(sessionData, fTrialData, spikeData)
# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
# trialSpikesData.to_pickle(f'{save_dir}/3s-1sTrialSpikes.pkl')

# # #######################################################################################################################################################################

# # Binarizing intervals and saving them as pickle files
interval = 20000
# spike_file = "3s-1sTrialSpikes.pkl"
# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"

# trialSpikesData = pd.read_pickle(f"{save_dir}/{spike_file}")
# trialBinData = utils.binarize_neurons_in_trial(trialSpikesData, interval)

# filename = f"binSpikeTrials_{int(interval/1000)}ms"
# trialBinData.to_pickle(f'{save_dir}/{filename}.pkl')

# ########################################################################################################################################################################

# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"
# plot_dir = "/Users/vojtamazur/Documents/Capstone_code/raster_plots/"
# spike_file = "binSpikeTrials_10ms.pkl"

# # Getting the saved binarized data
# trialBinData = pd.read_pickle(f"{save_dir}/{spike_file}")

# # Plotting superimposed Raster plots for each session
# for index, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]
#     rplt.raster_plot_superimposed(ses_trials, spikeData, ses_ID, interval, f"Session {index+1} trials", plot_dir)

# # Plotting the binarized Raster plots for each combination of trials in each session
# for index, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     for visGroup in ses_trials.visGroupPreChange.unique():
#         for audioGroup in ses_trials.audioGroupPreChange.unique():
#             comb_trials = ses_trials[(ses_trials["visGroupPreChange"] == visGroup) & (ses_trials["audioGroupPreChange"] == audioGroup)]
#             plt_save_dir = f"{plot_dir}/Session_{index+1}-{ses_ID}_trial_groups"
#             rplt.raster_plot_superimposed(comb_trials, spikeData, ses_ID, interval, f"Pre-change visual{visGroup} audio{audioGroup}", plt_save_dir)

# # Plotting the binarized Raster plots for each combination of trials before and after change in session 3
# for index, session in sessionData.iterrows():
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     for visGroupPre in ses_trials.visGroupPreChange.unique():
#         for audioGroupPre in ses_trials.audioGroupPreChange.unique():
#             for visGroupPost in ses_trials.visGroupPostChange.unique():
#                 for audioGroupPost in ses_trials.audioGroupPostChange.unique():
#                     comb_trials = ses_trials[(ses_trials["visGroupPreChange"] == visGroupPre) & (ses_trials["audioGroupPreChange"] == audioGroupPre) & (ses_trials["visGroupPostChange"] == visGroupPost) & (ses_trials["audioGroupPostChange"] == audioGroupPost)]
#                     plt_save_dir = f"{plot_dir}/Session_{index+1}-{ses_ID}_trial_groups"
#                     rplt.raster_plot_superimposed(comb_trials, spikeData, ses_ID, interval, f"Pre visual-{visGroupPre} audio-{audioGroupPost}_Post visual-{visGroupPost} audio-{audioGroupPost}", plt_save_dir)


# # Plotting each trial in session 3
# ses_ID = sessionData.loc[2, "session_ID"]
# ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

# for _, trial in ses_trials.iterrows():
#     plt_save_dir = f"{plot_dir}/Session_3-{ses_ID}_individual_test"
#     rplt.raster_plot_individual(trial, spikeData, ses_ID, interval, plt_save_dir)


# # Plotting the Raster plots to be used in the Capstone text
# visGroup, audioGroup = ["225-230", "13000-13020"]
# comb_trials = trialBinData[(trialBinData["visGroupPreChange"] == visGroup) & (trialBinData["audioGroupPreChange"] == audioGroup)]
# trial1 = comb_trials.iloc[0, :]

# rplt.raster_plot_superimposed(
#     comb_trials,
#     spikeData,
#     trial1["session_ID"],
#     interval,
#     f"visual {visGroup} audio {audioGroup} text plot", f"{plot_dir}/main_text",
#     title=False)



########################################################################################################################################################################

# interval = 10000
# spike_file = "3s-1sTrialSpikes.pkl"
# save_dir = "/Users/vojtamazur/Documents/Capstone_code/spike_data/"

# trialSpikesData = pd.read_pickle(f"{save_dir}/{spike_file}")
# trialBinCountData = utils.count_bin_spikes(trialSpikesData, interval)

# filename = f"binSpikeCounts_{int(interval/1000)}ms"
# trialBinCountData.to_pickle(f'{save_dir}/{filename}.pkl')


