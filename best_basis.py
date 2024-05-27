import numpy as np
import pandas as pd
import MinCompSpin_Python.MinCompSpin as mod
import models

# Imports for getting the session data
import utils
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Getting the saved binarized data
time_bin = 20
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


data_dir = "./binData"


######################## Find the session-wide best basis ####################################

# for index, session in sessionData.iterrows():
#     # get the trials from this session
#     ses_ID = session["session_ID"]
#     ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

#     # get the neurons from the session and their number
#     ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
#     neuron_series = ses_neurons["cell_ID"]
#     n = len(neuron_series)

#      # Converting the data into a format usable by the MCM
#     filename = f"session{ses_ID}_full"
#     models.create_input_file(ses_trials, 0, int(2000/time_bin)-1, filename, data_dir)


############################################################################################################
######################## Examine the 3 neuron interactions ##################################################

binarized_spikes = trialBinData["binSpikes"]

# # Session 4
# session_1 = sessionData.loc[0, "session_ID"]
# ses_neurons = spikeData[spikeData["session_ID"] == session_1]
# neuron_arr = np.array(ses_neurons["cell_ID"])
# n1, n2, n3 = [3, 39, 83]
# three_neurons = [neuron_arr[n1], neuron_arr[n2], neuron_arr[n3]]

# neuron1 = []
# neuron2 = []
# neuron3 = []

# for _, trial in binarized_spikes.items():
#     for key, value in trial.items():
#         if key == three_neurons[0]:
#             for data_point in value:
#                 neuron1.append(data_point)
#         if key == three_neurons[1]:
#             for data_point in value:
#                 neuron2.append(data_point)
#         if key == three_neurons[2]:
#             for data_point in value:
#                 neuron3.append(data_point)

# print(f"The firing frequency of neuron {n1+1}: {np.sum(neuron1)/len(neuron1)}")
# print(f"The firing frequency of neuron {n2+1}: {np.sum(neuron2)/len(neuron2)}")
# print(f"The firing frequency of neuron {n3+1}: {np.sum(neuron3)/len(neuron3)}\n")

# correlation_matrix = np.corrcoef([neuron1, neuron2, neuron3])
# print(correlation_matrix)

# data_li = []

# ses_trialDF = trialBinData[trialBinData["session_ID"] == session_1]
# bin_spikes = ses_trialDF["binSpikes"]

# for _, trial in bin_spikes.items():
#     neuron_arr = np.array(list(trial.values()))
#     t = np.transpose(neuron_arr)
#     data_li.append(np.transpose(neuron_arr))

# data_t = tuple(data_li)
# data_arr = np.concatenate(data_t)

# t_array = np.transpose(data_arr)

# freq_list = []
# for row in t_array:
#     freq_list.append(np.sum(row)/len(row))

# print(np.mean(freq_list))




############################################################################################################
######################## Examine the combination frequencies in the 3 neuron interactions ##################################################


ses_ID = sessionData.loc[3, "session_ID"]
ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
neuron_arr = np.array(ses_neurons["cell_ID"])
n1, n2, n3 = [20, 16, 5]
three_neurons = [neuron_arr[n1], neuron_arr[n2], neuron_arr[n3]]

data_file = f"/Users/vojtamazur/Documents/Capstone_code/MinCompSpin_BasisSearch128-main/INPUT/session{ses_ID}_full.dat"

with open(data_file, "r") as file:
    rows = file.readlines()
    data = []

    for row in rows:
        data.append([int(char) for char in row if char != "\n"])

trans_data = np.transpose(np.array(data))

rel_neuron_data = np.transpose(trans_data[[n1, n2, n3]])

count = np.zeros(8)

for row in rel_neuron_data:
    if ((row[0] == 0) & (row[1] == 0)) & (row[2] == 0):
        count[0] += 1
    elif ((row[0] == 0) & (row[1] == 0)) & (row[2] == 1):
        count[1] += 1
    elif ((row[0] == 0) & (row[1] == 1)) & (row[2] == 0):
        count[2] += 1
    elif ((row[0] == 0) & (row[1] == 1)) & (row[2] == 1):
        count[3] += 1
    elif ((row[0] == 1) & (row[1] == 0)) & (row[2] == 0):
        count[4] += 1
    elif ((row[0] == 1) & (row[1] == 0)) & (row[2] == 1):
        count[5] += 1
    elif ((row[0] == 1) & (row[1] == 1)) & (row[2] == 0):
        count[6] += 1
    elif ((row[0] == 1) & (row[1] == 1)) & (row[2] == 1):
        count[1] += 1

frequencies = count/len(rel_neuron_data)
print(frequencies)

