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

for index, session in sessionData.iterrows():
    # get the trials from this session
    ses_ID = session["session_ID"]
    ses_trials = trialBinData[trialBinData["session_ID"] == ses_ID]

    # get the neurons from the session and their number
    ses_neurons = spikeData[spikeData["session_ID"] == ses_ID]
    neuron_series = ses_neurons["cell_ID"]
    n = len(neuron_series)

     # Converting the data into a format usable by the MCM
    filename = f"session{ses_ID}_full"
    models.create_input_file(ses_trials, 0, int(2000/time_bin)-1, filename, data_dir)
