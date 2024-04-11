import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings

import os


def trim_spikes(trial_df):
    trial_spikes = trial_df["binSpikes"]

    # Step 1: Find the minimum array length across all dictionaries
    min_length = np.inf  # Start with infinity as the minimum length

    for _, trial_dict in trial_spikes.items():
        for arr in trial_dict.values():
            min_length = min(min_length, len(arr))

    # Step 2: Trim arrays in each dictionary to the minimum length
    trimmed_spikes = []
    for trial_dict in trial_spikes:
        trimmed_dict = {neuron: np.array(firing_timestamps)[0:min_length]
                        for neuron, firing_timestamps in trial_dict.items()}
        trimmed_spikes.append(trimmed_dict)

    return pd.Series(trimmed_spikes), min_length


def raster_plot_superimposed(trial_df, spike_df, session_ID, interval, trial_selection, path, title=True):
    # Replace the original 'trial_spikes' with the trimmed version
    # which trims all trials to the length of the shortest trial
    trial_spikes, min_length = trim_spikes(trial_df)

    # Filter out neurons from other sessions
    sessionNeurons = spike_df[spike_df["session_ID"] == session_ID]
    neuron_series = sessionNeurons["cell_ID"]

    f_trial_spikes = []

    for index, trial in trial_spikes.items():
        filtered_trial = {k: v for k, v in trial.items() if k in neuron_series.values}
        f_trial_spikes.append(filtered_trial)

    trial_spikes = pd.Series(f_trial_spikes)

    # Additively superimpose the spike data from each trial
    first_trial = trial_spikes.iloc[0]
    total_spikes = np.zeros((len(first_trial), len(first_trial[f"{next(iter(first_trial))}"])))

    for _, trial in trial_spikes.items():
        neuron_arr = np.array(list(trial.values()))
        total_spikes += neuron_arr

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Generate the plot of superimposed spiking data
    plt.imshow(total_spikes, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of spikes in time bin')  # Show color scale

    end_time = int(min_length * (interval/1000))

    # Create the title and axis labels and ticks
    plt.xlabel('Time since start of trial (ms)')
    plt.ylabel('Neuron')
    plt.xticks(ticks=range(0, total_spikes.shape[1], 50), labels=range(0, end_time, 500))
    plt.yticks(ticks=range(total_spikes.shape[0]), labels=neuron_series, fontsize=8)

    # Only include title if the boolean is True
    if title:
        plt.title(f'Superimposed Raster Plot: {trial_selection},\n {interval/1000}ms time bins ({len(trial_spikes)} trials)', loc='left')
    else:
        plt.title('A superimposed Raster plot of binarized neuron firings', fontsize=16)

    # Color the y-axis labels based on which brain area they belong to
    tick_labels = plt.gca().get_yticklabels()
    label_colors = ["blue", "green", "magenta"]
    for tick_label in tick_labels:
        neuron_ID = tick_label.get_text()
        row_index = spike_df.index[spike_df["cell_ID"] == neuron_ID].to_list()
        area = spike_df.at[row_index[0], "area"]
        if area == "V1":
            tick_label.set_color(label_colors[0])
        elif area == "CG1":
            tick_label.set_color(label_colors[1])
        else:
            tick_label.set_color(label_colors[2])

    # Add an additional legend to explain the different brain areas
    area_labels = ["V1", "AC1", "PPC"]
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(label_colors, area_labels)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='best', title="Brain Area")

    # Add a label for the stimulus change:
    stim_change = 2000000/interval
    plt.axvline(x=stim_change, color='r', linestyle='--', linewidth=1)
    plt.text(stim_change, len(tick_labels)-3, 'Stimulus\n Change', rotation=90, color='r', verticalalignment='bottom')

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the figure in the save directory with the given filename
    filename = f"{path}/{trial_selection.replace(' ', '_')}_{interval/1000}ms.png"
    plt.savefig(filename)

    print(f"Plot saved as {filename}")


def raster_plot_individual(trial, spike_df, session_ID, interval, path):
    # Convert the spike data into an array for easy plotting
    trial_spikes = trial["binSpikes"]
    neuron_arr = np.array(list(trial_spikes.values()))

    # Set up the plot
    plt.figure(figsize=(16, 6))

    # Generate the plot of superimposed spiking data
    plt.imshow(neuron_arr, aspect='auto', cmap='viridis', interpolation='nearest')

    end_time = int(neuron_arr.shape[1] * (interval/1000))

    # Create the title and axis labels and ticks
    plt.xlabel('Time since start of trial (ms)')
    plt.ylabel('Neuron')
    plt.xticks(ticks=range(0, neuron_arr.shape[1], 50), labels=range(0, end_time, 500))
    plt.yticks(ticks=range(neuron_arr.shape[0]), labels=trial_spikes.keys())
    plt.title(f"Binarized Raster plot of trial {trial['trialNum']} in session {session_ID}")

    # Color the y-axis labels based on which brain area they belong to
    tick_labels = plt.gca().get_yticklabels()
    label_colors = ["blue", "green", "magenta"]
    for tick_label in tick_labels:
        neuron_ID = tick_label.get_text()
        row_index = spike_df.index[spike_df["cell_ID"] == neuron_ID].to_list()
        area = spike_df.at[row_index[0], "area"]
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

    # Add a label for the stimulus change:
    stim_change = 2000000/interval
    plt.axvline(x=stim_change, color='r', linestyle='--', linewidth=1)
    plt.text(stim_change, -1, 'Stimulus\n Change', rotation=90, color='r', verticalalignment='bottom')

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/trial{trial['trialNum']}_individual.png")
    print(f"Trial {trial['trialNum']} done")



def generate_raster_plot(df, start_time, stop_time, session_ID, path, filename=None):
    """
    Generates a Raster plot from neuron firing data.

    :param df: DataFrame where each row represents one neuron, and the 'ts' column contains
               arrays of timestamps (in microseconds) of the neuron's spikes.
    :param start_time: Start time of the experiment (in microseconds).
    :param stop_time: Stop time of the experiment (in microseconds).
    """
    # Set up the plot
    plt.figure(figsize=(24, 8))
    plt.title("Neuron Firing Raster Plot")
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Neuron")

    # Adjust the time axis to reflect the duration of the experiment
    plt.xlim(start_time, stop_time)

    # Iterate over each neuron and plot its spikes
    for index, row in df.iterrows():
        neuron_id = index + 1  # Assuming neuron IDs are 1-indexed based on row position
        spike_times = row['ts']
        plt.scatter(spike_times, [neuron_id] * len(spike_times), marker='|')

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if a filename was given, otherwise generate one automatically
    if not filename:
        filename = f"raster_plot-{session_ID}"

    plt.savefig(f"{path}/{filename}.png")



def generate_binarized_raster_plot(df, session_ID, path, filename=None):
    """
    Generates a Raster plot from binarized neuron firing data.

    :param df: DataFrame where each row represents one neuron, and the 'binarized_ts' column contains
               arrays of binarized firing data for each neuron over discrete time intervals.
    """
    # Set up the plot
    plt.figure(figsize=(6, 4))
    plt.title("Binarized Neuron Firing Raster Plot")
    plt.xlabel("Time Interval")
    plt.ylabel("Neuron")

    # Iterate over each neuron and plot its firing intervals
    for index, row in df.iterrows():
        neuron_id = index + 1  # Assuming neuron IDs are 1-indexed based on row position
        firing_intervals = np.where(row['binarized_ts'] == 1)[0]  # Get indices of intervals where neuron fired
        plt.scatter(firing_intervals, [neuron_id] * len(firing_intervals), marker='|')
        print(firing_intervals)

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    if not filename:
        filename = f"raster_plot-{session_ID}"

    plt.savefig(f"{path}/{filename}.png")


def generate_raster_plot_trial(df, start_time, stop_time, session_ID, path, filename=None):
    """
    Generates a Raster plot from neuron firing data.

    :param df: DataFrame where each row represents one neuron, and the 'ts' column contains
               arrays of timestamps (in microseconds) of the neuron's spikes.
    :param start_time: Start time of the experiment (in microseconds).
    :param stop_time: Stop time of the experiment (in microseconds).
    """
    # Set up the plot
    plt.figure(figsize=(24, 8))
    plt.title("Neuron Firing Raster Plot")
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Neuron")

    # Adjust the time axis to reflect the duration of the experiment
    plt.xlim(start_time, stop_time)

    # Iterate over each neuron and plot its spikes
    for index, row in df.iterrows():
        neuron_id = index + 1  # Assuming neuron IDs are 1-indexed based on row position
        spike_times = row['ts']
        trial_spike_times = spike_times[(spike_times >= start_time) & (spike_times <= stop_time)]
        print((spike_times >= start_time) & (spike_times <= stop_time))
        plt.scatter(trial_spike_times, [neuron_id] * len(trial_spike_times), marker='|')

    # Check if the save directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if a filename was given, otherwise generate one automatically
    if not filename:
        filename = f"raster_plot-{session_ID}"

    plt.savefig(f"{path}/{filename}.png")















# for i,s in sessionData.iterrows():
#     neurons = spikeData[spikeData["session_ID"] == s["session_ID"]]
#     generate_raster_plot(neurons, s["t_start"], s["t_stop"], s["session_ID"], savepath)


# s = sessionData.iloc[0,:]
# s_id = s["session_ID"]
# neurons = spikeData[spikeData["session_ID"] == s_id]
# savepath = f"/Users/vojtamazur/Documents/Capstone_code/raster_plots/session_{s_id}"
# time_interval = 60000000
# n_plots = int((s["t_stop"] - s["t_start"])/time_interval)

# for t in range(n_plots):
#     generate_raster_plot(neurons, s["t_start"]+((t)*time_interval), s["t_start"]+((t+1)*time_interval), s["session_ID"], savepath, filename=f"r_plot_{t}")


# pkl_path = "/Users/vojtamazur/Documents/Capstone_code/bin_spike_data/test_23-02-2024_11-29-15.pkl"
# binDF = pd.read_pickle(pkl_path)
# savepath = f"/Users/vojtamazur/Documents/Capstone_code/raster_plots/binarized"
# binDF["binarized_ts"] = binDF["binarized_ts"].apply(lambda x: np.array([int(i) for i in x]))

# print(binDF["binarized_ts"][0][0])

# for i,s in sessionData.iterrows():
#     neurons = binDF[binDF["session_ID"] == s["session_ID"]]
#     generate_binarized_raster_plot(neurons, s["session_ID"], savepath)

# savepath = "/Users/vojtamazur/Documents/Capstone_code/raster_plots/trials"
# session = sessionData.loc[0, "session_ID"]
# trialData_ses = trialData[trialData["session_ID"] == session]
# spikeData_ses = spikeData[spikeData["session_ID"] == session]

# for index, trial in trialData_ses.iterrows():
#     # Generate and save a Raster plot around 1 trial
#     generate_raster_plot_trial(spikeData_ses, trial["trialStart"], trial["trialEnd"], trial["session_ID"], savepath, f"trial_{index}")
