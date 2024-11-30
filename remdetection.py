import numpy as np
import mne
from mne.filter import filter_data
from yasa import rem_detect
import matplotlib.pyplot as plt

# Load the EDF file
raw = mne.io.read_raw_edf('/Users/francescacentini/Desktop/Life/Academia/02_Research/EOG/mnc/ssc/ssc_1558_1-nsrr.edf', preload=True)

# Load the saved hypnogram
hypno_pred = np.load("yasa_hypnogram.npy")

# Upsample the hypnogram to match the length of the EOG data
sf = raw.info['sfreq']  # Sampling frequency (e.g., 256 Hz)
eog_data_length = raw.n_times  # Total number of samples in the EOG data
hypno_upsampled = np.repeat(hypno_pred, int(sf * 30))  # Repeat each epoch value for 30 * sf samples

# Adjust the hypnogram length to match EOG data
if len(hypno_upsampled) > eog_data_length:
    hypno_upsampled = hypno_upsampled[:eog_data_length]  # Truncate if too long
elif len(hypno_upsampled) < eog_data_length:
    hypno_upsampled = np.pad(hypno_upsampled, (0, eog_data_length - len(hypno_upsampled)), constant_values=-2)  # Pad with -2 (unscored)

# Map sleep stage integers to colors for visualization
stage_colors = {-2: "grey", -1: "orange", 0: "white", 1: "lightblue", 2: "blue", 3: "darkblue", 4: "pink"}
stage_labels = {-2: "Unscored", -1: "Artefact", 0: "Wake", 1: "NREM1", 2: "NREM2", 3: "NREM3", 4: "REM"}

# Extract EOG channels
e1 = raw.get_data(picks='E1')[0]
e2 = raw.get_data(picks='E2')[0]

# Filter the EOG signals
low_cut = 0.3
high_cut = 50.0
e1_filtered = filter_data(e1, sf, low_cut, high_cut)
e2_filtered = filter_data(e2, sf, low_cut, high_cut)

# Adjust REM detection parameters
rem = rem_detect(
    e1_filtered, e2_filtered, sf,
    hypno=hypno_upsampled,  # Use adjusted hypnogram
    include=4,  # Focus on REM stage
    amplitude=(30, 600),       # Relaxed amplitude range
    duration=(0.03, 1.0),      # Wider duration range
    freq_rem=(0.3, 8),         # Wider frequency range for REM saccades
    remove_outliers=False,     # Disable outlier removal to allow more detections
    verbose=True
)

# Create a mask for REM highlights
if rem is not None:
    mask = rem.get_mask()
    e1_highlight = e1_filtered * mask[0, :]
    e2_highlight = e2_filtered * mask[1, :]
    e1_highlight[e1_highlight == 0] = np.nan
    e2_highlight[e2_highlight == 0] = np.nan
else:
    e1_highlight = e2_highlight = None

# Create a time vector
times = np.arange(len(e1_filtered)) / sf

# Plot EOG signals with REM highlights and sleep stages
fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot E1
ax[0].plot(times, e1_filtered, label='E1 (Filtered)', color='grey')
if e1_highlight is not None:
    ax[0].plot(times, e1_highlight, 'indianred', label='REM Detected')
for stage, color in stage_colors.items():
    ax[0].fill_between(times, ax[0].get_ylim()[0], ax[0].get_ylim()[1],
                       where=(hypno_upsampled == stage), facecolor=color, alpha=0.2, label=stage_labels[stage])
ax[0].set_ylabel('Amplitude (uV)')
ax[0].set_title('E1 Eye Signal with Sleep Stages and REM Detection')
ax[0].legend(loc='upper right', frameon=False)

# Plot E2
ax[1].plot(times, e2_filtered, label='E2 (Filtered)', color='grey')
if e2_highlight is not None:
    ax[1].plot(times, e2_highlight, 'indianred', label='REM Detected')
for stage, color in stage_colors.items():
    ax[1].fill_between(times, ax[1].get_ylim()[0], ax[1].get_ylim()[1],
                       where=(hypno_upsampled == stage), facecolor=color, alpha=0.2, label=stage_labels[stage])
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (uV)')
ax[1].set_title('E2 Eye Signal with Sleep Stages and REM Detection')
ax[1].legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.show()