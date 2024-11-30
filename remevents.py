import numpy as np
import pandas as pd
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
    hypno_upsampled = hypno_upsampled[:eog_data_length]
elif len(hypno_upsampled) < eog_data_length:
    hypno_upsampled = np.pad(hypno_upsampled, (0, eog_data_length - len(hypno_upsampled)), constant_values=-2)

# Extract EOG channels
e1 = raw.get_data(picks='E1')[0]
e2 = raw.get_data(picks='E2')[0]

# Filter the EOG signals
low_cut = 0.3
high_cut = 50.0
e1_filtered = filter_data(e1, sf, low_cut, high_cut)
e2_filtered = filter_data(e2, sf, low_cut, high_cut)

# Detect saccades across sleep stages
rem = rem_detect(
    e1_filtered, e2_filtered, sf,
    hypno=hypno_upsampled,
    include=[1, 2, 3, 4],  # Include NREM1, NREM2, NREM3, REM
    amplitude=(30, 600),       # Relaxed amplitude range
    duration=(0.03, 1.0),      # Wider duration range
    freq_rem=(0.3, 8),         # Wider frequency range
    remove_outliers=False,     # Allow more detections
    verbose=True
)

if rem is not None:
    # Extract REM event summary
    rem_summary = rem.summary()

    # Map detected saccades to their respective sleep stages
    rem_summary['SleepStage'] = rem_summary['Peak'].apply(
        lambda peak: hypno_upsampled[int(peak * sf)]  # Map hypnogram using the Peak time in seconds
    )

    # Count saccades by sleep stage
    saccade_counts = rem_summary['SleepStage'].value_counts()

    # Calculate total time spent in each sleep stage
    stage_durations = pd.Series(hypno_upsampled).value_counts()  # Count samples per stage
    stage_durations = stage_durations / sf / 60  # Convert samples to minutes

    # Map stage labels
    stage_labels = {4: 'REM', 3: 'NREM3', 2: 'NREM2', 1: 'NREM1'}
    saccade_counts.index = saccade_counts.index.map(stage_labels)
    stage_durations.index = stage_durations.index.map(stage_labels)

    # Align counts and durations
    saccade_counts = saccade_counts.reindex(stage_durations.index, fill_value=0)

    # Normalize saccades by time spent in each stage
    average_saccade_rate = saccade_counts / stage_durations

    # Create a DataFrame for visualization
    normalized_saccade_rates = pd.DataFrame({
        'Average Saccades per Minute': average_saccade_rate
    }).fillna(0)

    # Plot the normalized saccade rates
    normalized_saccade_rates.plot(kind='bar', figsize=(10, 6), title='Normalized Saccades per Minute by Sleep Stage')
    plt.xlabel('Sleep Stage')
    plt.ylabel('Saccades per Minute')
    plt.legend(['Saccades per Minute'])
    plt.tight_layout()
    plt.show()

    # Debugging outputs
    print("\nDebugging Outputs:")
    print("Time Spent in Each Stage (minutes):")
    print(stage_durations)
    print("\nTotal Saccades per Stage:")
    print(saccade_counts)
    print("\nNormalized Saccade Rates (per minute):")
    print(normalized_saccade_rates)
else:
    print("No saccades detected.")