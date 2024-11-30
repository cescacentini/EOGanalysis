import numpy as np
import mne
from yasa import SleepStaging, hypno_str_to_int, plot_hypnogram

# Load the EDF file
raw = mne.io.read_raw_edf('/Users/francescacentini/Desktop/Life/Academia/02_Research/EOG/mnc/ssc/ssc_1558_1-nsrr.edf', preload=True)

# Check available channels
print("Available channels:", raw.ch_names)

# Select an EEG channel for sleep staging
eeg_candidates = [ch for ch in ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'Cz'] if ch in raw.ch_names]
if eeg_candidates:
    eeg_name = eeg_candidates[0]  # Use the first available EEG channel
    print(f"Using EEG channel for sleep staging: {eeg_name}")
else:
    raise ValueError("No EEG channel found for sleep staging. Please check your data.")

# Perform sleep staging with YASA
sls = SleepStaging(raw, eeg_name=eeg_name)
hypno_pred = sls.predict()  # Predict sleep stages
hypno_pred = hypno_str_to_int(hypno_pred)  # Convert string labels to integers

# Save the hypnogram for later use
np.save("yasa_hypnogram.npy", hypno_pred)

# Plot the hypnogram
plot_hypnogram(hypno_pred)
print("Sleep staging completed and hypnogram saved.")