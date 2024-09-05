# EEG Signal Analysis for Attention Identification Using LSTM

## Dataset Overview

The dataset used in this project comprises EEG signals collected from subjects to analyze their attention levels. EEG data typically includes continuous recordings of brain wave activity, measured from various electrodes placed on the scalp. For this project, the data is provided in CSV format, where each row represents a time point with features corresponding to different EEG channels and a label indicating the attention level.

The dataset may be obtained from publicly available sources or recorded in a controlled environment, capturing brain activity under specific conditions. The data may include various artifacts and noise, which requires preprocessing and cleaning to ensure accurate analysis. Real-time EEG data handling introduces additional complexities such as streaming data management and real-time processing requirements.

This project outlines a simplified approach to EEG signal analysis using LSTM models to identify attention levels. The provided code serves as a demonstration and is intended to give an overview of the key steps involved in the process.

## Overview

**Objective:** To classify attention levels from EEG signals using an LSTM neural network.

**Data:** The EEG data is assumed to be in CSV format, with signals as features and attention levels as labels. EEG data can also be collected in real-time, providing a continuous stream of brain activity data. Real-time EEG data enables dynamic monitoring and analysis of cognitive states, but handling such data requires additional considerations like data acquisition, real-time processing, and latency management.

**Techniques Used:**
- **Band-Pass Filtering:** To isolate relevant frequency ranges in EEG data.
- **LSTM Model:** For sequential data processing to capture temporal dependencies in EEG signals.

## Important Note

**Disclaimer:** This code is intended as an outline for demonstration purposes only. It serves as a basic template and should be expanded and adjusted according to the specific needs of your project, including proper data handling, feature engineering, model tuning, and performance evaluation.


## Sample Code

The following Python code demonstrates the process of loading, preprocessing EEG data, and training an LSTM model to classify attention levels.

### Import Libraries

```python
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_eeg_data(file_path):
    """
    Load EEG data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file containing EEG data and labels.

    Returns:
    - eeg_signals: Numpy array of EEG signals.
    - labels: Numpy array of attention level labels.
    """
    data = pd.read_csv(file_path)
    eeg_signals = data.iloc[:, :-1].values  # All columns except the last one
    labels = data.iloc[:, -1].values        # Last column as labels
    return eeg_signals, labels



def bandpass_filter(data, lowcut, highcut, fs):
    """
    Apply band-pass filter to EEG signals.

    Parameters:
    - data: Numpy array of EEG signals.
    - lowcut: Low cutoff frequency in Hz.
    - highcut: High cutoff frequency in Hz.
    - fs: Sampling frequency in Hz.

    Returns:
    - Filtered EEG signals as a Numpy array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def preprocess_data(signals):
    """
    Preprocess EEG signals with band-pass filtering and scaling.

    Parameters:
    - signals: Numpy array of raw EEG signals.

    Returns:
    - Processed signals as a Numpy array.
    """
    filtered = bandpass_filter(signals, 0.5, 30, fs=100)
    return StandardScaler().fit_transform(filtered)

def build_lstm(input_shape):
    """
    Build an LSTM model for attention classification.

    Parameters:
    - input_shape: Tuple representing the input shape of the LSTM model.

    Returns:
    - Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load and preprocess data
    eeg_signals, labels = load_eeg_data('eeg_data.csv')
    X = preprocess_data(eeg_signals).reshape((-1, 1, eeg_signals.shape[1]))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Build and train the LSTM model
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.2f}')

