import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy, kurtosis, skew

def extract_time_domain_features(signal):
    features = {
        'signal_variance': np.var(signal),
        'signal_iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'energy': np.sum(np.square(signal)),
        'median': np.median(signal),
        'entropy': entropy(np.abs(signal))
    }

    # Calculate the slope of the signal
    if len(signal) > 1:
        signal_slope = np.diff(signal)  # First derivative (slope)
        features['mean_slope'] = np.mean(signal_slope)
        features['std_slope'] = np.std(signal_slope)
    else:
        features['mean_slope'] = 0
        features['std_slope'] = 0

        
    return pd.Series(features)

def extract_frequency_domain_features(signal):
    fft_vals = fft(signal)
    fft_magnitude = np.abs(fft_vals)
    power = np.square(fft_magnitude)

    features = {
        'fft_max_freq': np.argmax(fft_magnitude),
        'fft_mean_power': np.mean(power),
        'fft_peak_power': np.max(power)
    }
    return pd.Series(features)

def extract_features(hEOG, vEOG):
    # Time-domain features
    hEOG_time_features = hEOG.apply(extract_time_domain_features, axis=1)
    hEOG_time_features.columns = ['h_' + col for col in hEOG_time_features.columns]
    vEOG_time_features = vEOG.apply(extract_time_domain_features, axis=1)
    vEOG_time_features.columns = ['v_' + col for col in vEOG_time_features.columns]

    # Frequency-domain features
    hEOG_freq_features = hEOG.apply(lambda row: extract_frequency_domain_features(row.values), axis=1)
    hEOG_freq_features.columns = ['h_' + col for col in hEOG_freq_features.columns]
    vEOG_freq_features = vEOG.apply(lambda row: extract_frequency_domain_features(row.values), axis=1)
    vEOG_freq_features.columns = ['v_' + col for col in vEOG_freq_features.columns]

    # Combine features
    combined_time_features = pd.concat([hEOG_time_features, vEOG_time_features], axis=1)
    combined_freq_features = pd.concat([hEOG_freq_features, vEOG_freq_features], axis=1)

    combined_features = pd.concat([combined_time_features, combined_freq_features], axis=1)

    return combined_features
