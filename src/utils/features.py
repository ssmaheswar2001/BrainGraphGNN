"""
Feature extraction utilities for brain connectivity data
"""

import numpy as np
from scipy import stats


def extract_rich_node_features(ts):
    """
    Extract comprehensive features from time series
    
    Args:
        ts: numpy array of shape (timepoints, n_rois)
    
    Returns:
        numpy array of shape (n_rois, 14 features)
    """
    features = []
    
    # Basic statistics (5 features)
    features.append(np.mean(ts, axis=0))
    features.append(np.std(ts, axis=0))
    features.append(np.median(ts, axis=0))
    features.append(np.percentile(ts, 25, axis=0))
    features.append(np.percentile(ts, 75, axis=0))
    
    # Range statistics (3 features)
    features.append(np.max(ts, axis=0))
    features.append(np.min(ts, axis=0))
    features.append(np.ptp(ts, axis=0))
    
    # Higher order moments (2 features)
    features.append(stats.skew(ts, axis=0))
    features.append(stats.kurtosis(ts, axis=0))
    
    # Frequency domain (2 features)
    fft_features = np.abs(np.fft.fft(ts, axis=0))
    low_freq = np.mean(fft_features[:len(fft_features)//4], axis=0)
    mid_freq = np.mean(fft_features[len(fft_features)//4:len(fft_features)//2], axis=0)
    features.append(low_freq)
    features.append(mid_freq)
    
    # Autocorrelation (1 feature)
    autocorr = []
    for i in range(ts.shape[1]):
        if len(ts[:, i]) > 1:
            try:
                corr_val = np.corrcoef(ts[:-1, i], ts[1:, i])[0, 1]
                autocorr.append(corr_val if not np.isnan(corr_val) else 0)
            except:
                autocorr.append(0)
        else:
            autocorr.append(0)
    features.append(np.array(autocorr))
    
    # Coefficient of variation (1 feature)
    cv = np.std(ts, axis=0) / (np.mean(ts, axis=0) + 1e-8)
    features.append(cv)
    
    return np.column_stack(features)


def adaptive_threshold(corr_matrix, percentile=85, min_edges=50):
    """
    Compute adaptive threshold for edge creation
    
    Args:
        corr_matrix: Correlation matrix
        percentile: Percentile for thresholding
        min_edges: Minimum number of edges to ensure
    
    Returns:
        float: Threshold value
    """
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri = corr_matrix[upper_tri_indices]
    abs_upper = np.abs(upper_tri)
    
    threshold = np.percentile(abs_upper, percentile)
    
    n_edges_above = np.sum(abs_upper > threshold) * 2
    if n_edges_above < min_edges:
        sorted_vals = np.sort(abs_upper)[::-1]
        if len(sorted_vals) > min_edges // 2:
            threshold = sorted_vals[min_edges // 2]
        else:
            threshold = np.min(abs_upper)
    
    return threshold