"""
Download and preprocess fMRI data
"""

import os
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure


def download_and_preprocess_data(
    dataset_path="dataset",
    atlas_name="msdl",
    standardize=True
):
    """
    Download fMRI data and preprocess it for GNN training.
    
    Args:
        dataset_path (str): Path to save dataset
        atlas_name (str): Atlas to use (default: 'msdl')
        standardize (bool): Whether to standardize time series
        
    Returns:
        dict: Contains time_series and labels
    """
    
    print("="*80)
    print("DOWNLOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Create directories
    corr_matrices_dir = f'{dataset_path}/corr_matrices'
    pcorr_matrices_dir = f'{dataset_path}/pcorr_matrices'
    tan_matrices_dir = f'{dataset_path}/tan_matrices'
    time_series_dir = f'{dataset_path}/time_series'
    labels_file = f'{dataset_path}/labels.csv'
    avg_pcorr_file = f'{dataset_path}/avg_pcorr.csv'
    
    os.makedirs(corr_matrices_dir, exist_ok=True)
    os.makedirs(pcorr_matrices_dir, exist_ok=True)
    os.makedirs(tan_matrices_dir, exist_ok=True)
    os.makedirs(time_series_dir, exist_ok=True)
    
    # Fetch atlas
    print("\nFetching atlas...")
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas.maps
    atlas_labels = atlas.labels
    
    print(f"Atlas: {atlas_name.upper()}")
    print(f"Number of ROIs: {len(atlas_labels)}")
    
    # Fetch fMRI data
    print("\nFetching fMRI data...")
    data = datasets.fetch_development_fmri()
    print(f"Number of subjects: {len(data.func)}")
    
    # Create masker
    print("\nExtracting time series...")
    masker = NiftiMapsMasker(
        maps_img=atlas_filename,
        standardize=standardize,
        memory='nilearn_cache'
    )
    
    # Extract time series
    time_series = [None] * len(data.func)
    labels = [None] * len(data.func)
    
    for i in range(len(data.func)):
        # Get subject number to avoid ordering issues
        sub_num = int(data.phenotypic.iloc[i]['participant_id'][9:]) - 1
        
        # Extract time series
        ts = masker.fit_transform(data.func[i], confounds=data.confounds[i])
        time_series[sub_num] = ts
        labels[sub_num] = data.phenotypic.iloc[i]['Child_Adult']
        
        # Save time series
        np.savetxt(
            f'{time_series_dir}/time_series_{sub_num:04d}.csv',
            ts,
            delimiter=','
        )
    
    print(f"✓ Time series extracted and saved to {time_series_dir}")
    
    # Compute connectivity matrices
    print("\nComputing connectivity matrices...")
    
    corr_measure = ConnectivityMeasure(kind='correlation')
    pcorr_measure = ConnectivityMeasure(kind='partial correlation')
    tan_measure = ConnectivityMeasure(kind='tangent')
    
    print("  - Correlation...")
    corr_matrices = corr_measure.fit_transform(time_series)
    print("  - Partial correlation...")
    pcorr_matrices = pcorr_measure.fit_transform(time_series)
    print("  - Tangent...")
    tan_matrices = tan_measure.fit_transform(time_series)
    
    # Save average partial correlation matrix
    avg_pcorr_matrix = np.mean(pcorr_matrices, axis=0)
    np.savetxt(avg_pcorr_file, avg_pcorr_matrix, delimiter=',')
    
    # Save connectivity matrices
    print("\nSaving connectivity matrices...")
    for i in range(len(corr_matrices)):
        np.savetxt(
            f'{corr_matrices_dir}/corr_{i:04d}.csv',
            corr_matrices[i],
            delimiter=','
        )
        np.savetxt(
            f'{pcorr_matrices_dir}/pcorr_{i:04d}.csv',
            pcorr_matrices[i],
            delimiter=','
        )
        np.savetxt(
            f'{tan_matrices_dir}/tan_{i:04d}.csv',
            tan_matrices[i],
            delimiter=','
        )
    
    # Save labels
    label_nums = [0 if label == 'child' else 1 for label in labels]
    np.savetxt(labels_file, np.asarray(label_nums).astype(int), delimiter=',')
    
    # Print statistics
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Total subjects: {len(time_series)}")
    print(f"Children: {labels.count('child')}")
    print(f"Adults: {labels.count('adult')}")
    print(f"Time series shape: {time_series[0].shape}")
    print(f"\nData saved to: {dataset_path}/")
    print("="*80 + "\n")
    
    return {
        'time_series': time_series,
        'labels': labels,
        'atlas_labels': atlas_labels,
        'n_rois': len(atlas_labels)
    }


if __name__ == "__main__":
    download_and_preprocess_data()