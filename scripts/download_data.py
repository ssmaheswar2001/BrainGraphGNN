#!/usr/bin/env python3
"""
Script to download and preprocess fMRI data
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.download_data import download_and_preprocess_data


def main():
    print("\n" + "="*80)
    print("BrainGNN Data Download and Preprocessing")
    print("="*80 + "\n")
    
    # Download and preprocess
    data = download_and_preprocess_data(
        dataset_path='dataset',
        atlas_name='msdl',
        standardize=True
    )
    
    print("\n✓ Data download and preprocessing complete!")
    print(f"  Total subjects: {len(data['time_series'])}")
    print(f"  ROIs: {data['n_rois']}")
    print(f"  Time series shape: {data['time_series'][0].shape}")
    print("\nYou can now run: python scripts/train.py")


if __name__ == "__main__":
    main()