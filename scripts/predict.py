#!/usr/bin/env python3
"""
Script to make predictions with a trained BrainGNN model
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
from torch.nn import functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import BrainDataset
from src.models import GCN_Classifier, GAT_Classifier, GIN_Classifier, GraphSAGE_Classifier, HGNN_Classifier


def main():
    parser = argparse.ArgumentParser(description='Make predictions with BrainGNN model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--architecture', type=str, default='Hybrid',
                       choices=['GCN', 'GAT', 'GIN', 'GraphSAGE', 'Hybrid'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--data-idx', type=int, default=0, help='Index of data to predict')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    time_series_dir = config['data']['time_series_dir']
    labels_file = config['data']['labels_file']
    
    time_series = []
    labels_array = np.loadtxt(labels_file, delimiter=',').astype(int)
    
    for i in range(len(labels_array)):
        ts_file = f"{time_series_dir}/time_series_{i:04d}.csv"
        if os.path.exists(ts_file):
            ts = np.loadtxt(ts_file, delimiter=',')
            time_series.append(ts)
    
    labels = ['child' if l == 0 else 'adult' for l in labels_array]
    
    dataset = BrainDataset(
        root='brain_dataset_pyg',
        time_series=time_series,
        labels=labels,
        threshold_percentile=config['data']['threshold_percentile'],
        normalize_features=config['data']['normalize_features']
    )
    
    print(f"✓ Loaded {len(dataset)} graphs\n")
    
    # Load model
    print(f"Loading {args.architecture} model from {args.model_path}...")
    
    model_classes = {
        'GCN': GCN_Classifier,
        'GAT': GAT_Classifier,
        'GIN': GIN_Classifier,
        'GraphSAGE': GraphSAGE_Classifier,
        'Hybrid': HGNN_Classifier
    }
    
    model_params = config['model']['architectures'][args.architecture]
    model = model_classes[args.architecture](
        num_node_features=dataset[0].x.shape[1],
        num_classes=2,
        **model_params
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded\n")
    
    # Make prediction
    sample_data = dataset[args.data_idx].to(device)
    
    with torch.no_grad():
        # Create a batch with single sample
        from torch_geometric.data import Batch
        batch_data = Batch.from_data_list([sample_data])
        
        output = model(batch_data)
        probs = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1)
    
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Sample index: {args.data_idx}")
    print(f"True label:   {'Child' if sample_data.y.item() == 0 else 'Adult'}")
    print(f"Predicted:    {'Child' if prediction.item() == 0 else 'Adult'}")
    print(f"Confidence:   {probs.max().item():.4f}")
    print(f"\nProbabilities:")
    print(f"  Child: {probs[0, 0].item():.4f}")
    print(f"  Adult: {probs[0, 1].item():.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()