#!/usr/bin/env python3
"""
Script to evaluate a trained BrainGNN model
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import BrainDataset
from src.models import GCN_Classifier, GAT_Classifier, GIN_Classifier, GraphSAGE_Classifier, HGNN_Classifier
from src.training import evaluate_model
from src.utils import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate BrainGNN model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--architecture', type=str, default='Hybrid',
                       choices=['GCN', 'GAT', 'GIN', 'GraphSAGE', 'Hybrid'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
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
    
    # Evaluate
    print("Evaluating model...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_model(model, loader, device)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print("="*60 + "\n")
    
    # Plot confusion matrix
    os.makedirs('figures', exist_ok=True)
    plot_confusion_matrix(metrics['labels'], metrics['predictions'], 'figures/confusion_matrix.png')
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()