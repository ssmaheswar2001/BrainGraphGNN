#!/usr/bin/env python3
"""
Script to train BrainGNN models with k-fold cross-validation
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import BrainDataset
from src.models import GCN_Classifier, GAT_Classifier, GIN_Classifier, GraphSAGE_Classifier, HGNN_Classifier
from src.training import train_model, create_balanced_loader, evaluate_model
from src.utils import plot_architecture_comparison, plot_performance_heatmap


# Define all architectures
ARCHITECTURES = {
    'GCN': {
        'class': GCN_Classifier,
        'params': {'hidden_channels': 128, 'num_layers': 4, 'dropout': 0.3, 'use_residual': True},
        'description': 'Standard GCN with residual connections'
    },
    'GAT': {
        'class': GAT_Classifier,
        'params': {'hidden_channels': 128, 'num_layers': 4, 'heads': 4, 'dropout': 0.3},
        'description': 'Graph Attention Network with multi-head attention'
    },
    'GIN': {
        'class': GIN_Classifier,
        'params': {'hidden_channels': 128, 'num_layers': 4, 'dropout': 0.3},
        'description': 'Graph Isomorphism Network'
    },
    'GraphSAGE': {
        'class': GraphSAGE_Classifier,
        'params': {'hidden_channels': 128, 'num_layers': 4, 'dropout': 0.3, 'aggr': 'mean'},
        'description': 'GraphSAGE with neighborhood sampling'
    },
    'Hybrid': {
        'class': HGNN_Classifier,
        'params': {'hidden_channels': 128, 'dropout': 0.3, 'heads': 4},
        'description': 'Hybrid combining GCN, GAT, GIN, and SAGE'
    }
}


def load_dataset(config):
    """Load preprocessed dataset"""
    import numpy as np
    
    # Load time series
    time_series_dir = config['data']['time_series_dir']
    labels_file = config['data']['labels_file']
    
    print("Loading time series...")
    time_series = []
    labels_array = np.loadtxt(labels_file, delimiter=',').astype(int)
    
    for i in range(len(labels_array)):
        ts_file = f"{time_series_dir}/time_series_{i:04d}.csv"
        if os.path.exists(ts_file):
            ts = np.loadtxt(ts_file, delimiter=',')
            time_series.append(ts)
    
    labels = ['child' if l == 0 else 'adult' for l in labels_array]
    
    print(f"✓ Loaded {len(time_series)} time series")
    
    # Create dataset
    dataset = BrainDataset(
        root='brain_dataset_pyg',
        time_series=time_series,
        labels=labels,
        threshold_percentile=config['data']['threshold_percentile'],
        normalize_features=config['data']['normalize_features']
    )
    
    dataset.get_statistics()
    return dataset


def train_architecture(arch_name, arch_config, dataset, config, device):
    """Train a single architecture with k-fold CV"""
    print(f"\n{'#'*80}")
    print(f"# Testing: {arch_name}")
    print(f"# {arch_config['description']}")
    print(f"{'#'*80}\n")
    
    k_folds = config['training']['k_folds']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    labels_array = dataset.get_labels().numpy()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['training']['random_seed'])
    
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_array)), labels_array)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{k_folds}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        
        # Create loaders
        train_loader = create_balanced_loader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Learning rate search
        best_val_f1 = 0
        best_model_state = None
        best_lr = None
        
        for lr in config['training']['learning_rates']:
            print(f"\nTrying LR: {lr}-------")
            
            # Initialize model
            model = arch_config['class'](
                num_node_features=dataset[0].x.shape[1],
                num_classes=2,
                **arch_config['params']
            )
            
            # Train
            trained_model, _, val_f1s = train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=lr, device=device, 
                patience=config['training']['patience']
            )
            
            # Check if best
            final_f1 = max(val_f1s) if val_f1s else 0
            if final_f1 > best_val_f1:
                best_val_f1 = final_f1
                best_model_state = trained_model.state_dict()
                best_lr = lr
        
        print(f"\nBest LR for fold {fold+1}: {best_lr} (F1: {best_val_f1:.4f})")
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        val_metrics = evaluate_model(model, val_loader, device)
        fold_results.append(val_metrics)
        
        # Save first fold model
        if fold == 0:
            best_models.append(best_model_state)
        
        print(f"\nFold {fold+1} Final Results:")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1:        {val_metrics['f1']:.4f}")
        print(f"  AUC:       {val_metrics['auc']:.4f}")
    
    # Print architecture summary
    print(f"\n{'-'*60}")
    print(f"{arch_name} Summary (5-Fold CV)")
    print(f"{'-'*60}")
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    return fold_results, best_models[0] if best_models else None


def main():
    parser = argparse.ArgumentParser(description='Train BrainGNN models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--architecture', type=str, default='all', 
                       choices=['all', 'GCN', 'GAT', 'GIN', 'GraphSAGE', 'Hybrid'],
                       help='Architecture to train')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Determine which architectures to train
    if args.architecture == 'all':
        archs_to_train = ARCHITECTURES
    else:
        archs_to_train = {args.architecture: ARCHITECTURES[args.architecture]}
    
    # Train architectures
    all_results = {}
    best_models = {}
    
    for arch_name, arch_config in archs_to_train.items():
        fold_results, best_model_state = train_architecture(
            arch_name, arch_config, dataset, config, device
        )
        all_results[arch_name] = fold_results
        best_models[arch_name] = best_model_state
    
    # Compare results
    print("\n" + "="*80)
    print("OVERALL COMPARISON - ALL ARCHITECTURES")
    print("="*80 + "\n")
    
    comparison_data = []
    for arch_name, results in all_results.items():
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        row = {'Architecture': arch_name}
        
        for metric in metrics:
            values = [r[metric] for r in results]
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
        
        comparison_data.append(row)
    
    df_results = pd.DataFrame(comparison_data)
    df_results = df_results.sort_values('f1_mean', ascending=False)
    
    print("Ranking by F1 Score:")
    print("-" * 80)
    for idx, row in df_results.iterrows():
        print(f"{row['Architecture']:15s} | "
              f"Acc: {row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
              f"F1: {row['f1_mean']:.4f}±{row['f1_std']:.4f} | "
              f"AUC: {row['auc_mean']:.4f}±{row['auc_std']:.4f}")
    
    best_arch = df_results.iloc[0]['Architecture']
    best_f1 = df_results.iloc[0]['f1_mean']
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST ARCHITECTURE: {best_arch}")
    print(f"   Average F1 Score: {best_f1:.4f}")
    print(f"   Description: {ARCHITECTURES[best_arch]['description']}")
    print(f"{'='*80}\n")
    
    # Save results
    df_results.to_csv(f'{args.output_dir}/architecture_comparison_results.csv', index=False)
    print(f"✓ Results saved to {args.output_dir}/architecture_comparison_results.csv")
    
    # Save best model
    best_model_class = ARCHITECTURES[best_arch]['class']
    best_model_params = ARCHITECTURES[best_arch]['params']
    final_model = best_model_class(
        num_node_features=dataset[0].x.shape[1],
        num_classes=2,
        **best_model_params
    )
    final_model.load_state_dict(best_models[best_arch])
    torch.save(final_model.state_dict(), 'models/best_brain_gnn_model.pt')
    print(f"✓ Best model saved to models/best_brain_gnn_model.pt")
    
    # Save detailed results
    detailed_results = {}
    for arch_name, results in all_results.items():
        detailed_results[arch_name] = {
            'description': ARCHITECTURES[arch_name]['description'],
            'params': str(ARCHITECTURES[arch_name]['params']),
            'fold_results': [
                {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                 for k, v in result.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
                for result in results
            ]
        }
    
    with open(f'{args.output_dir}/detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✓ Detailed results saved to {args.output_dir}/detailed_results.json")
    
    # Create visualizations
    if len(all_results) > 1:
        print("\nCreating visualizations...")
        plot_architecture_comparison(all_results, 'figures/architecture_comparison.png')
        plot_performance_heatmap(all_results, 'figures/performance_heatmap.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()