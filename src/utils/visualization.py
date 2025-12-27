"""
Visualization utilities for model comparison and results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_architecture_comparison(all_results, save_path='architecture_comparison.png'):
    """
    Create box plots comparing all architectures across metrics
    
    Args:
        all_results: Dictionary of architecture results
        save_path: Path to save figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data = []
        labels_list = []
        
        for arch_name, results in all_results.items():
            values = [r[metric] for r in results]
            data.append(values)
            labels_list.append(arch_name)
        
        bp = ax.boxplot(data, labels=labels_list, patch_artist=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels_list)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Box plots saved: {save_path}")


def plot_performance_heatmap(all_results, save_path='performance_heatmap.png'):
    """
    Create heatmap of architecture performance
    
    Args:
        all_results: Dictionary of architecture results
        save_path: Path to save figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    arch_names = list(all_results.keys())
    data_matrix = np.zeros((len(arch_names), len(metrics)))
    
    for i, arch_name in enumerate(arch_names):
        for j, metric in enumerate(metrics):
            values = [r[metric] for r in all_results[arch_name]]
            data_matrix[i, j] = np.mean(values)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, fmt='.3f',
               xticklabels=[m.capitalize() for m in metrics],
               yticklabels=arch_names,
               cmap='YlGnBu', vmin=0, vmax=1,
               cbar_kws={'label': 'Score'})
    
    plt.title('Architecture Performance Heatmap\n(5-Fold Cross-Validation Average)',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: {save_path}")


def plot_training_curves(train_losses, val_metrics, save_path='training_curves.png'):
    """
    Plot training loss and validation metrics over epochs
    
    Args:
        train_losses: List of training losses
        val_metrics: List of validation F1 scores
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation F1
    ax2.plot(val_metrics, label='Validation F1', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score over Epochs', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Child', 'Adult'],
               yticklabels=['Child', 'Adult'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {save_path}")