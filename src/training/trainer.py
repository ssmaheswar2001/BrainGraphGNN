"""
Model training utilities
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from .losses import FocalLoss
from .evaluator import evaluate_model


def create_balanced_loader(dataset, batch_size=32):
    """
    Create DataLoader with balanced sampling for class imbalance
    
    Args:
        dataset: PyTorch Geometric dataset
        batch_size: Batch size
    
    Returns:
        DataLoader with balanced sampling
    """
    labels = dataset.get_labels().numpy()
    class_counts = np.bincount(labels)
    weights = 1. / class_counts[labels]
    
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True
    )
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001,
                weight_decay=5e-4, device='cuda', patience=20):
    """
    Train model with focal loss and early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        patience: Early stopping patience
    
    Returns:
        tuple: (trained_model, train_losses, val_f1s)
    """
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_f1s = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluate
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1s.append(val_metrics['f1'])
        
        scheduler.step(val_metrics['f1'])
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_f1s