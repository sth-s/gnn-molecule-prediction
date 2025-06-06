#!/usr/bin/env python3
"""
Standalone training script for molecular toxicity prediction using GIN.

This script provides a command-line interface for training GIN models on the Tox21 dataset
with configurable hyperparameters and data splitting methods.

Usage:
    python scripts/train.py --epochs 200 --lr 1e-3 --hidden_channels 64
    python scripts/train.py --split_method scaffold --batch_size 64 --dropout 0.3
    python scripts/train.py --help  # for all available options
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import load_tox21, split_dataset
from model import GIN
from train import train_epoch, evaluate, save_checkpoint, save_model_only


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GIN model for molecular toxicity prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/Tox21',
                        help='Root directory for dataset')
    parser.add_argument('--filename', type=str, default='tox21.csv',
                        help='CSV filename containing the dataset')
    parser.add_argument('--split_method', type=str, default='random',
                        choices=['random', 'scaffold', 'index'],
                        help='Method for splitting the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data for testing')
    parser.add_argument('--recreate_data', action='store_true',
                        help='Force recreation of processed dataset')
    
    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels in GIN layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GIN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate')
    parser.add_argument('--max_lr_factor', type=float, default=10.0,
                        help='Factor for maximum learning rate (max_lr = lr * factor)')
    parser.add_argument('--pct_start', type=float, default=0.3,
                        help='Percentage of steps for learning rate increase')
    parser.add_argument('--div_factor', type=float, default=10.0,
                        help='Initial learning rate divisor')
    parser.add_argument('--final_div_factor', type=float, default=1000.0,
                        help='Final learning rate divisor')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs (0 to disable)')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Only save the best model (not intermediate checkpoints)')
    
    # Fast search and early stopping arguments
    parser.add_argument('--fast_search', action='store_true',
                        help='Enable fast search mode (fewer epochs, early stopping)')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Log training progress every N epochs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def setup_output_dir(output_dir, exp_name, args):
    """Setup output directory and return paths."""
    if exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"gin_tox21_{timestamp}"
    
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Configuration saved to: {config_path}")
    
    return exp_dir


def log_dataset_info(dataset, train_dataset, val_dataset, test_dataset):
    """Log information about the dataset splits."""
    print("\n=== Dataset Information ===")
    print(f"Total molecules: {len(dataset)}")
    print(f"Node features: {dataset.num_node_features}")
    try:
        print(f"Number of tasks: {dataset.num_classes}")
    except AttributeError:
        # Get num_classes from the first data point if not available from dataset
        if len(dataset) > 0:
            sample_data = dataset[0]
            print(f"Number of tasks: {sample_data.y.shape[1] if len(sample_data.y.shape) > 1 else 1}")
        else:
            print("Number of tasks: Unknown (empty dataset)")
    print("\nDataset splits:")
    print(f"  Training:   {len(train_dataset):5d} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Validation: {len(val_dataset):5d} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Test:       {len(test_dataset):5d} ({len(test_dataset)/len(dataset)*100:.1f}%)")


def log_model_info(model, optimizer, scheduler, args):
    """Log information about the model and training setup."""
    print("\n=== Model Configuration ===")
    print("Architecture: GIN")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n=== Training Configuration ===")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Base learning rate: {args.lr}")
    print(f"Max learning rate: {args.lr * args.max_lr_factor}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print(f"Log every: {args.log_every} epochs")
    if args.fast_search:
        print("Fast search mode: ENABLED")


def main():
    """Main training function."""
    args = parse_args()
    
    # Apply fast search settings if enabled
    if args.fast_search:
        if args.epochs == 200:  # Only change if using default
            args.epochs = 75
        if args.early_stop_patience is None:
            args.early_stop_patience = 10
        if args.log_every == 1:  # Only change if using default
            args.log_every = 5
        print("Fast search mode enabled:")
        print(f"  - Epochs reduced to: {args.epochs}")
        print(f"  - Early stopping patience: {args.early_stop_patience}")
        print(f"  - Logging every: {args.log_every} epochs")
    else:
        # Set default early stopping patience for normal mode
        if args.early_stop_patience is None:
            args.early_stop_patience = 20
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Setup
    torch.manual_seed(args.seed)
    device = setup_device(args.device)
    exp_dir = setup_output_dir(args.output_dir, args.exp_name, args)
    
    print("=== Molecular Toxicity Prediction Training ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {args.seed}")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    print(f"Data root: {args.data_root}")
    print(f"Split method: {args.split_method}")
    
    dataset = load_tox21(
        root=args.data_root,
        filename=args.filename,
        smiles_col="smiles",
        mol_id_col="mol_id",
        cache_file="data.pt",
        recreate=args.recreate_data,
        auto_download=True,
        device=device
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset,
        split_type=args.split_method,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    log_dataset_info(dataset, train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get number of tasks from first sample
    sample_data = dataset[0]
    num_tasks = sample_data.y.shape[1] if len(sample_data.y.shape) > 1 else sample_data.y.shape[0]
    
    # Create model
    model = GIN(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        num_classes=num_tasks,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * args.max_lr_factor,
        total_steps=total_steps,
        pct_start=args.pct_start,
        anneal_strategy='cos',
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor
    )
    
    log_model_info(model, optimizer, scheduler, args)
    
    # Training loop
    print("\n=== Training ===")
    best_val_roc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    train_start_time = time.time()
    
    history = {
        'train_loss': [],
        'train_roc_auc': [],
        'train_pr_auc': [],
        'val_loss': [],
        'val_roc_auc': [],
        'val_pr_auc': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        train_results, lr_history = train_epoch(model, train_loader, optimizer, device, scheduler)
        
        # Validation
        val_results = evaluate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_results['loss'])
        history['train_roc_auc'].append(train_results['roc_auc'])
        history['train_pr_auc'].append(train_results['pr_auc'])
        history['val_loss'].append(val_results['loss'])
        history['val_roc_auc'].append(val_results['roc_auc'])
        history['val_pr_auc'].append(val_results['pr_auc'])
        
        # Save current learning rate (one per epoch)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Timing
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging (conditional based on log_every)
        should_log = (epoch % args.log_every == 0) or (epoch == args.epochs)
        if should_log:
            print(f"Epoch {epoch:3d}/{args.epochs} ({epoch_time:.1f}s) | "
                  f"Train: Loss={train_results['loss']:.4f}, ROC={train_results['roc_auc']:.4f}, PR={train_results['pr_auc']:.4f} | "
                  f"Val: Loss={val_results['loss']:.4f}, ROC={val_results['roc_auc']:.4f}, PR={val_results['pr_auc']:.4f} | "
                  f"LR={current_lr:.2e}")
        
        # Save best model and check for early stopping
        if val_results['roc_auc'] > best_val_roc:
            best_val_roc = val_results['roc_auc']
            best_epoch = epoch
            epochs_without_improvement = 0
            
            best_model_path = exp_dir / "best_model.pt"
            save_model_only(model, best_model_path)
            if should_log:
                print(f"  → New best model saved! ROC AUC: {best_val_roc:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            print(f"No improvement for {epochs_without_improvement} epochs (patience: {args.early_stop_patience})")
            print(f"Best validation ROC AUC: {best_val_roc:.4f} (epoch {best_epoch})")
            break
        
        # Save checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0 and not args.save_best_only:
            checkpoint_path = exp_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_results['roc_auc'], checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    total_time = time.time() - train_start_time
    print("\n=== Training Complete ===")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Best validation ROC AUC: {best_val_roc:.4f} (epoch {best_epoch})")
    
    # Overfitting analysis
    if len(history['train_roc_auc']) > 10:  # Enough data for analysis
        final_epochs = min(20, len(history['train_roc_auc']))  # Last 20 epochs or all available
        final_train_roc = sum(history['train_roc_auc'][-final_epochs:]) / final_epochs
        final_val_roc = sum(history['val_roc_auc'][-final_epochs:]) / final_epochs
        roc_gap = final_train_roc - final_val_roc
        
        print(f"\n=== Overfitting Analysis (last {final_epochs} epochs) ===")
        print(f"Average train ROC AUC: {final_train_roc:.4f}")
        print(f"Average val ROC AUC:   {final_val_roc:.4f}")
        print(f"Train-Val gap:         {roc_gap:.4f}")
        
        if roc_gap > 0.05:
            print("Warning: Potential overfitting detected (gap > 0.05)")
        elif roc_gap > 0.02:
            print("Mild overfitting detected (gap > 0.02)")
        else:
            print("No significant overfitting detected")
    
    # Load best model and evaluate on test set
    best_model_path = exp_dir / "best_model.pt"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, device)
    print("\n=== Final Test Results ===")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test ROC AUC: {test_results['roc_auc']:.4f}")
    print(f"Test PR AUC: {test_results['pr_auc']:.4f}")
    
    # Save training history and final results
    results = {
        'args': vars(args),
        'training_info': {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'best_val_roc_auc': best_val_roc,
            'best_epoch': best_epoch,
            'total_epochs': epoch,  # Actual epochs completed
            'planned_epochs': args.epochs,
            'early_stopped': epoch < args.epochs,
            'epochs_without_improvement': epochs_without_improvement
        },
        'history': {
            'train_loss': history['train_loss'],
            'train_roc_auc': history['train_roc_auc'],
            'train_pr_auc': history['train_pr_auc'],
            'val_loss': history['val_loss'],
            'val_roc_auc': history['val_roc_auc'],
            'val_pr_auc': history['val_pr_auc'],
            'learning_rates': history['lr']
        },
        'final_results': {
            'best_validation': {
                'roc_auc': best_val_roc,
                'epoch': best_epoch
            },
            'test': test_results
        }
    }
    
    results_path = exp_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Best model saved to: {best_model_path}")
    
    return results


if __name__ == "__main__":
    main()
