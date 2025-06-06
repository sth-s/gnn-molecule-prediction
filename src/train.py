import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
from datetime import datetime

# Model checkpoint utilities
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (can be None)
        epoch: Current epoch number
        metrics: Dictionary with current metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into (can be None)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics

def save_model_only(model, save_path, metadata=None):
    """
    Save only model state dict with optional metadata.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save model
        metadata: Optional dictionary with model metadata
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        save_dict.update(metadata)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def load_model_only(model, model_path, device):
    """
    Load only model state dict.
    
    Args:
        model: PyTorch model to load state into
        model_path: Path to model file
        device: Device to load model on
        
    Returns:
        metadata: Any additional metadata stored with model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    save_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    
    # Return metadata (everything except model state)
    metadata = {k: v for k, v in save_dict.items() if k != 'model_state_dict'}
    
    print(f"Model loaded from {model_path}")
    return metadata

def train_epoch(model, loader, optimizer, device, scheduler=None):
    """
    Train the model for one epoch.
    
    Parameters:
    -- model: the neural network model
    -- loader: DataLoader containing training batches
    -- optimizer: the optimizer for parameter updates
    -- device: the device to use (CPU or GPU)
    -- scheduler: optional learning rate scheduler (default: None)
    
    Returns:
    -- dictionary with average loss and metrics for training data
    -- list of learning rates used during each step (if scheduler is provided)
    """
    model.train()
    total_loss = 0.
    total_masks = 0
    lr_history = []
    
    # For collecting predictions and true labels for each task
    all_true, all_pred = {}, {}
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)             # [batch_size, num_tasks]
        
        # Apply mask to focus only on available labels
        mask = batch.y_mask            # mask of same size
        y_true = batch.y[mask]         # get labels where mask is true
        y_pred = out[mask]             # get predictions where mask is true
        
        # Compute loss using binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # If scheduler is provided, step it after optimizer update and track LR
        if scheduler is not None and scheduler._step_count < scheduler.total_steps:
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            scheduler.step()

        # Accumulate statistics
        total_loss += loss.item() * mask.sum().item()
        total_masks += mask.sum().item()
        
        # Collect predictions and true labels for ROC AUC calculation
        # Detach and convert to CPU for metrics calculation
        with torch.no_grad():
            out_sigmoid = out.sigmoid().detach().cpu()
            mask_cpu = mask.cpu()
            y_cpu = batch.y.cpu()
            
            # Collect predictions and true labels for each task
            for task in range(out_sigmoid.size(1)):
                idx = mask_cpu[:, task]
                if idx.sum() == 0:
                    continue
                    
                true = y_cpu[idx, task].numpy()
                pred = out_sigmoid[idx, task].numpy()
                
                all_true.setdefault(task, []).append(true)
                all_pred.setdefault(task, []).append(pred)
    
    # Calculate average loss
    avg_loss = total_loss / total_masks if total_masks > 0 else float('inf')
    
    # Calculate metrics for each task and average them
    roc_list, pr_list = [], []
    task_metrics = {}
    
    for task in all_true:
        t = np.concatenate(all_true[task])
        p = np.concatenate(all_pred[task])
        
        # Skip if only one class is present (can't compute AUC)
        if len(np.unique(t)) < 2:
            continue
            
        # Calculate ROC AUC
        roc = roc_auc_score(t, p)
        
        # Calculate PR AUC
        prec, rec, _ = precision_recall_curve(t, p)
        pr = auc(rec, prec)
        
        # Store individual task metrics
        task_metrics[task] = {'roc_auc': roc, 'pr_auc': pr}
        
        roc_list.append(roc)
        pr_list.append(pr)
    
    # Create results dictionary
    results = {
        'loss': avg_loss,
        'roc_auc': np.mean(roc_list) if roc_list else float('nan'),
        'pr_auc': np.mean(pr_list) if pr_list else float('nan'),
        'task_metrics': task_metrics
    }
    
    # Return results and lr history if scheduler is provided
    if scheduler is not None:
        return results, lr_history
    else:
        return results

def evaluate(model, loader, device):
    """
    Evaluate the model and compute metrics.
    
    Parameters:
    -- model: the neural network model
    -- loader: DataLoader containing evaluation batches
    -- device: the device to use (CPU or GPU)
    
    Returns:
    -- dictionary with metrics (ROC AUC and PR AUC)
    -- average loss
    """
    model.eval()
    all_true, all_pred = {}, {}
    total_loss = 0.0
    total_masks = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass 
            out = model(batch)
            
            # Get mask and labels
            mask = batch.y_mask
            y = batch.y
            
            # Compute loss using binary cross-entropy with logits 
            y_true = y[mask]
            y_pred = out[mask]
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
            
            # Accumulate loss statistics
            total_loss += loss.item() * mask.sum().item()
            total_masks += mask.sum().item()
            
            # Convert to sigmoid for metrics calculation
            out = out.sigmoid().detach().cpu()
            mask = mask.cpu()
            y = y.cpu()
            
            # Collect predictions and true labels for each task
            for task in range(out.size(1)):
                idx = mask[:, task]
                if idx.sum() == 0:
                    continue
                    
                true = y[idx, task].numpy()
                pred = out[idx, task].numpy()
                
                all_true.setdefault(task, []).append(true)
                all_pred.setdefault(task, []).append(pred)
    
    # Calculate average loss for validation/test set
    avg_loss = total_loss / total_masks if total_masks > 0 else float('inf')
    
    # Calculate metrics for each task and average them
    roc_list, pr_list = [], []
    task_metrics = {}
    
    for task in all_true:
        t = np.concatenate(all_true[task])
        p = np.concatenate(all_pred[task])
        
        # Skip if only one class is present (can't compute AUC)
        if len(np.unique(t)) < 2:
            continue
            
        # Calculate ROC AUC
        roc = roc_auc_score(t, p)
        
        # Calculate PR AUC
        prec, rec, _ = precision_recall_curve(t, p)
        pr = auc(rec, prec)
        
        # Store individual task metrics
        task_metrics[task] = {'roc_auc': roc, 'pr_auc': pr}
        
        roc_list.append(roc)
        pr_list.append(pr)
    
    # Return average metrics and per-task metrics
    return {
        'roc_auc': np.mean(roc_list) if roc_list else float('nan'),
        'pr_auc': np.mean(pr_list) if pr_list else float('nan'),
        'task_metrics': task_metrics,
        'loss': avg_loss
    }