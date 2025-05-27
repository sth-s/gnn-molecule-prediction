import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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