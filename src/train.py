import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def train_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    
    Parameters:
    -- model: the neural network model
    -- loader: DataLoader containing training batches
    -- optimizer: the optimizer for parameter updates
    -- device: the device to use (CPU or GPU)
    
    Returns:
    -- average loss for the epoch
    """
    model.train()
    total_loss = 0.
    total_masks = 0
    
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
        
        # Accumulate statistics
        total_loss += loss.item() * mask.sum().item()
        total_masks += mask.sum().item()
    
    # Return average loss
    return total_loss / total_masks

def evaluate(model, loader, device):
    """
    Evaluate the model and compute metrics.
    
    Parameters:
    -- model: the neural network model
    -- loader: DataLoader containing evaluation batches
    -- device: the device to use (CPU or GPU)
    
    Returns:
    -- dictionary with metrics (ROC AUC and PR AUC)
    """
    model.eval()
    all_true, all_pred = {}, {}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass with sigmoid activation
            out = model(batch).sigmoid().detach().cpu()
            
            # Get mask and labels
            mask = batch.y_mask.cpu()
            y = batch.y.cpu()
            
            # Collect predictions and true labels for each task
            for task in range(out.size(1)):
                idx = mask[:, task]
                if idx.sum() == 0:
                    continue
                    
                true = y[idx, task].numpy()
                pred = out[idx, task].numpy()
                
                all_true.setdefault(task, []).append(true)
                all_pred.setdefault(task, []).append(pred)
    
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
        'roc_auc': np.mean(roc_list),
        'pr_auc': np.mean(pr_list),
        'task_metrics': task_metrics
    }