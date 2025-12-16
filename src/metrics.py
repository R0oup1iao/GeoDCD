import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve, accuracy_score, f1_score, recall_score, precision_score

def count_accuracy(B_true, B_prob, ignore_diag=True):
    """
    Compute common metrics for causal discovery.
    
    Args:
        B_true: true adjacency matrix (N, N) (binary, 0 or 1)
        B_prob: predicted matrix (N, N). Can be continuous probabilities or binary.
        ignore_diag: whether to ignore diagonal elements (self-loops).
        
    Returns:
        dict: containing AUROC, AUPRC, F1, Precision, Recall, SHD, etc.
    """
    if hasattr(B_true, 'cpu'): B_true = B_true.cpu().numpy()
    if hasattr(B_prob, 'cpu'): B_prob = B_prob.cpu().numpy()
        
    n = B_true.shape[0]
    if ignore_diag:
        mask = ~np.eye(n, dtype=bool)
        true_flat = B_true[mask].flatten()
        prob_flat = B_prob[mask].flatten()
    else:
        true_flat = B_true.flatten()
        prob_flat = B_prob.flatten()
    
    true_flat = true_flat.astype(int)
    
    # --- 1. AUROC & AUPRC ---
    if len(np.unique(true_flat)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        fpr, tpr, _ = roc_curve(true_flat, prob_flat)
        auroc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(true_flat, prob_flat)
        auprc = auc(recall_curve, precision_curve)
    
    # --- 2. Threshold-based Metrics (F1, SHD, Precision, etc.) ---
    is_binary_prediction = np.all(np.isin(prob_flat, [0, 1]))
    best_metrics = {
        'F1': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'ACC': 0.0,
        'SHD': 0,
        'Threshold': 0.5
    }

    if is_binary_prediction:
        pred_binary = prob_flat.astype(int)
        best_metrics = _compute_binary_metrics(true_flat, pred_binary)
        best_metrics['Threshold'] = 0.5
    else:
        if len(prob_flat) < 1000:
            thresholds = np.unique(prob_flat)
        else:
            thresholds = np.linspace(prob_flat.min(), prob_flat.max(), 100)
        thresholds = thresholds[1:-1] if len(thresholds) > 2 else thresholds
        if len(thresholds) == 0: thresholds = [0.5]

        max_f1 = -1
        
        for th in thresholds:
            pred_binary = (prob_flat > th).astype(int)
            f1 = f1_score(true_flat, pred_binary, zero_division=0)
            
            if f1 > max_f1:
                max_f1 = f1
                # 计算当前阈值下的所有指标
                current_metrics = _compute_binary_metrics(true_flat, pred_binary)
                current_metrics['Threshold'] = th
                best_metrics = current_metrics

    # --- 3. Return Combined Results ---
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'F1': float(best_metrics['F1']),
        'Precision': float(best_metrics['Precision']),
        'Recall': float(best_metrics['Recall']),
        'ACC': float(best_metrics['ACC']),
        'SHD': int(best_metrics['SHD']),
        'Best_Threshold': float(best_metrics['Threshold'])
    }

def _compute_binary_metrics(true_flat, pred_flat):
    """
    Helper to calculate metrics given binary vectors.
    """
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    acc = accuracy_score(true_flat, pred_flat)
    
    # --- SHD Calculation ---
    # SHD (Structural Hamming Distance) = False Positives + False Negatives
    shd = np.sum(np.abs(true_flat - pred_flat))
    
    return {
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'ACC': acc,
        'SHD': shd
    }