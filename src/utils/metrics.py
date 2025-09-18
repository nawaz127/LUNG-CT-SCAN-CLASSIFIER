import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_prob = softmax_np(logits)
    y_pred = y_prob.argmax(1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    return report, cm, auc

def softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
