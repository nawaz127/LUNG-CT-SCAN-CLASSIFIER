import os, argparse, json
import numpy as np
from sklearn.metrics import brier_score_loss
import torch
import torch.nn as nn
import torch.optim as optim

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob.max(axis=1) > bins[i]) & (y_prob.max(axis=1) <= bins[i+1])
        if mask.any():
            acc = (y_true[mask] == y_prob[mask].argmax(1)).mean()
            conf = y_prob[mask].max(axis=1).mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)

def compute_brier(y_true, y_prob, n_classes):
    # one-hot encode y_true
    onehot = np.eye(n_classes)[y_true]
    return brier_score_loss(onehot.ravel(), y_prob.ravel())

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)
    def temperature_scale(self, logits):
        return logits / self.temperature
    def set_temperature(self, valid_loader, device):
        nll_criterion = nn.CrossEntropyLoss().to(device)
        logits_list = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        return self

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()
    npz_path = os.path.join(args.exp_dir, "probs.npz")
    assert os.path.exists(npz_path), "Run 3_eval.py first to generate probs.npz"
    data = np.load(npz_path, allow_pickle=True)
    y_true, y_prob, classes = data['y_true'], data['y_prob'], list(data['classes'])
    # ECE and Brier
    ece = expected_calibration_error(y_true, y_prob)
    brier = compute_brier(y_true, y_prob, len(classes))
    metrics = {"ECE": ece, "Brier": brier}
    with open(os.path.join(args.exp_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved calibration metrics:", metrics)
    print("To apply temperature scaling, wrap your model with ModelWithTemperature and call set_temperature() on a validation loader.")

if __name__ == "__main__":
    main()
