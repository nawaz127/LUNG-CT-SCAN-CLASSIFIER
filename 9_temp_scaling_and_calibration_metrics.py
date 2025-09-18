import os, json, argparse, numpy as np
from sklearn.metrics import brier_score_loss, log_loss

def ece_score(y_true, y_prob, n_bins=15):
    # y_true: (N,) int labels ; y_prob: (N, C)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def temp_scale_logits(y_logits, T):
    # y_logits shape: (N, C)
    z = y_logits / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def fit_temperature(y_logits, y_true, grid=None):
    if grid is None:
        grid = np.concatenate([[0.5], np.linspace(0.75, 3.0, 23)])
    best_T, best_nll = None, float('inf')
    for T in grid:
        p = temp_scale_logits(y_logits, T)
        nll = log_loss(y_true, p, labels=list(range(p.shape[1])))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, help='experiments/expXX_*')
    ap.add_argument('--split', choices=['val','test'], default='val', help='use validation by default to fit T')
    ap.add_argument('--bins', type=int, default=15)
    args = ap.parse_args()

    npz_path = os.path.join(args.exp_dir, f'probs_{args.split}.npz')
    assert os.path.exists(npz_path), f"Not found: {npz_path}. Re-run 3_eval.py with --split {args.split}."
    data = np.load(npz_path, allow_pickle=True)
    y_true = data['y_true']; y_prob = data['y_prob']; y_logits = data['y_logits']; classes = list(data['classes'])

    # Pre-calibration metrics
    ece_before = ece_score(y_true, y_prob, n_bins=args.bins)
    # One-vs-rest Brier averaged across classes
    briers = []
    for i in range(len(classes)):
        y_bin = (y_true == i).astype(int)
        briers.append(brier_score_loss(y_bin, y_prob[:, i]))
    brier_before = float(np.mean(briers))

    # Fit temperature on selected split
    T, nll_after = fit_temperature(y_logits, y_true)
    y_prob_cal = temp_scale_logits(y_logits, T)

    # Post-calibration metrics
    ece_after = ece_score(y_true, y_prob_cal, n_bins=args.bins)
    briers_after = []
    for i in range(len(classes)):
        y_bin = (y_true == i).astype(int)
        briers_after.append(brier_score_loss(y_bin, y_prob_cal[:, i]))
    brier_after = float(np.mean(briers_after))

    # Save report & calibrated probs
    report = {
        "split": args.split,
        "classes": classes,
        "temperature": float(T),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after
    }
    with open(os.path.join(args.exp_dir, f'calibration_report_{args.split}.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(os.path.join(args.exp_dir, f'probs_{args.split}_calibrated.npz'),
                        y_true=y_true, y_prob=y_prob_cal, classes=np.array(classes, dtype=object))
    print("Saved:", os.path.join(args.exp_dir, f'calibration_report_{args.split}.json'))
    print("Saved:", os.path.join(args.exp_dir, f'probs_{args.split}_calibrated.npz'))

if __name__ == "__main__":
    main()
