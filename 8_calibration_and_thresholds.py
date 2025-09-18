import argparse, os, json, numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def reliability_diagram(y_true, y_prob, classes, out_dir, n_bins=10):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        prob = y_prob[:, i]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_pred = [], []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b+1]
            mask = (prob > lo) & (prob <= hi)
            if not np.any(mask):
                continue
            frac_pos.append(y_bin[mask].mean())
            mean_pred.append(prob[mask].mean())
        frac_pos, mean_pred = np.array(frac_pos), np.array(mean_pred)
        plt.figure()
        plt.plot([0,1], [0,1], '--')
        plt.plot(mean_pred, frac_pos, marker='o')
        plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
        plt.title(f'Reliability Diagram — {cls}')
        p = os.path.join(out_dir, f'calibration_{i}_{cls}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        paths.append(p)
    return paths

def find_npz(exp_dir, split):
    for name in (f'probs_{split}.npz','probs.npz','probs_val.npz','probs_test.npz'):
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            return path
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, help='experiments/expXX_* (folder, not file)')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--split', choices=['val','test'], default='val')
    args = ap.parse_args()

    npz_path = find_npz(args.exp_dir, args.split)
    assert npz_path is not None, f"Could not find probs file in {args.exp_dir}. Run 3_eval.py first (val/test)."
    data = np.load(npz_path, allow_pickle=True)
    y_true, y_prob, classes = data['y_true'], data['y_prob'], list(data['classes'])

    figs_dir = os.path.join(args.exp_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    reliability_diagram(y_true, y_prob, classes, figs_dir, n_bins=args.bins)

    # Per-class threshold search (macro-F1)
    grid = np.arange(0.1, 0.9 + 1e-9, 0.05)
    best_f1, best_thr = -1.0, np.array([0.5]*len(classes))
    for t in np.stack(np.meshgrid(*([grid]*len(classes))), -1).reshape(-1, len(classes)):
        scaled = y_prob / np.clip(t, 1e-6, 1)
        y_pred = scaled.argmax(1)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = f1, t.copy()

    with open(os.path.join(args.exp_dir, 'optimal_thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "classes": classes,
            "thresholds": best_thr.tolist(),
            "best_macro_f1_with_thresholds": float(best_f1),
            "split": args.split
        }, f, indent=2)

    print("Saved reliability curves and thresholds in:", args.exp_dir)

if __name__ == "__main__":
    main()
