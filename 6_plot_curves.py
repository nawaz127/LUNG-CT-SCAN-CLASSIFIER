import argparse, os, json
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt

def plot_pr(y_true, y_prob, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        p, r, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap = average_precision_score(y_bin, y_prob[:, i])
        plt.figure()
        plt.step(r, p, where='post')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR — {cls} (AP={ap:.3f})')
        plt.savefig(os.path.join(out_dir, f'pr_{i}_{cls}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def plot_roc(y_true, y_prob, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        A = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0,1],[0,1],'--', lw=1)
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title(f'ROC — {cls} (AUC={A:.3f})')
        plt.savefig(os.path.join(out_dir, f'roc_{i}_{cls}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def find_npz(exp_dir, split):
    for name in (f'probs_{split}.npz','probs.npz','probs_val.npz','probs_test.npz'):
        p = os.path.join(exp_dir, name)
        if os.path.exists(p): return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, help='e.g., experiments/exp01_resnet')
    ap.add_argument('--split', choices=['val','test'], default='test')
    args = ap.parse_args()

    npz = find_npz(args.exp_dir, args.split)
    assert npz is not None, f"Could not find probs file in {args.exp_dir}. Run 3_eval.py first (val/test)."
    data = np.load(npz, allow_pickle=True)
    y_true, y_prob, classes = data['y_true'], data['y_prob'], list(data['classes'])

    figs = os.path.join(args.exp_dir, 'figs')
    plot_pr(y_true, y_prob, classes, figs)
    plot_roc(y_true, y_prob, classes, figs)

    with open(os.path.join(figs, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({"classes": classes,
                   "figs": sorted([x for x in os.listdir(figs) if x.endswith('.png')])}, f, indent=2)
    print("Saved PR/ROC curves to", figs)

if __name__ == "__main__":
    main()
