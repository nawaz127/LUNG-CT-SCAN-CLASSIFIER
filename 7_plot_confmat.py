import argparse, os, json, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    args = ap.parse_args()
    eval_json = os.path.join(args.exp_dir, 'evaluation.json')
    assert os.path.exists(eval_json), f"Missing {eval_json}"
    with open(eval_json, 'r', encoding='utf-8') as f:
        d = json.load(f)
    cm = np.array(d['confusion_matrix'])
    classes = d['classes']
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    figs_dir = os.path.join(args.exp_dir, 'figs'); os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, 'confusion_matrix.png')
    plt.savefig(out, bbox_inches='tight', dpi=150); plt.close()
    print("Saved", out)
    # Also dump class-wise metrics table
    report = d['classification_report']
    per_class = {c: report[c] for c in classes if c in report}
    with open(os.path.join(args.exp_dir, 'classwise_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(per_class, f, indent=2)
    print("Saved classwise_metrics.json")

if __name__ == "__main__":
    main()
