import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, out_path):
    cm = np.array(cm, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure()
    plt.imshow(cm_norm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix (normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    # numbers
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def classwise_markdown_table(report_dict, classes, out_path_md):
    # Build a markdown table for per-class precision/recall/f1
    lines = ["| class | precision | recall | f1-score | support |",
             "|---|---:|---:|---:|---:|"]
    for cls in classes:
        stats = report_dict.get(cls, {})
        p = stats.get('precision', 0.0)
        r = stats.get('recall', 0.0)
        f1 = stats.get('f1-score', 0.0)
        sup = stats.get('support', 0)
        lines.append(f"| {cls} | {p:.4f} | {r:.4f} | {f1:.4f} | {sup} |")
    md = "\n".join(lines)
    with open(out_path_md, "w", encoding="utf-8") as f:
        f.write(md)
    return md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="experiments/expXX_*")
    args = ap.parse_args()
    ej = os.path.join(args.exp_dir, "evaluation.json")
    assert os.path.exists(ej), f"Not found: {ej}. Run 3_eval.py first."
    with open(ej, "r", encoding="utf-8") as f:
        d = json.load(f)
    classes = d.get("classes", [])
    report = d.get("classification_report", {})
    cm = d.get("confusion_matrix", [])
    figs_dir = os.path.join(args.exp_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    # Confusion matrix heatmap
    cm_path = os.path.join(figs_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path)
    # Class-wise metrics table (markdown)
    md_path = os.path.join(args.exp_dir, "classwise_metrics.md")
    classwise_markdown_table(report, classes, md_path)
    print("Saved:", cm_path, "and", md_path)

if __name__ == "__main__":
    main()
