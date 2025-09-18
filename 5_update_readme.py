import os, json, glob, pandas as pd, argparse, re

def load_rows(exp_root="experiments"):
    rows = []
    for ej in glob.glob(os.path.join(exp_root, "*/evaluation.json")):
        with open(ej, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = os.path.basename(os.path.dirname(ej))
        report = d.get("classification_report", {})
        rows.append({
            "experiment": exp,
            "model": d.get("model"),
            "accuracy": report.get("accuracy", 0.0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
            "auc_ovr": d.get("auc_ovr", float("nan"))
        })
    return pd.DataFrame(rows)

def to_markdown_table(df):
    # Manual Markdown builder (no 'tabulate' needed)
    if df.empty:
        return "| experiment | model | split | accuracy | macro_f1 | auc_ovr |\n|---|---|---|---:|---:|---:|\n| _no results yet_ |  |  |  |  |  |"
    dfx = df.copy()
    # Ensure numeric formatting
    for c in ["accuracy", "macro_f1", "auc_ovr"]:
        if c in dfx.columns:
            dfx[c] = dfx[c].astype(float)
    lines = [
        "| experiment | model | split | accuracy | macro_f1 | auc_ovr |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in dfx.iterrows():
        acc = f"{r.get('accuracy', 0.0):.4f}" if 'accuracy' in dfx.columns else ""
        f1  = f"{r.get('macro_f1', 0.0):.4f}" if 'macro_f1' in dfx.columns else ""
        auc = f"{r.get('auc_ovr', 0.0):.4f}" if 'auc_ovr' in dfx.columns else ""
        lines.append(f"| {r.get('experiment','')} | {r.get('model','')} | {r.get('split','')} | {acc} | {f1} | {auc} |")
    return "\n".join(lines)





def classwise_markdown(exp_dir):
    path = os.path.join(exp_dir, "classwise_metrics.json")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return ""
    import pandas as pd
    df = pd.DataFrame(data).T
    # Keep precision, recall, f1-score, support
    cols = [c for c in ["precision","recall","f1-score","support"] if c in df.columns]
    df = df[cols]
    md = df.to_markdown()
    return "\n**Class-wise metrics:**\n\n" + md + "\n"

def confmat_markdown(exp_dir):
    p = os.path.join(exp_dir, "figs", "confusion_matrix.png")
    if not os.path.exists(p):
        return ""
    rel = p.replace("\\","/")
    return f"\n**Confusion Matrix:**\n\n![]({rel})\n"


##__CONFUSION_AND_CLASSWISE__
def classwise_table_markdown(exp_dir):
    md_path = os.path.join(exp_dir, "classwise_metrics.md")
    if not os.path.exists(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

def confusion_md(exp_dir):
    cm_path = os.path.join(exp_dir, "figs", "confusion_matrix.png")
    if not os.path.exists(cm_path):
        return ""
    rel = os.path.join(exp_dir, "figs", "confusion_matrix.png").replace("\\", "/")
    return f"\n**Confusion Matrix**\n\n![]({rel})\n"



##__CALIBRATION_AND_THRESHOLDS__
def calibration_md(exp_dir):
    figs_dir = os.path.join(exp_dir, "figs")
    if not os.path.isdir(figs_dir):
        return ""
    items = sorted([f for f in os.listdir(figs_dir) if f.startswith("calibration_") and f.endswith(".png")])
    if not items:
        return ""
    lines = ["\n**Calibration (Reliability) Curves:**"]
    for f in items:
        rel = os.path.join(exp_dir, "figs", f).replace("\\", "/")
        lines.append(f"\n- {f}\n\n  ![]({rel})")
    return "\n".join(lines)


##__ECE_AND_BRIER__
def calibration_metrics_md(exp_dir):
    path = os.path.join(exp_dir, "calibration_metrics.json")
    if not os.path.exists(path):
        return ""
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    ece = d.get("ECE", None)
    brier = d.get("Brier", None)
    lines = ["**Calibration Metrics:**"]
    if ece is not None: lines.append(f"- Expected Calibration Error (ECE): {ece:.4f}")
    if brier is not None: lines.append(f"- Brier score: {brier:.4f}")
    return "\n".join(lines) + "\n"



##__ECE_BRIER_TEMP__
def calibration_report_md(exp_dir):
    # Try reading validation-based calibration report first, fallback to test
    import json
    for split in ["val","test"]:
        p = os.path.join(exp_dir, f"calibration_report_{split}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            lines = [
                "**Calibration Metrics**",
                "",
                f"- Split: `{d.get('split')}`",
                f"- Temperature (T): **{d.get('temperature')}**",
                f"- ECE: {d.get('ece_before'):.4f} → **{d.get('ece_after'):.4f}**",
                f"- Brier: {d.get('brier_before'):.4f} → **{d.get('brier_after'):.4f}**",
            ]
            return "\n".join(lines) + "\n"
    return ""


def thresholds_md(exp_dir):
    path = os.path.join(exp_dir, "thresholds.json")
    if not os.path.exists(path):
        return ""
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    lines = ["\n**Recommended thresholds (maximize F1 per class):**", ""]
    lines.append(f"- Default macro-F1 = {d.get('macro_f1_default',0):.4f}")
    for cls, v in d.get("per_class", {}).items():
        lines.append(f"- {cls}: t={v['best_threshold']:.2f}, F1={v['f1_at_best']:.4f}")
    return "\n".join(lines)


def figs_markdown(exp_dir):
    figs_dir = os.path.join(exp_dir, "figs")
    if not os.path.isdir(figs_dir):
        return ""
    items = sorted([f for f in os.listdir(figs_dir) if f.endswith(".png")])
    if not items:
        return ""
    lines = ["\n**Figures:**"]
    for f in items:
        rel = os.path.join(exp_dir, "figs", f).replace("\\", "/")
        lines.append(f"\n- {f}\n\n  ![]({rel})")
    return "\n".join(lines)


def inject_readme(readme_path="README.md", table_md=""):
    with open(readme_path, "r", encoding="utf-8") as f:
        txt = f.read()
    start_marker = "<!-- MODEL_COMPARISON_START -->"
    end_marker   = "<!-- MODEL_COMPARISON_END -->"

    # Append figures for each experiment below the table
    exp_root = "experiments"
    fig_blocks = []
    for exp in sorted(os.listdir(exp_root)):
        exp_dir = os.path.join(exp_root, exp)
        if not os.path.isdir(exp_dir):
            continue
        md = figs_markdown(exp_dir)
        if md:
            cw = classwise_table_markdown(exp_dir)
            cm = confusion_md(exp_dir)
            section = f"\n### {exp}\n" + md + cm
            if cw:
                section += "\n**Class-wise Metrics**\n\n" + cw + "\n"
            cal = calibration_md(exp_dir)
            th = thresholds_md(exp_dir)
            section = section + cal + "\n" + th
            fig_blocks.append(section)
    figs_section = "\n".join(fig_blocks)
    block = f"{start_marker}\n\n## Model Comparison (test set)\n\n{table_md}\n\n{figs_section}\n\n{end_marker}"

    if start_marker in txt and end_marker in txt:
        new = re.sub(re.escape(start_marker)+r".*?"+re.escape(end_marker), block, txt, flags=re.S)
    else:
        new = txt.strip() + "\n\n" + block + "\n"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new)
    print("README updated with model comparison table.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="experiments")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()
    df = load_rows(args.exp_root)
    table = to_markdown_table(df)
    inject_readme(args.readme, table)

if __name__ == "__main__":
    main()
