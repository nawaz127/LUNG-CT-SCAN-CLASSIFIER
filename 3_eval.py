import argparse, os, json
import numpy as np
import torch

from src.utils.common import device
from src.utils.dataset import build_loaders
from src.utils.metrics import evaluate_model, softmax_np
from src.models.resnet import build_resnet
from src.models.vit import build_vit
try:
    from src.models.resvit import ResViTClassifier
except Exception:
    ResViTClassifier = None

# Optional: pandas is used to maintain experiments/summary.csv
try:
    import pandas as _pd
except Exception:
    _pd = None


def get_model(name, num_classes):
    name = name.lower()
    if name == "resnet":
        return build_resnet(num_classes, pretrained=False)
    elif name == "vit":
        return build_vit(num_classes, pretrained=False)
    elif name == "resvit":
        assert ResViTClassifier is not None, "ResViT not available — implement src/models/resvit.py"
        return ResViTClassifier(num_classes)
    else:
        raise ValueError("Unknown model: " + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', required=True, choices=['resnet', 'vit', 'resvit'])
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    args = ap.parse_args()

    # Derive output paths
    ckpt_dir = os.path.dirname(args.ckpt)
    out_json = os.path.join(ckpt_dir, "evaluation.json")
    out_csv  = os.path.join(os.path.dirname(ckpt_dir), "summary.csv")

    # Data
    _, val_loader, test_loader, classes = build_loaders(args.data_root)
    loader = val_loader if args.split == 'val' else test_loader

    # Model
    model = get_model(args.model, num_classes=len(classes))
    sd = torch.load(args.ckpt, map_location=device())
    model.load_state_dict(sd)
    model.to(device())
    model.eval()

    # Metrics
    report, cm, auc = evaluate_model(model, loader, device(), num_classes=len(classes))
    out = {
        "model": args.model,
        "split": args.split,
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "auc_ovr": auc
    }
    print(json.dumps(out, indent=2))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save raw arrays for downstream plotting / calibration
    logits_list, targets_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device())
            logits_list.append(model(x).cpu())
            targets_list.append(y)
    y_logits = torch.cat(logits_list).numpy()
    y_true = torch.cat(targets_list).numpy()
    y_prob = softmax_np(y_logits)

    npz_path = os.path.join(ckpt_dir, f"probs_{args.split}.npz")
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        y_prob=y_prob,
        y_logits=y_logits,
        classes=np.array(classes, dtype=object),
    )
    print("Saved arrays ->", npz_path)

    # Update summary.csv (if pandas available)
    acc = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    row = {"experiment": os.path.basename(ckpt_dir),
           "model": args.model, "split": args.split,
           "accuracy": acc, "macro_f1": macro_f1, "auc_ovr": auc}
    if _pd is not None:
        df = _pd.DataFrame([row])
        if os.path.exists(out_csv):
            old = _pd.read_csv(out_csv)
            df = _pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["experiment", "split"], keep="last", inplace=True)
        df.to_csv(out_csv, index=False)
        print("Updated summary ->", out_csv)
    else:
        # Fallback: write a minimal CSV if pandas isn't installed
        header = "experiment,model,split,accuracy,macro_f1,auc_ovr\n"
        line = f"{row['experiment']},{row['model']},{row['split']},{acc},{macro_f1},{auc}\n"
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(line)
        print("Updated summary (no pandas) ->", out_csv)


if __name__ == "__main__":
    main()
