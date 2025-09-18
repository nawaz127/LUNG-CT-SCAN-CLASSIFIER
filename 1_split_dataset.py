import argparse, os, shutil, random, json
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True, help='raw/raw with class subfolders')
    ap.add_argument('--out_dir', required=True, help='raw/processed')
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    classes = [d for d in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, d))]
    assert len(classes)>=2, "Need at least 2 classes"
    print("Classes:", classes)

    split_map = {"train": {}, "val": {}, "test": {}}

    for c in classes:
        files = []
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            files += glob(os.path.join(args.in_dir, c, ext))
        files.sort()
        random.shuffle(files)
        n = len(files)
        n_test = int(n*args.test_ratio)
        n_val  = int(n*args.val_ratio)
        test_files = files[:n_test]
        val_files  = files[n_test:n_test+n_val]
        train_files= files[n_test+n_val:]

        for split, subset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            outc = os.path.join(args.out_dir, split, c)
            os.makedirs(outc, exist_ok=True)
            split_map[split].setdefault(c, [])
            for f in subset:
                dest = os.path.join(outc, os.path.basename(f))
                shutil.copy2(f, dest)
                split_map[split][c].append(dest)

    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(split_map, f, indent=2)
    print("Done. Wrote split_manifest.json")

if __name__ == "__main__":
    main()
