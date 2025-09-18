import argparse, torch, os
from src.utils.common import set_seed, device, ensure_dir
from src.utils.dataset import build_loaders
from src.utils.train_loop import train
from src.models.resnet import build_resnet
from src.models.vit import build_vit
try:
    from src.models.resvit import ResViTClassifier
except Exception:
    ResViTClassifier = None

def get_model(name, num_classes):
    name = name.lower()
    if name == "resnet":
        return build_resnet(num_classes)
    elif name == "vit":
        return build_vit(num_classes)
    elif name == "resvit":
        assert ResViTClassifier is not None, "ResViT not available—implement in src/models/resvit.py"
        return ResViTClassifier(num_classes)
    else:
        raise ValueError("Unknown model: "+name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['resnet','vit','resvit'])
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    train_loader, val_loader, test_loader, classes = build_loaders(args.data_root, img_size=args.img_size, batch_size=args.batch_size)
    model = get_model(args.model, num_classes=len(classes)).to(device())
    suffix = {'resnet':'1','vit':'2','resvit':'3'}[args.model]
    out_dir = f"experiments/exp0{suffix}_{args.model}"
    ensure_dir(out_dir)

    lr = args.lr if args.lr is not None else (1e-3 if args.model=='resnet' else 3e-4)
    best_ckpt = train(model, train_loader, val_loader, device(), args.epochs, lr, out_dir)
    print("Best checkpoint:", best_ckpt)

if __name__ == "__main__":
    main()
