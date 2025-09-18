import argparse, glob, torch
from PIL import Image
import numpy as np, cv2, os
from src.utils.gradcam_utils import get_cam, overlay_cam
from src.models.resnet import build_resnet
from src.models.vit import build_vit
try:
    from src.models.resvit import ResViTClassifier
except Exception:
    ResViTClassifier = None
import torchvision.transforms as T

def load_model(name, num_classes, ckpt):
    name=name.lower()
    if name=='resnet':
        m = build_resnet(num_classes, pretrained=False)
        target_layers = [m.layer4[-1]]
        is_vit = False
    elif name=='vit':
        m = build_vit(num_classes, pretrained=False)
        target_layers = [m.encoder.layers[-1].ln_1] if hasattr(m, 'encoder') else [m.transformer.encoder.layers[-1]]
        is_vit = True
    else:
        assert ResViTClassifier is not None, "ResViT not implemented"
        m = ResViTClassifier(num_classes)
        target_layers = [m.backbone]  # TODO adjust to your architecture
        is_vit = True
    sd = torch.load(ckpt, map_location='cpu'); m.load_state_dict(sd); m.eval()
    return m, target_layers, is_vit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', required=True, choices=['resnet','vit','resvit'])
    ap.add_argument('--images_glob', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images_glob))[:12]
    assert imgs, "No images found"
    tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    classes = sorted({os.path.basename(os.path.dirname(p)) for p in imgs})
    model, target_layers, is_vit = load_model(args.model, num_classes=len(classes), ckpt=args.ckpt)
    cam = get_cam(model, target_layers, is_vit=is_vit)

    tiles = []
    for p in imgs:
        pil = Image.open(p).convert("RGB").resize((224,224))
        x = tf(pil).unsqueeze(0)
        grayscale_cam = cam(input_tensor=x)[0]
        overlay = overlay_cam(np.array(pil), grayscale_cam)
        tiles.append(overlay)

    rows = []
    cols = 4
    for i in range(0, len(tiles), cols):
        row = np.hstack(tiles[i:i+cols])
        rows.append(row)
    panel = np.vstack(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print("Saved", args.out)

if __name__ == "__main__":
    main()
