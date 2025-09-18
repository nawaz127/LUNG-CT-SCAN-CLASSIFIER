import torch
import torch.nn as nn
import torchvision.models as M

class ResViTBackbone(nn.Module):
    """
    Simple hybrid backbone:
      • ResNet50 trunk  -> global avg pool -> 2048-d
      • ViT-B/16 trunk  -> CLS feature     -> 768-d
      • Concatenate -> 2816-d feature
    Notes:
      - Exposes .res4 (last ResNet conv block) and .vit_last (last ViT encoder block)
        so you can target them for Grad-CAM.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # --- ResNet50 trunk (remove final FC) ---
        w = M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        rn = M.resnet50(weights=w)
        self.res_stem = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4
        )
        self.res_pool = nn.AdaptiveAvgPool2d(1)
        self.res4 = rn.layer4[-1]  # last conv block (great Grad-CAM target)

        # --- ViT-B/16 trunk (remove classifier head to get features) ---
        vw = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vt = M.vit_b_16(weights=vw)
        vt.heads.head = nn.Identity()       # so vt(x) returns 768-d features
        self.vit = vt
        # last encoder block (works as a Grad-CAM target for ViT-style CAM)
        self.vit_last = (
            self.vit.encoder.layers[-1].ln_1
            if hasattr(self.vit.encoder.layers[-1], "ln_1")
            else self.vit.encoder.layers[-1]
        )

        self.out_dim = 2048 + 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet path
        fr = self.res_stem(x)              # [B, 2048, H, W]
        fr = self.res_pool(fr).flatten(1)  # [B, 2048]
        # ViT path (returns 768-d CLS)
        fv = self.vit(x)                   # [B, 768]
        return torch.cat([fr, fv], dim=1)  # [B, 2816]


class ResViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = ResViTBackbone(pretrained=pretrained)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)
