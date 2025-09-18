import torch.nn as nn
import torchvision.models as M

def build_vit(num_classes=3, pretrained=True):
    weights = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    m = M.vit_b_16(weights=weights)
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m
