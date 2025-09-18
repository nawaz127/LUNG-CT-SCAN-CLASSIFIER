import torch.nn as nn
import torchvision.models as M

def build_resnet(num_classes=3, pretrained=True):
    weights = M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = M.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
