from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .common import IMAGENET_MEAN, IMAGENET_STD

def build_loaders(data_root, img_size=224, batch_size=32, num_workers=4):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=tf_train)
    val_ds   = datasets.ImageFolder(f"{data_root}/val",   transform=tf_eval)
    test_ds  = datasets.ImageFolder(f"{data_root}/test",  transform=tf_eval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes
