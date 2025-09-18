import torch, os
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

def train(model, train_loader, val_loader, device, epochs, lr, out_dir, save_best_metric='macro avg'):
    os.makedirs(out_dir, exist_ok=True)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=lr)
    best_f1, best_path = -1.0, os.path.join(out_dir, "best.pt")
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for x,y in tqdm(train_loader, desc=f"epoch {ep}/{epochs}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); logits = model(x); loss = crit(logits, y); loss.backward(); opt.step()
            total += loss.item()*x.size(0)
        from .metrics import evaluate_model
        report, cm, auc = evaluate_model(model, val_loader, device, num_classes=len(set(train_loader.dataset.targets)))
        macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)
        print(f"[ep {ep}] train_loss={total/len(train_loader.dataset):.4f} macro_f1={macro_f1:.4f} auc={auc:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best to {best_path}")
    return best_path
