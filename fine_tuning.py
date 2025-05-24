import torch.optim.lr_scheduler as lr_scheduler

# Unfreeze last two MBConv stages (features indices 6 & 7)
for idx, block in enumerate(model.features):
    requires = idx >= 6
    for p in block.parameters():
        p.requires_grad = requires

# Optimizer & scheduler for fine-tuning
opt_ft = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, momentum=0.9, weight_decay=1e-4
)
sched = lr_scheduler.CosineAnnealingLR(opt_ft, T_max=10)

# Fine-tuning loop
print("==> Fine-tuning last blocks")
for epoch in range(1, 11):
    # train
    model.train()
    t_loss = t_correct = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_ft.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt_ft.step()
        preds = out.argmax(dim=1)
        t_loss    += loss.item() * xb.size(0)
        t_correct += (preds == yb).sum().item()
    sched.step()
    t_loss /= len(train_loader.dataset); t_acc = t_correct / len(train_loader.dataset)

    # validate
    model.eval()
    v_loss = v_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            preds = out.argmax(dim=1)
            v_loss    += loss.item() * xb.size(0)
            v_correct += (preds == yb).sum().item()
    v_loss /= len(val_loader.dataset); v_acc = v_correct / len(val_loader.dataset)

    print(f"FT Epoch {epoch}/10 — "
          f"Train loss {t_loss:.4f}, acc {t_acc:.4f} — "
          f"Val   loss {v_loss:.4f}, acc {v_acc:.4f}")



