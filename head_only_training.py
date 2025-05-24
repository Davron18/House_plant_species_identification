import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# Loading pretrained EfficientNet-B0 and replacing heads
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)


in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

#  Freezing all backbone parameters
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

#  Loss & optimizer for head only
criterion = nn.CrossEntropyLoss()
opt_head  = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, momentum=0.9
)

# 5) Head training loop
def train_head(model, loader, optimizer, epochs=20):
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = total_correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            preds = out.argmax(dim=1)
            total_loss += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
        avg_loss = total_loss / len(loader.dataset)
        avg_acc  = total_correct / len(loader.dataset)
        print(f"Head Epoch {epoch}/{epochs} — loss {avg_loss:.4f}, acc {avg_acc:.4f}")
    return model

print("==> Training head only")
model = train_head(model, train_loader, opt_head, epochs=20)
