import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR    = "path/to/your/dataset"
train_dir   = os.path.join(DATA_DIR, "train")
class_names = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

model = efficientnet_b0(weights=None, num_classes=num_classes)

ckpt = "path/to/your/checkpoint.pth"
state = torch.load(ckpt, map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device).eval()
print("Loaded checkpoint:", ckpt)

def predict_top1(img_path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    xb  = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xb)
        prob, idx = F.softmax(logits, dim=1).topk(1, dim=1)
    return class_names[idx.item()], prob.item()

def batch_predict(folder_path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    results = {}
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(exts):
            continue
        full = os.path.join(folder_path, fname)
        label, conf = predict_top1(full)
        results[fname] = (label, conf)
    return results

new_folder = "/kaggle/input/my-plants"
results = batch_predict(new_folder)

for img, (lbl, cf) in results.items():
    print(f"{img:20} → {lbl:25} ({cf*100:5.1f}%)")
