import os, random, shutil, glob

base = "path/to/your/dataset"  # Change this to your dataset path

subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))]
if len(subs) == 1 and os.listdir(os.path.join(base, subs[0])):
    SRC_DIR = os.path.join(base, subs[0])
else:
    SRC_DIR = base

DST_ROOT = "/kaggle/working/house_plant_species_split"

if os.path.exists(DST_ROOT):
    shutil.rmtree(DST_ROOT)

splits = {"train": 0.8, "val": 0.1, "test": 0.1}
for name in splits:
    os.makedirs(os.path.join(DST_ROOT, name), exist_ok=True)

for cls in sorted(os.listdir(SRC_DIR)):
    cls_src = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(cls_src):
        continue

    imgs = [p for p in glob.glob(f"{cls_src}/*")
            if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff"))]
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * splits["train"])
    n_val   = int(n * splits["val"])
    groups = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train+n_val],
        "test":  imgs[n_train+n_val:]
    }
    for split, files in groups.items():
        outc = os.path.join(DST_ROOT, split, cls)
        os.makedirs(outc, exist_ok=True)
        for f in files:
            shutil.copy(f, outc)

print("Done splitting; structure now is:")
