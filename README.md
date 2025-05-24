# Plant Species Identification

Tiny PyTorch project that fine-tunes **EfficientNet-B0** to classify common house-plant species.

## Quick start
```bash
git clone https://github.com/<you>/PlantSpeciesIdentification.git
cd PlantSpeciesIdentification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, torchvision, tqdm, etc.
python main.py --image path/to/leaf.jpg

