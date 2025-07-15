import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Config
model_name = "google/vit-base-patch16-224"
test_dir = "./vit-data/val"  # Use a small sample validation folder
num_classes = 3
batch_size = 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Feature extractor
extractor = ViTFeatureExtractor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
])

# Dataset + Loader
try:
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
except Exception as e:
    print(f"[ERROR] Dataset loading failed: {e}")
    exit(1)

# Load model
try:
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    exit(1)

# Run one mini-batch
model.eval()
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        print(f"[SMOKE TEST] Output shape: {outputs.logits.shape}")
        break

print("[✅] ViT smoke test completed successfully.")