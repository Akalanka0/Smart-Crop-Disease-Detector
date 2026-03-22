import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. Advanced Transformations (Data Augmentation)
# Adding random flips and rotations helps the model recognize 
# disease spots even if the leaf is tilted or upside down.
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# 2. Load the full dataset
full_dataset = datasets.ImageFolder(root="../dataset", transform=train_transform)

# 3. Split into Train (80%) and Validation (20%)
# This ensures you can test the model on images it has never seen during training.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 4. Create DataLoaders
# pin_memory=True speeds up the transfer from CPU RAM to your RTX 2050 VRAM.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# 5. Summary Info
print(f"Total images: {len(full_dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Detected Classes: {full_dataset.classes}")

# 6. Sanity Check
images, labels = next(iter(train_loader))
print(f"Batch Image Shape: {images.shape}") # Expect [32, 3, 128, 128]