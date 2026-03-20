import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # resize all images
    transforms.ToTensor()            # convert image to tensor
])

# 2. Load dataset
dataset = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

# 3. Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# 4. Print dataset info
print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# 5. Test loading one batch
for images, labels in dataloader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break