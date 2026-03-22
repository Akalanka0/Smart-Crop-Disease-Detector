import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

from models.cnn_model import CNNModel

# 1. Device Setup (Targeting your RTX 2050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {device}")

# 2. Advanced Transforms
# Normalization helps the math converge faster on the GPU
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(), # Augmentation for better leaf recognition
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Dataset & Split
full_dataset = datasets.ImageFolder(root="dataset", transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

# 4. DataLoaders (Optimized with pin_memory for CUDA)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=True)

# 5. Model Initialization
model = CNNModel(num_classes=len(full_dataset.classes))
model.to(device)

# 6. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training & Validation Loop
epochs = 10 # Increased epochs since GPU is faster

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Quick Validation Check
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    epoch_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - Val Acc: {accuracy:.2f}% - Time: {epoch_time:.2f}s")

# 8. Save final weights
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved successfully.")