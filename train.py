"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🌿  CROP DISEASE DETECTOR  –  Training Script                     ║
║                                                                              ║
║  HOW TO USE THIS TEMPLATE                                                    ║
║  ─────────────────────────                                                   ║
║  1. Organise your dataset like this:                                         ║
║       dataset/                                                               ║
║         ├── Crop_Disease_Name_1/   (one folder per class)                    ║
║         ├── Crop_Disease_Name_2/                                             ║
║         └── Crop_Healthy/                                                    ║
║                                                                              ║
║  2. Edit the CONFIG block below (only change what you need).                 ║
║  3. Run:  python train.py                                                    ║
║  4. The best model is saved to  model_best.pth                               ║
║     Class names are saved to    class_names.json  (used by predict.py)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import copy
import json
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ←  Only edit this section
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Path to your dataset folder (ImageFolder structure: one sub-folder per class)
    "dataset_dir":  "dataset",

    # Image dimensions — 224 works best for ResNet pretrained weights
    "img_size":     224,

    # Training hyper-parameters
    "batch_size":   32,
    "epochs":       30,
    "lr":           1e-3,
    "weight_decay": 1e-4,   # L2 regularisation
    "patience":     6,       # Early-stop if no improvement for N epochs
    "val_split":    0.2,     # 20 % of data held out for validation

    # Windows: keep this at 0 to avoid multiprocessing crashes
    "num_workers":  0,

    # Output file names
    "save_path":    "model_best.pth",
    "history_path": "training_history.json",
}
# ══════════════════════════════════════════════════════════════════════════════


# ── Subset wrapper — different transform per split ────────────────────────────
class SubsetWithTransform(torch.utils.data.Dataset):
    """Applies a distinct transform to a subset of a parent dataset."""

    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label


# ══════════════════════════════════════════════════════════════════════════════
#  Everything below MUST be inside __main__ on Windows — otherwise each
#  DataLoader worker re-runs the whole script and crashes.
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Device ───────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"\n🚀 Device : {device}  |  AMP (mixed precision) : {use_amp}\n")

    # ── Transforms ───────────────────────────────────────────────────────────
    SIZE  = CFG["img_size"]
    _mean = [0.485, 0.456, 0.406]   # ImageNet statistics
    _std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((SIZE + 32, SIZE + 32)),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(root=CFG["dataset_dir"])
    n_classes    = len(full_dataset.classes)

    print(f"📁 Dataset : {CFG['dataset_dir']}")
    print(f"   Classes ({n_classes}):")
    for i, cls in enumerate(full_dataset.classes, 1):
        print(f"     {i:>3}. {cls}")
    print()

    # Save class names → used by predict.py and the web backend
    with open("class_names.json", "w") as f:
        json.dump(full_dataset.classes, f, indent=2)
    print(f"✅ class_names.json saved ({n_classes} classes)\n")

    # Train / val split
    val_size   = int(CFG["val_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_idx, val_idx = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_data = SubsetWithTransform(full_dataset, train_idx.indices, train_transform)
    val_data   = SubsetWithTransform(full_dataset, val_idx.indices,   val_transform)

    train_loader = DataLoader(train_data, batch_size=CFG["batch_size"],
                              shuffle=True,  pin_memory=True,
                              num_workers=CFG["num_workers"])
    val_loader   = DataLoader(val_data,   batch_size=CFG["batch_size"],
                              shuffle=False, pin_memory=True,
                              num_workers=CFG["num_workers"])

    print(f"   Train samples : {len(train_data)}")
    print(f"   Val   samples : {len(val_data)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    # ResNet-50: A deeper, more powerful architecture for complex patterns
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze early layers; fine-tune layer2, layer3, layer4, and fc
    for name, param in model.named_parameters():
        if not any(name.startswith(k) for k in ["layer2", "layer3", "layer4", "fc"]):
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, n_classes),
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model      : ResNet-50 (fine-tuned layer2 + layer3 + layer4 + fc)")
    print(f"   Trainable params : {trainable:,} / {total:,}\n")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6,
    )

    # ── Epoch helper ─────────────────────────────────────────────────────────
    def run_epoch(loader, training: bool):
        model.train() if training else model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # ── Training loop ─────────────────────────────────────────────────────────
    history        = {"train_loss": [], "val_loss": [],
                      "train_acc":  [], "val_acc":  [], "lr": []}
    best_val_acc   = 0.0
    best_weights   = None
    patience_count = 0

    print(f"{'Epoch':>6} {'LR':>8} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("─" * 70)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss,   val_acc   = run_epoch(val_loader,   training=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        flag    = " ✅" if val_acc > best_val_acc else ""
        print(f"{epoch:>6} {current_lr:>8.2e} {train_loss:>11.4f} {train_acc:>9.2f}% "
              f"{val_loss:>10.4f} {val_acc:>8.2f}%{flag} {elapsed:>5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_weights   = copy.deepcopy(model.state_dict())
            patience_count = 0
            torch.save(best_weights, CFG["save_path"])
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"\n⏹  Early stopping — no improvement for {CFG['patience']} epochs.")
                break

    print(f"\n🏆 Best Validation Accuracy : {best_val_acc:.2f}%")

    # ── Save history + plot ───────────────────────────────────────────────────
    with open(CFG["history_path"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"📄 Training history saved → {CFG['history_path']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("📊 Training curves saved  → training_curves.png")
    print(f"✅ Best model saved       → {CFG['save_path']}")