import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification



def main():
    # =========================
    # CONFIG
    # =========================
    train_dir = "data/train"
    val_dir   = "data/val"
    test_dir  = "data/test"

    os.makedirs("models", exist_ok=True)

    # =========================
    # TRANSFORMS
    # =========================
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # =========================
    # DATASETS
    # =========================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # =========================
    # LABEL MAPPING
    # =========================
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("Class mapping:", class_to_idx)

    # save mapping
    with open("models/class_mapping.json", "w") as f:
        json.dump(class_to_idx, f)

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # =========================
    # VISUAL CHECK
    # =========================
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)

    img = images[0].permute(1, 2, 0).numpy()
    img = (img * std) + mean  # unnormalize
    plt.imshow(img)
    plt.title(idx_to_class[labels[0].item()])
    plt.show()

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # MODEL
    # =========================
    num_classes = len(train_dataset.classes)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    for param in model.vit.parameters():
        param.requires_grad = False

    model.to(device)
 

    # =========================
    # LOSS + OPTIMIZER
    # =========================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # =========================
    # EVALUATION FUNCTION
    # =========================
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(pixel_values=images)
                preds = torch.argmax(outputs.logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
    images, labels = next(iter(train_loader))
    
    start_epoch = 0

    if os.path.exists("models/checkpoint.pth"):
        checkpoint = torch.load("models/checkpoint.pth")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        print(f"Resuming from epoch {start_epoch}")

    # =========================
    # TRAINING LOOP
    # =========================
    epochs = 8
    best_acc = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, "models/checkpoint.pth")

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_vit_model.pth")
            print("✅ Best model saved!")


    # =========================
    # FINAL TEST
    # =========================
    test_acc = evaluate(model, test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()