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

    model.load_state_dict(torch.load("models/best_vit_model.pth"))
    model.to(device)
 


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
    
    

    # =========================
    # 🔥 PHASE 2: FINE-TUNING
    # =========================

    print("\n🚀 Starting Fine-Tuning Phase...")

    

    # Unfreeze backbone
    for param in model.vit.parameters():
        param.requires_grad = True

    # Use smaller learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5
    )


    fine_tune_epochs = 3
    best_ft_acc = 0
    start_epoch = 0

    if os.path.exists("models/fine_tune_checkpoint.pth"):
        checkpoint = torch.load("models/fine_tune_checkpoint.pth")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_ft_acc = checkpoint['best_ft_acc']

        print(f"Resuming fine-tuning from epoch {start_epoch}")
    
    else:
        best_ft_acc = 0

    for epoch in range(start_epoch, fine_tune_epochs):
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

            loop.set_description(f"FineTune Epoch [{epoch+1}/{fine_tune_epochs}]")
            loop.set_postfix(loss=loss.item())

        val_acc = evaluate(model, val_loader)

        print(f"\nFine-Tune Epoch {epoch+1}")
        print(f"Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_ft_acc': best_ft_acc
        }, "models/fine_tune_checkpoint.pth")

        # Save best fine-tuned model
        if val_acc > best_ft_acc:
            best_ft_acc = val_acc
            torch.save(model.state_dict(), "models/fine_tuned_vit.pth")
            print("✅ Best Fine-Tuned Model Saved!")

    # =========================
    # FINAL TEST AFTER FINE-TUNING
    # =========================


    model.load_state_dict(torch.load("models/fine_tuned_vit.pth"))

    test_acc = evaluate(model, test_loader)
    print(f"\n🔥 Final Test Accuracy After Fine-Tuning: {test_acc:.4f}")


if __name__ == "__main__":
    main()