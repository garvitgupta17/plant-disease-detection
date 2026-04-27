import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# CONFIG
train_dir = "data/train"
test_dir  = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

# DATA
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8)

class_names = train_dataset.classes
print("Classes:", class_names)

# SAVE CLASS NAMES
import json
os.makedirs("models", exist_ok=True)
with open("models/classes.json", "w") as f:
    json.dump(class_names, f)

# MODEL
model = models.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

# TRAIN
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm

print("Starting training...")

epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"\nEpoch {epoch+1} Completed | Loss: {total_loss/len(train_loader):.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "models/prototype_model.pth")
print("✅ Model saved!")