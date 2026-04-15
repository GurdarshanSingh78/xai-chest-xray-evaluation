import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '/Users/gurdarshansingh/Downloads/archive/chest_xray/train'
dataset = datasets.ImageFolder(data_dir, transform=transform)

print("Scrambling dataset labels...")
np.random.seed(42)
scrambled_labels = np.random.permutation(dataset.targets)
dataset.targets = scrambled_labels.tolist()

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training on randomized labels for 3 epochs...")
model.train()
for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "resnet50_scrambled_labels.pth")
print("✅ Saved scrambled model as resnet50_scrambled_labels.pth. You can now extract heatmaps from this model!")