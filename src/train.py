import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import DigitCNN

# Configuración
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Modelo
model = DigitCNN().to(DEVICE)

# Loss y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Entrenamiento
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), "./models/digit_cnn.pth")
print("✅ Modelo guardado en models/digit_cnn.pth")
