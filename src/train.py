# ============================================================
# ENTRENAMIENTO DEL MODELO CNN PARA MNIST
# ============================================================
# Este script entrena una red neuronal convolucional (CNN)
# para reconocer dígitos manuscritos (0–9) usando el dataset MNIST.
#
# Flujo general:
# 1. Definir hiperparámetros
# 2. Preparar transformaciones y dataset
# 3. Crear DataLoader
# 4. Inicializar modelo, loss y optimizador
# 5. Entrenar el modelo por varias épocas
# 6. Guardar los pesos entrenados
# ============================================================

# ================= IMPORTS =================

# PyTorch base
import torch

# Torchvision:
# - datasets: MNIST
# - transforms: preprocesamiento de imágenes
from torchvision import datasets, transforms

# DataLoader para batching y shuffling
from torch.utils.data import DataLoader

# Componentes de redes neuronales
import torch.nn as nn

# Optimizadores
import torch.optim as optim

# Arquitectura del modelo
from model import DigitCNN


# ================= CONFIGURACIÓN =================
# Número de pasadas completas sobre el dataset
EPOCHS = 5

# Tamaño del batch:
# - Más grande = entrenamiento más estable
# - Más pequeño = más ruido pero menos memoria
BATCH_SIZE = 64

# Learning rate:
# Controla qué tan grandes son los pasos del optimizador
LR = 0.001

# Selección automática de GPU o CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= TRANSFORMACIONES =================
# Estas transformaciones DEBEN coincidir con las usadas
# en inferencia y en la GUI
transform = transforms.Compose([
    # Convierte imagen PIL a tensor [1, 28, 28]
    transforms.ToTensor(),

    # Normaliza los valores para centrar los datos
    transforms.Normalize((0.5,), (0.5,))
])


# ================= DATASET =================
# Carga del dataset MNIST de entrenamiento
train_dataset = datasets.MNIST(
    root="./data",        # Carpeta de almacenamiento
    train=True,           # Conjunto de entrenamiento
    download=True,        # Descarga automática si no existe
    transform=transform  # Transformaciones aplicadas
)


# ================= DATALOADER =================
# Crea batches y mezcla los datos en cada época
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ================= MODELO =================
# Inicialización de la CNN
model = DigitCNN().to(DEVICE)


# ================= FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR =================
# CrossEntropyLoss:
# - Combina Softmax + LogLoss
# - Ideal para clasificación multiclase
criterion = nn.CrossEntropyLoss()

# Adam:
# - Optimización adaptativa
# - Muy estable para CNNs
optimizer = optim.Adam(model.parameters(), lr=LR)


# ================= ENTRENAMIENTO =================
for epoch in range(EPOCHS):
    # Variables para estadísticas
    total_loss = 0.0
    correct = 0
    total = 0

    # Iteramos sobre todos los batches
    for images, labels in train_loader:
        # Enviamos datos al dispositivo (CPU/GPU)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ---------------- FORWARD ----------------
        # Reiniciamos gradientes acumulados
        optimizer.zero_grad()

        # Paso forward: predicción
        outputs = model(images)

        # Cálculo de la pérdida
        loss = criterion(outputs, labels)

        # ---------------- BACKWARD ----------------
        # Calcula gradientes
        loss.backward()

        # Actualiza pesos
        optimizer.step()

        # ---------------- MÉTRICAS ----------------
        # Acumulamos la pérdida
        total_loss += loss.item()

        # Obtenemos la clase predicha
        _, predicted = torch.max(outputs, 1)

        # Contamos aciertos
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Accuracy por época
    acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {acc:.2f}%"
    )


# ================= GUARDADO DEL MODELO =================
# Guardamos SOLO los pesos (state_dict)
# Es la forma recomendada por PyTorch
torch.save(model.state_dict(), "./models/digit_cnn.pth")

print("✅ Modelo guardado en models/digit_cnn.pth")
