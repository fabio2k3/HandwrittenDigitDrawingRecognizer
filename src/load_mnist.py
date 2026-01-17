# ============================================================
# INSPECCIÓN DEL DATASET MNIST
# ============================================================
# Este script tiene como objetivo:
# - Descargar el dataset MNIST
# - Aplicar transformaciones estándar
# - Cargar los datos con DataLoader
# - Inspeccionar la forma de los tensores
# - Visualizar ejemplos reales del dataset
#
# Es un script educativo y de validación,
# no de entrenamiento.
# ============================================================

# ================= IMPORTS =================
# PyTorch: framework principal de deep learning
import torch

# Torchvision:
# - datasets: datasets clásicos (MNIST)
# - transforms: preprocesamiento de imágenes
from torchvision import datasets, transforms

# Matplotlib: visualización de imágenes
import matplotlib.pyplot as plt


# ================= TRANSFORMACIONES =================
# Las transformaciones convierten la imagen MNIST
# en un formato numérico que el modelo puede procesar.
transform = transforms.Compose([
    # Convierte imagen PIL (0–255) a tensor float (0–1)
    # Resultado: tensor [1, 28, 28]
    transforms.ToTensor(),

    # Normalización:
    # (x - mean) / std
    # MNIST suele normalizarse alrededor de 0
    transforms.Normalize((0.5,), (0.5,))
])


# ================= DATASET =================
# Descarga y carga el dataset MNIST de entrenamiento
# Si ya existe en ./data, no se vuelve a descargar
train_dataset = datasets.MNIST(
    root="./data",        # Carpeta donde se guarda el dataset
    train=True,           # True = conjunto de entrenamiento
    download=True,        # Descarga automática si no existe
    transform=transform  # Transformaciones aplicadas a cada imagen
)


# ================= DATALOADER =================
# El DataLoader se encarga de:
# - Crear batches
# - Mezclar los datos
# - Optimizar la carga en memoria
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # Número de imágenes por batch
    shuffle=True    # Mezcla los datos en cada epoch
)


# ================= INSPECCIÓN DE UN BATCH =================
# Obtenemos un batch del DataLoader
# images: tensor de imágenes
# labels: etiquetas correspondientes
images, labels = next(iter(train_loader))

# Forma del tensor de imágenes
# Esperado: [batch_size, canales, alto, ancho]
print("Images shape:", images.shape)

# Mostramos las primeras 10 etiquetas del batch
print("Labels:", labels[:10])


# ================= VISUALIZACIÓN =================
# Mostramos la primera imagen del batch
# squeeze() elimina la dimensión del canal (1)
plt.imshow(images[0].squeeze(), cmap="gray")

# Título con la etiqueta real
plt.title(f"Label: {labels[0].item()}")

# Mostrar ventana
plt.show()
