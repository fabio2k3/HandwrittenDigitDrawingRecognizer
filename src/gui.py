# ============================================================
# GUI PARA RECONOCIMIENTO DE DÍGITOS MANUSCRITOS CON FEEDBACK
# ============================================================
# Este archivo implementa:
# - Una interfaz gráfica (Tkinter) para dibujar dígitos
# - Inferencia con un modelo CNN entrenado en MNIST
# - Visualización de probabilidades por clase
# - Sistema de feedback humano (corrección)
# - Fine-tuning incremental mezclando MNIST + feedback
# ============================================================

# ================= IMPORTS =================
# Tkinter: interfaz gráfica nativa de Python
import tkinter as tk

# Numpy: operaciones numéricas y manipulación de arrays
import numpy as np

# Utilidades del sistema de archivos y tiempo
import os
import time
import threading

# PIL: manipulación de imágenes (canvas → imagen)
from PIL import Image, ImageDraw

# PyTorch: framework de deep learning
import torch
import torch.nn.functional as F

# Torchvision: datasets y transformaciones estándar (MNIST)
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Matplotlib: visualización de probabilidades
import matplotlib.pyplot as plt

# Modelo CNN definido en model.py
from model import DigitCNN


# ================= CONFIGURACIÓN GENERAL =================
# Tamaño del canvas donde el usuario dibuja (pixeles)
CANVAS_SIZE = 280

# Tamaño final de la imagen de entrada al modelo (MNIST = 28x28)
IMAGE_SIZE = 28

# Selección automática de dispositivo
# Usa GPU si está disponible, si no CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directorio donde se almacenan las correcciones del usuario
FEEDBACK_DIR = "data/user_feedback"

# Número de muestras aleatorias de MNIST usadas en fine-tuning
# (se usa un subset pequeño para que sea rápido)
MNIST_SUBSET_SIZE = 800

# Tamaño de batch para el entrenamiento incremental
BATCH_SIZE = 8

# Learning rate del fine-tuning
LR = 1e-3

# Número de epochs por actualización (bajo para evitar bloqueos)
EPOCHS = 1


# ================= CARGA DEL MODELO =================
# Se instancia la arquitectura CNN
model = DigitCNN().to(DEVICE)

# Se cargan los pesos previamente entrenados
model.load_state_dict(
    torch.load("models/digit_cnn.pth", map_location=DEVICE)
)

# Se pone el modelo en modo evaluación por defecto
model.eval()


# ================= TRANSFORMACIONES =================
# Estas transformaciones replican el preprocesamiento de MNIST
transform = transforms.Compose([
    # Asegura 1 solo canal (escala de grises)
    transforms.Grayscale(num_output_channels=1),

    # Convierte la imagen PIL a tensor [0,1]
    transforms.ToTensor(),

    # Normalización estándar de MNIST
    transforms.Normalize((0.5,), (0.5,))
])


# ================= VARIABLES GLOBALES =================
# Última imagen procesada (post-preprocesamiento)
# Se usa para guardar feedback correctamente
last_processed_img = None

# Referencia a la figura de matplotlib (para poder cerrarla)
prob_fig = None


# ================= PREPROCESAMIENTO =================
def preprocess_pil(img):
    """
    Preprocesa la imagen dibujada para que sea compatible con MNIST.

    Pasos:
    1. Conversión a array numpy
    2. Inversión de colores si es necesario
    3. Detección del bounding box del dígito
    4. Centrado automático
    5. Redimensionado a 28x28
    """

    global last_processed_img

    # Convertimos la imagen PIL a array numpy
    img = np.array(img)

    # Si el fondo es claro y el dígito oscuro → invertimos
    if img.mean() > 127:
        img = 255 - img

    # Localizamos los pixeles no negros (el dígito)
    coords = np.column_stack(np.where(img > 0))

    # Si no hay dibujo, se cancela
    if len(coords) == 0:
        return None

    # Bounding box del dígito
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1+1, x0:x1+1]

    # Creamos una imagen cuadrada para centrar el dígito
    h, w = img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    # Centramos el dígito en el cuadrado
    square[
        (size-h)//2:(size-h)//2+h,
        (size-w)//2:(size-w)//2+w
    ] = img

    # Redimensionamos a 28x28 con interpolación de alta calidad
    img = Image.fromarray(square).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.LANCZOS
    )

    # Guardamos la imagen procesada para feedback
    last_processed_img = img

    return img


# ================= FEEDBACK DEL USUARIO =================
def save_feedback(label):
    """
    Guarda la imagen dibujada en la carpeta correspondiente
    a la etiqueta corregida por el usuario.

    Esto permite aprendizaje supervisado humano (Human-in-the-loop).
    """

    if last_processed_img is None:
        return

    # Carpeta por clase (0–9)
    path = os.path.join(FEEDBACK_DIR, str(label))
    os.makedirs(path, exist_ok=True)

    # Guardamos la imagen con timestamp
    last_processed_img.save(
        os.path.join(path, f"{int(time.time())}.png")
    )

    status_label.config(text=f"Saved correction as {label}")


# ================= FINE-TUNING =================
def fine_tune():
    """
    Actualiza el modelo combinando:
    - Un subset aleatorio de MNIST (evita catastrophic forgetting)
    - Las correcciones del usuario (feedback)

    Se ejecuta en un hilo separado para no bloquear la GUI.
    """

    # Carga del dataset MNIST
    mnist = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Selección aleatoria de un subset pequeño
    idx = np.random.choice(
        len(mnist),
        MNIST_SUBSET_SIZE,
        replace=False
    )
    mnist_subset = Subset(mnist, idx)

    # Si hay feedback, se concatena con MNIST
    if os.path.exists(FEEDBACK_DIR):
        feedback = datasets.ImageFolder(
            FEEDBACK_DIR,
            transform=transform
        )
        dataset = ConcatDataset([mnist_subset, feedback])
    else:
        dataset = mnist_subset

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Optimizador y función de pérdida
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Entrenamiento
    model.train()
    for _ in range(EPOCHS):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    # Volvemos a modo evaluación
    model.eval()

    # Guardamos pesos actualizados
    torch.save(model.state_dict(), "models/digit_cnn.pth")

    status_label.config(
        text=f"Model updated ({len(dataset)} samples)"
    )


# ================= GUI =================
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

# Canvas de dibujo
canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black"
)
canvas.pack()

# Imagen interna donde se dibuja realmente
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)


def paint(event):
    """
    Dibuja círculos blancos siguiendo el movimiento del mouse.
    """
    r = 8
    x1, y1 = event.x - r, event.y - r
    x2, y2 = event.x + r, event.y + r
    canvas.create_oval(
        x1, y1, x2, y2,
        fill="white",
        outline="white"
    )
    draw.ellipse([x1, y1, x2, y2], fill=255)


canvas.bind("<B1-Motion>", paint)


# ================= PREDICCIÓN =================
def predict():
    """
    Ejecuta inferencia sobre el dígito dibujado
    y muestra probabilidades por clase.
    """

    global prob_fig

    img = preprocess_pil(image.copy())
    if img is None:
        return

    # Preparamos el tensor [1,1,28,28]
    x = transform(img).unsqueeze(0).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]

    pred = probs.argmax()

    result_label.config(
        text=f"Prediction: {pred} ({probs[pred]*100:.2f}%)"
    )

    # Cerramos gráfica anterior
    if prob_fig:
        plt.close(prob_fig)

    # Histograma de probabilidades
    prob_fig = plt.figure(figsize=(6, 3))
    plt.bar(range(10), probs)
    plt.xticks(range(10))
    plt.title("Probabilities")
    plt.show()


# ================= CLEAR =================
def clear():
    """
    Limpia el canvas y cierra la gráfica de probabilidades.
    """

    global prob_fig

    canvas.delete("all")
    draw.rectangle(
        [0, 0, CANVAS_SIZE, CANVAS_SIZE],
        fill=0
    )
    result_label.config(text="Draw a digit")

    if prob_fig:
        plt.close(prob_fig)
        prob_fig = None


# ================= BOTONES =================
frame = tk.Frame(root)
frame.pack()

tk.Button(frame, text="Predict", command=predict).pack(side=tk.LEFT)
tk.Button(frame, text="Clear", command=clear).pack(side=tk.LEFT)


def run_update():
    """
    Ejecuta fine-tuning en un hilo separado
    para no congelar la interfaz.
    """
    threading.Thread(target=fine_tune).start()


tk.Button(
    frame,
    text="Update Model",
    command=run_update
).pack(side=tk.LEFT)


# ================= LABELS =================
result_label = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 18)
)
result_label.pack()

status_label = tk.Label(
    root,
    text="",
    font=("Arial", 10)
)
status_label.pack()


# ================= CORRECCIÓN =================
corr = tk.Frame(root)
corr.pack()

tk.Label(corr, text="Correction:").pack(side=tk.LEFT)

# Botones 0–9 para feedback humano
for i in range(10):
    tk.Button(
        corr,
        text=str(i),
        command=lambda x=i: save_feedback(x)
    ).pack(side=tk.LEFT)


# ================= LOOP PRINCIPAL =================
root.mainloop()
