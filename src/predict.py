# ============================================================
# SCRIPT DE INFERENCIA PARA RECONOCIMIENTO DE DÍGITOS
# ============================================================
# Este archivo permite:
# - Cargar un modelo CNN entrenado
# - Preprocesar una imagen externa
# - Realizar una predicción del dígito manuscrito (0–9)
#
# Se utiliza principalmente para:
# - Pruebas rápidas
# - Validación del modelo entrenado
# - Inferencia fuera de la GUI
# ============================================================

# ================= IMPORTS =================

# PyTorch: framework de deep learning
import torch

# Torchvision: transformaciones estándar para imágenes
from torchvision import transforms

# PIL: carga y manipulación de imágenes
from PIL import Image

# Importamos la arquitectura del modelo
from model import DigitCNN


# ================= CONFIGURACIÓN DEL DISPOSITIVO =================
# Se selecciona GPU si está disponible, de lo contrario CPU
# Esto permite que el mismo código funcione en cualquier máquina
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= CARGA DEL MODELO =================
# Se instancia el modelo con la misma arquitectura usada
# durante el entrenamiento
model = DigitCNN().to(DEVICE)

# Se cargan los pesos entrenados desde disco
# map_location garantiza compatibilidad CPU/GPU
model.load_state_dict(
    torch.load("./models/digit_cnn.pth", map_location=DEVICE)
)

# Se pone el modelo en modo evaluación
# IMPORTANTE:
# - Desactiva dropout (si existiera)
# - Ajusta el comportamiento de BatchNorm (si existiera)
model.eval()


# ================= TRANSFORMACIONES =================
# Estas transformaciones DEBEN ser idénticas a las del entrenamiento.
# De lo contrario, el modelo verá datos con una distribución distinta.
transform = transforms.Compose([
    # Convierte la imagen a escala de grises
    # MNIST usa un solo canal
    transforms.Grayscale(),

    # Redimensiona la imagen a 28x28 píxeles
    # Tamaño esperado por el modelo
    transforms.Resize((28, 28)),

    # Convierte la imagen PIL a tensor PyTorch
    # Rango: [0, 255] → [0, 1]
    transforms.ToTensor(),

    # Normalización:
    # (x - mean) / std
    # Coincide con la normalización usada en entrenamiento
    transforms.Normalize((0.5,), (0.5,))
])


# ================= FUNCIÓN DE PREDICCIÓN =================
def predict_digit(image_path):
    """
    Realiza la predicción de un dígito manuscrito a partir
    de una imagen almacenada en disco.

    Parámetros:
    ----------
    image_path : str
        Ruta al archivo de imagen que contiene el dígito.

    Retorna:
    -------
    int
        Dígito predicho (0–9).
    """

    # Abrimos la imagen desde disco
    image = Image.open(image_path)

    # Aplicamos las transformaciones
    # unsqueeze(0) agrega la dimensión de batch:
    # [1, 1, 28, 28]
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Desactivamos el cálculo de gradientes
    # (más rápido y menos uso de memoria)
    with torch.no_grad():
        output = model(image)

        # output son logits (no probabilidades)
        # argmax devuelve el índice con mayor valor
        prediction = torch.argmax(output, dim=1).item()

    return prediction


# ================= EJECUCIÓN DIRECTA =================
# Este bloque solo se ejecuta si el archivo se corre directamente:
#   python predict.py
if __name__ == "__main__":
    # Ruta a una imagen de prueba
    # Puede ser cualquier imagen con un dígito manuscrito
    img_path = "test_digit.png"

    # Realizamos la predicción y la mostramos por consola
    print("Predicted digit:", predict_digit(img_path))
