# ============================================================
# DEFINICIÓN DEL MODELO CNN PARA RECONOCIMIENTO DE DÍGITOS
# ============================================================
# Este archivo define una red neuronal convolucional (CNN)
# diseñada específicamente para el dataset MNIST y para
# imágenes manuscritas de dígitos (0–9).
#
# El modelo recibe imágenes:
#   - Escala de grises
#   - Tamaño: 28x28
#
# Y devuelve:
#   - Un vector de 10 valores (logits)
#   - Cada valor representa la "confianza" no normalizada
#     para cada dígito del 0 al 9
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Red neuronal convolucional (CNN) para clasificación
    de dígitos manuscritos (MNIST-like).

    Arquitectura general:
    - Conv → ReLU → Pool
    - Conv → ReLU → Pool
    - Fully Connected
    - Fully Connected (output)
    """

    def __init__(self):
        """
        Constructor del modelo.
        Aquí se definen TODAS las capas que el modelo utilizará.
        """
        super().__init__()

        # ---------------- CONVOLUTIONAL LAYERS ----------------
        # Primera capa convolucional
        # Entrada:
        #   - 1 canal (imagen en escala de grises)
        # Salida:
        #   - 32 mapas de características (features)
        #
        # kernel_size=3 → ventana 3x3
        # padding=1     → mantiene tamaño espacial (28x28)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Segunda capa convolucional
        # Entrada:
        #   - 32 mapas de características
        # Salida:
        #   - 64 mapas de características
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # ---------------- POOLING ----------------
        # MaxPooling reduce la resolución espacial
        # 2x2 con stride=2:
        #   28x28 → 14x14 → 7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------- FULLY CONNECTED ----------------
        # Tras dos pooling:
        #   - Canales: 64
        #   - Tamaño: 7x7
        # Total features = 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Capa de salida:
        # 128 → 10 clases (dígitos 0–9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass del modelo.
        Define cómo fluye la información desde la entrada
        hasta la salida.
        """

        # ---------------- FEATURE EXTRACTION ----------------
        # Conv1 → ReLU → MaxPool
        # Entrada: [B, 1, 28, 28]
        # Salida:  [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 → ReLU → MaxPool
        # Entrada: [B, 32, 14, 14]
        # Salida:  [B, 64, 7, 7]
        x = self.pool(F.relu(self.conv2(x)))

        # ---------------- FLATTEN ----------------
        # Convertimos el tensor 4D en 2D
        # [B, 64, 7, 7] → [B, 3136]
        x = x.view(x.size(0), -1)

        # ---------------- CLASSIFIER ----------------
        # Fully connected + activación
        x = F.relu(self.fc1(x))

        # Capa final SIN softmax
        # Devuelve logits (CrossEntropyLoss lo requiere así)
        x = self.fc2(x)

        return x


# ============================================================
# TEST RÁPIDO DEL MODELO
# ============================================================
# Este bloque solo se ejecuta si el archivo se corre directamente
# Permite verificar que:
# - El modelo acepta la entrada correcta
# - La salida tiene la forma esperada
# ============================================================

if __name__ == "__main__":
    model = DigitCNN()

    # Imagen de prueba:
    # 1 imagen, 1 canal, 28x28
    test = torch.randn(1, 1, 28, 28)

    # Debe devolver [1, 10]
    print(model(test).shape)
