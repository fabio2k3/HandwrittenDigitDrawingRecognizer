import tkinter as tk
import numpy as np
import os
import time
import threading
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import DigitCNN

# ================= CONFIG =================
CANVAS_SIZE = 280
IMAGE_SIZE = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEEDBACK_DIR = "data/user_feedback"
MIN_SAMPLES_TO_TRAIN = 1  # mínimo para fine-tune rápido
BATCH_SIZE = 8

# ================= MODEL =================
model = DigitCNN().to(DEVICE)
model.load_state_dict(torch.load("models/digit_cnn.pth", map_location=DEVICE))
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================= GLOBALS =================
last_processed_img = None
prob_fig = None

# ================= PREPROCESS =================
def preprocess_pil(img):
    global last_processed_img
    img = np.array(img)

    if img.mean() > 127:
        img = 255 - img

    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1+1, x0:x1+1]

    h, w = img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img
    img = Image.fromarray(square).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    last_processed_img = img
    return img

# ================= FEEDBACK =================
def save_feedback(label):
    if last_processed_img is None:
        return
    path = os.path.join(FEEDBACK_DIR, str(label))
    os.makedirs(path, exist_ok=True)
    filename = f"{int(time.time())}.png"
    last_processed_img.save(os.path.join(path, filename))
    status_label.config(text=f"Saved correction as {label}")

# ================= FINE-TUNING =================
def fine_tune():
    # Solo feedback para fine-tuning rápido
    if os.path.exists(FEEDBACK_DIR):
        dataset = datasets.ImageFolder(FEEDBACK_DIR, transform=transform)
    else:
        status_label.config(text="No feedback to train")
        return

    if len(dataset) < MIN_SAMPLES_TO_TRAIN:
        status_label.config(text="Not enough samples to retrain")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    torch.save(model.state_dict(), "models/digit_cnn.pth")
    status_label.config(text=f"Model updated with {len(dataset)} feedback samples")

# ================= GUI =================
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)

# ================= DRAW =================
def paint(event):
    r = 8
    x1, y1 = event.x - r, event.y - r
    x2, y2 = event.x + r, event.y + r
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill=255)

canvas.bind("<B1-Motion>", paint)

# ================= PREDICT =================
def predict():
    global prob_fig
    img = preprocess_pil(image.copy())
    if img is None:
        return

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        pred = np.argmax(probs)

    result_label.config(
        text=f"Prediction: {pred} ({probs[pred]*100:.2f}%)"
    )

    # Cerrar figura anterior si existe
    if prob_fig is not None:
        plt.close(prob_fig)

    prob_fig = plt.figure(figsize=(6,3))
    plt.bar(range(10), probs, color='blue')
    plt.xticks(range(10))
    plt.ylabel("Probability")
    plt.title("Class Probabilities")
    plt.show()

# ================= CLEAR =================
def clear():
    global prob_fig
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    result_label.config(text="Draw a digit")
    if prob_fig is not None:
        plt.close(prob_fig)
        prob_fig = None

# ================= BUTTONS =================
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict, width=10).pack(side=tk.LEFT)
tk.Button(btn_frame, text="Clear", command=clear, width=10).pack(side=tk.LEFT)

# Fine-tuning en hilo separado
def run_fine_tune():
    threading.Thread(target=fine_tune).start()

tk.Button(btn_frame, text="Update Model", command=run_fine_tune, width=12).pack(side=tk.LEFT)

result_label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
result_label.pack()

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack()

# ================= CORRECTION BUTTONS =================
corr_frame = tk.Frame(root)
corr_frame.pack()

tk.Label(corr_frame, text="Correction:").pack(side=tk.LEFT)
for i in range(10):
    tk.Button(
        corr_frame,
        text=str(i),
        command=lambda x=i: save_feedback(x),
        width=2
    ).pack(side=tk.LEFT)

root.mainloop()
