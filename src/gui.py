import tkinter as tk
import numpy as np
import os
import time
import threading
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt

from model import DigitCNN

# ================= CONFIG =================
CANVAS_SIZE = 280
IMAGE_SIZE = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEEDBACK_DIR = "data/user_feedback"
MNIST_SUBSET_SIZE = 800
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 1

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
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = img

    img = Image.fromarray(square).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    last_processed_img = img
    return img

# ================= FEEDBACK =================
def save_feedback(label):
    if last_processed_img is None:
        return

    path = os.path.join(FEEDBACK_DIR, str(label))
    os.makedirs(path, exist_ok=True)
    last_processed_img.save(
        os.path.join(path, f"{int(time.time())}.png")
    )
    status_label.config(text=f"Saved correction as {label}")

# ================= FINE-TUNING =================
def fine_tune():
    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    idx = np.random.choice(len(mnist), MNIST_SUBSET_SIZE, replace=False)
    mnist_subset = Subset(mnist, idx)

    if os.path.exists(FEEDBACK_DIR):
        feedback = datasets.ImageFolder(FEEDBACK_DIR, transform=transform)
        dataset = ConcatDataset([mnist_subset, feedback])
    else:
        dataset = mnist_subset

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    torch.save(model.state_dict(), "models/digit_cnn.pth")
    status_label.config(
        text=f"Model updated ({len(dataset)} samples)"
    )

# ================= GUI =================
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)

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

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]

    pred = probs.argmax()
    result_label.config(
        text=f"Prediction: {pred} ({probs[pred]*100:.2f}%)"
    )

    if prob_fig:
        plt.close(prob_fig)

    prob_fig = plt.figure(figsize=(6,3))
    plt.bar(range(10), probs)
    plt.xticks(range(10))
    plt.title("Probabilities")
    plt.show()

# ================= CLEAR =================
def clear():
    global prob_fig
    canvas.delete("all")
    draw.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE], fill=0)
    result_label.config(text="Draw a digit")
    if prob_fig:
        plt.close(prob_fig)
        prob_fig = None

# ================= BUTTONS =================
frame = tk.Frame(root)
frame.pack()

tk.Button(frame, text="Predict", command=predict).pack(side=tk.LEFT)
tk.Button(frame, text="Clear", command=clear).pack(side=tk.LEFT)

def run_update():
    threading.Thread(target=fine_tune).start()

tk.Button(frame, text="Update Model", command=run_update).pack(side=tk.LEFT)

result_label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
result_label.pack()

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack()

corr = tk.Frame(root)
corr.pack()
tk.Label(corr, text="Correction:").pack(side=tk.LEFT)
for i in range(10):
    tk.Button(corr, text=str(i), command=lambda x=i: save_feedback(x)).pack(side=tk.LEFT)

root.mainloop()
