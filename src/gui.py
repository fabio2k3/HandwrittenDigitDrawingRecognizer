import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms

from model import DigitCNN

# ---------------- CONFIG ----------------
CANVAS_SIZE = 280
IMAGE_SIZE = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL ----------------
model = DigitCNN().to(DEVICE)
model.load_state_dict(torch.load("./models/digit_cnn.pth", map_location=DEVICE))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
draw = ImageDraw.Draw(image)

# ---------------- DRAW FUNCTION ----------------
def paint(event):
    x1, y1 = event.x - 8, event.y - 8
    x2, y2 = event.x + 8, event.y + 8
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill=255)

canvas.bind("<B1-Motion>", paint)

# ---------------- PREDICT ----------------
def predict():
    img = image.copy()
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1).item()

    result_label.config(text=f"Prediction: {prediction}")

# ---------------- CLEAR ----------------
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    result_label.config(text="Draw a digit")

# ---------------- BUTTONS ----------------
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict).pack(side=tk.LEFT)
tk.Button(btn_frame, text="Clear", command=clear).pack(side=tk.LEFT)

result_label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
result_label.pack()

root.mainloop()
