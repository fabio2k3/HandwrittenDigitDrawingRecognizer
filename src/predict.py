import torch
from torchvision import transforms
from PIL import Image

from model import DigitCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo
model = DigitCNN().to(DEVICE)
model.load_state_dict(torch.load("./models/digit_cnn.pth", map_location=DEVICE))
model.eval()

# Transformaciones (IGUAL que entrenamiento)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_digit(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


if __name__ == "__main__":
    img_path = "test_digit.png"  # imagen 28x28 o similar
    print("Predicted digit:", predict_digit(img_path))
