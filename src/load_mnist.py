import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transformaciones: imagen → tensor + normalización
transform = transforms.Compose([
    transforms.ToTensor(),            # [0,255] → [0,1]
    transforms.Normalize((0.5,), (0.5,))
])

# Descargar dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Ver un batch
images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels:", labels[:10])

# Mostrar una imagen
plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label: {labels[0].item()}")
plt.show()
