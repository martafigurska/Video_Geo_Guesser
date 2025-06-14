import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# === DEFINICJA MODELU ===
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dropout1 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(78720, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# === PARAMETRY ===
num_classes = 16  # Ustaw zgodnie z liczbą klas w Twoim problemie
weights_path = 'model_weights.pth'  # możesz przenieść wagę lokalnie jeśli chcesz
test_image_path = 'test_image.jpg'  # lokalny plik np. po przesłaniu przez interfejs

# === WCZYTYWANIE MODELU I WAG ===
model = CNNModel(num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# === PRZETWARZANIE ZDJĘCIA ===
transform = transforms.Compose([
    transforms.ToTensor()
])

# Wczytaj obraz lokalnie
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {test_image_path}")

image = Image.open(test_image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # dodaj wymiar batcha

# === PRZEWIDYWANIE ===
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Wynik predykcji: klasa {predicted_class}")
