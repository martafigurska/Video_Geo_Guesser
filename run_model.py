import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# TRYB DZIAŁANIA
input_type = 'video'  # 'image' lub 'video'
test_image_path = 'test.png'
test_video_path = 'output_segment.mp4'

# KONFIG
NUM_FRAMES = 5
IMAGE_SIZE = (64, 64)

# MODEL
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

country_to_target = {
  "netherlands": 0, "jordan": 1, "hungary": 2, "india": 3, "russia": 4,
  "france": 5, "switzerland": 6, "kenya": 7, "canada": 8, "norway": 9,
  "usa": 10, "brazil": 11, "japan": 12, "grece": 13, "uganda": 14,
}
target_to_country = {v: k for k, v in country_to_target.items()}
num_classes = len(country_to_target)
weights_path = 'model_weights_2025-06-14_23-00-26.pth'

# MODEL
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

# TRANSFORMACJA OBRAZU
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# PREDYKCJA DLA POJEDYNCZEGO OBRAZU
def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
    return probs

# TRYB: OBRAZ
if input_type == 'image':
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    probs = predict_image(image)
    predicted_class = torch.argmax(probs).item()
    print(f"[IMAGE] Predykcja: {target_to_country[predicted_class]} ({probs[0, predicted_class]*100:.2f}%)")

    print("\nPrawdopodobieństwa dla wszystkich krajów:")
    for idx, prob in enumerate(probs[0]):
        print(f"{target_to_country[idx]}: {prob*100:.2f}%")

# TRYB: WIDEO
elif input_type == 'video':
    if not os.path.exists(test_video_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {test_video_path}")
    
    cap = cv2.VideoCapture(test_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames_idx = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    all_probs = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in selected_frames_idx:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            probs = predict_image(pil_img)
            all_probs.append(probs)
    cap.release()

    if not all_probs:
        raise RuntimeError("Nie udało się odczytać żadnej klatki z wideo.")
    avg_probs = torch.stack(all_probs)
    avg_probs = avg_probs.mean(dim=0)
    predicted_class = torch.argmax(avg_probs).item()
    print(f"[VIDEO] Predykcja (średnia z {len(all_probs)} klatek): {target_to_country[predicted_class]} ({avg_probs[0, predicted_class]*100:.2f}%)")

    print("\nPrawdopodobieństwa dla wszystkich krajów:")
    for idx, prob in enumerate(avg_probs[0]):
        print(f"{target_to_country[idx]}: {prob*100:.2f}%")

else:
    raise ValueError("Nieznany tryb wejściowy. Wybierz 'image' lub 'video'.")