import torch
import torch.nn as nn
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

# =========================
# MODEL ARCHITECTURE
# =========================
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*11*11,128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.net(x)

# =========================
# LOAD STATE DICT
# =========================
model = EyeCNN()
state = torch.load("eye_state_cnn.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

print("Model loaded successfully")

classes = ["closed","open"]

# =========================
# DATA TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((90,90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# =========================
# LOAD DATASET
# =========================
dataset = datasets.ImageFolder("dataset", transform=transform)

# =========================
# PICK 5 RANDOM SAMPLES
# =========================
indices = random.sample(range(len(dataset)), 5)

plt.figure(figsize=(12,6))

for i, idx in enumerate(indices):
    img_tensor, true_label = dataset[idx]

    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0))
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs)

    img_show = img_tensor.squeeze().numpy()

    plt.subplot(1,5,i+1)
    plt.imshow(img_show, cmap="gray")
    plt.axis("off")

    plt.title(
        f"PRED: {classes[pred]}\n"
        f"Open:{probs[0][1]:.2f}\n"
        f"Closed:{probs[0][0]:.2f}"
    )

plt.tight_layout()
plt.show()
