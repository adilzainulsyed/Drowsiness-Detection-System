import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1) DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2) DATA LOADING
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),                 # ensure 1 channel
    transforms.Resize((90, 90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_path = "dataset"   # <- change if needed

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# train-test split
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class_names = full_dataset.classes
print("Classes:", class_names)

# =========================
# 3) CNN MODEL
# =========================
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)  # open / closed
        )

    def forward(self, x):
        return self.net(x)

model = EyeCNN().to(device)
print(model)

# =========================
# 4) LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 5) TRAINING LOOP
# =========================
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# =========================
# 6) EVALUATION
# =========================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# accuracy
accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
print("Test Accuracy:", accuracy)

# =========================
# 7) CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)

sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# 8) SAVE MODEL FILE
# =========================
torch.save(model.state_dict(), "eye_state_cnn.pt")
print("Model saved as eye_state_cnn.pt")
