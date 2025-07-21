import os
import zipfile
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

# --- PATH SETUP ---
zip_file_path = r'C:\Users\Hemambika Sri\OneDrive\ドキュメント\Desktop\plantDisease\archive.zip'
extracted_folder = r'C:\Users\Hemambika Sri\OneDrive\ドキュメント\Desktop\plantDisease\plant_disease_dataset'

# Extract dataset if not done yet
if not os.path.exists(extracted_folder):
    os.makedirs(extracted_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
    print("Dataset extracted.")

# Locate training folder
dataset_path = os.path.join(extracted_folder, 
                            'New Plant Diseases Dataset(Augmented)',
                            'New Plant Diseases Dataset(Augmented)',
                            'train')
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(extracted_folder, 'train')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Train folder not found.")

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- FILTER APPLE & POTATO CLASSES ---
dataset = ImageFolder(root=dataset_path, transform=transform)

filtered_samples = []
for path, label in dataset.samples:
    if 'apple' in path.lower() or 'potato' in path.lower():
        if os.path.exists(path):  # Confirm file exists
            try:
                with Image.open(path) as img:
                    img.convert('RGB')
                filtered_samples.append((path, label))
            except Exception:
                continue

# --- UPDATE LABELS TO NEW CLASS INDICES ---
filtered_class_names = sorted(list({dataset.classes[label] for _, label in filtered_samples}))
class_to_idx = {cls: idx for idx, cls in enumerate(filtered_class_names)}
new_samples = [(path, class_to_idx[dataset.classes[label]]) for path, label in filtered_samples]

# --- CUSTOM DATASET ---
class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):  # Retry max 10 times
            path, label = self.samples[idx]
            if not os.path.exists(path):
                idx = (idx + 1) % len(self.samples)
                continue
            try:
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception:
                print(f"Skipping unreadable file: {path}")
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError("Too many unreadable files in a row.")

# --- DATALOADER ---
dataset_obj = CustomDataset(new_samples, transform=transform)
dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

# --- MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- TRAIN SETUP ---
model = SimpleCNN(num_classes=len(class_to_idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- TRAINING LOOP ---
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# --- SAVE MODEL ---
model_path = os.path.join(os.getcwd(), 'plant_disease_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
