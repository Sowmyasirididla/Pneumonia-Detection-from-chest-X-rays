# Pneumonia-Detection-from-chest-X-rays
Deep learning model that detects pneumonia from the Chest X-rays. The dataset used is Paul Mooney's chest x-ray pneumonia detection dataset, which is having a data imbalance issue. The dataset imbalance is solved and models like CNN and pre trained models like VGG19,ResNet50,EfficientNetb3 and DenseNet121 are used. Best model is EffiecientNetb3.

## Dataset Download

The dataset used in this project is the "Chest X-Ray Images (Pneumonia)" dataset,
curated by Paul Mooney and publicly available on Kaggle. It consists of chest X-ray
images categorized into two classes:
1. Normal – No signs of pneumonia
2. Pneumonia – Presence of lung opacity indicating pneumonia

# Download the dataset from Kaggle
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

Pneumonia cases (74%) dominate the dataset compared to normal
cases (26%).

 This imbalance must be addressed during model training to avoid bias.

# Solving the Imbalance Issue

By duplicating the underrepresented samples to match the number of the majority class, creating a balanced dataset. This way, each class contributes equally during training without relying on weighted sampling.

from collections import defaultdict
import random

def rebalance_dataset(dataset):
    class_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    # Find the class with the maximum count
    max_count = max(len(indices) for indices in class_indices.values())

    # Oversample each class to match the max_count
    balanced_indices = []
    for label, indices in class_indices.items():
        if len(indices) < max_count:
            # Oversample with replacement
            oversampled = np.random.choice(indices, max_count, replace=True)
            balanced_indices.extend(oversampled)
        else:

            balanced_indices.extend(indices)

    # Shuffle the final list of indices
    random.shuffle(balanced_indices)

    return torch.utils.data.Subset(dataset, balanced_indices)



def plot_class_distribution(dataset, title="Class Distribution"):
    if hasattr(dataset, 'dataset') and isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    class_counts = Counter(targets)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts, tick_label=[str(c) for c in classes])
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plot_class_distribution(train_dataset, title="Original Class Distribution")

balanced_train_dataset = rebalance_dataset(train_dataset)

plot_class_distribution(balanced_train_dataset, title="Balanced Class Distribution")

train_dataloader = DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)


Generating Random Images

import matplotlib.pyplot as plt
import random

def show_random_images(dataset, classes=None, num_images=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]

        plt.subplot(3, 3, i + 1)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        plt.imshow(image)
        if classes:
            plt.title(f"Label: {classes[label]}")
        else:
            plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


Random Images from the dataset with imbalance issue

class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None

show_random_images(train_dataset, classes=class_names)

Random Images from the balanced dataset

show_random_images(balanced_train_dataset, classes=class_names)

Model Trained from Scratch - CNN

import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define the CNN Model (input size 224x224)
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 56 → 28
        )
        # dynamically calculate the input size to the first linear layer
        self.classifier = nn.Sequential(
            nn.Linear(self._get_classifier_input_size(), 128), # this line was changed
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def _get_classifier_input_size(self):
        # Pass a dummy input through the features to get the output size
        dummy_input = torch.randn(1, 3, 224, 224) # Assuming input size is 224x224
        output = self.features(dummy_input)
        return output.view(1, -1).shape[1] # Return the flattened size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# Initialize model, loss function, and optimizer
model = CustomCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = corrects.double() / total
    return avg_loss, avg_acc

# Training loop with progress bar
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        corrects += torch.sum(preds == labels)
        total += labels.size(0)

        progress_bar.set_postfix(loss=running_loss / total, acc=(corrects.double() / total).item())

    train_loss = running_loss / total
    train_acc = corrects.double() / total
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Final evaluation on test set
test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style for better aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Manually defined metrics based on your training output
num_epochs = 10
train_losses = [0.2908, 0.1741, 0.1459, 0.1297, 0.1256, 0.1158, 0.1104, 0.1035, 0.1028, 0.1031]
val_losses = [0.7187, 0.5760, 0.9300, 1.3191, 0.6420, 0.7960, 0.7627, 0.7120, 0.5124, 1.0231]
train_accuracies = [0.8825, 0.9352, 0.9477, 0.9513, 0.9540, 0.9559, 0.9594, 0.9597, 0.9615, 0.9624]
val_accuracies = [0.6875, 0.5625, 0.6250, 0.5625, 0.6250, 0.5625, 0.5625, 0.5625, 0.6250, 0.5000]
test_acc = 0.7340

# Define epoch indices for x-axis
epochs = range(1, num_epochs + 1)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Loss Curves on the first subplot
ax1.plot(epochs, train_losses, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Loss")
ax1.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Loss")
ax1.set_xlabel("Epochs", fontsize=14)
ax1.set_ylabel("Loss", fontsize=14)
ax1.set_title("Training vs. Validation Loss", fontsize=16, fontweight='bold')
ax1.set_xticks(list(epochs))
ax1.legend(fontsize=12)

# Plot Accuracy Curves on the second subplot
ax2.plot(epochs, train_accuracies, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Accuracy")
ax2.plot(epochs, val_accuracies, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Accuracy")
ax2.axhline(y=test_acc, color='tab:red', linestyle=':', linewidth=2, label="Test Accuracy")
ax2.set_xlabel("Epochs", fontsize=14)
ax2.set_ylabel("Accuracy", fontsize=14)
ax2.set_title("Training vs. Validation Accuracy", fontsize=16, fontweight='bold')
ax2.set_xticks(list(epochs))
ax2.set_ylim(0.5, 1.0)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()


Pre-Trained models


VGG19 model


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure device is defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- VGG19 Model Definition -----
vgg_model = models.vgg19(pretrained=True)
for param in vgg_model.features.parameters():
    param.requires_grad = False
num_features = vgg_model.classifier[0].in_features
vgg_model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)
vgg_model = vgg_model.to(device)
criterion_vgg = nn.BCEWithLogitsLoss()
optimizer_vgg = optim.Adam(vgg_model.classifier.parameters(), lr=1e-4)

# Assume that train_loader, val_loader, test_loader are defined in previous cells.

# Evaluation function for loss and accuracy
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = corrects.double() / total
    return avg_loss, avg_acc

# Training function that tracks train, validation, and test metrics
def train_model_vgg(model, criterion, optimizer, train_loader, val_loader, test_loader, device, num_epochs):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    total_time = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = corrects.double() / total
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())

        # Evaluate on validation and test sets
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        test_losses.append(test_loss)
        test_accs.append(test_acc.item())

        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"VGG19 - Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Epoch Time: {epoch_time:.2f} sec")

    avg_epoch_time = total_time / num_epochs
    print(f"VGG19 - Epochs per second: {1/avg_epoch_time:.4f}")

    return train_losses, train_accs, val_losses, val_accs, test_losses, test_accs

# Train the model and capture metrics
num_epochs = 10
train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = train_model_vgg(
    vgg_model, criterion_vgg, optimizer_vgg, train_loader, val_loader, test_loader, device, num_epochs
)

# Plotting the curves side by side using subplots
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
epochs = range(1, num_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot Loss Curves
ax1.plot(epochs, train_losses, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Loss")
ax1.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Loss")
ax1.plot(epochs, test_losses, marker='^', linestyle='-.', linewidth=2, markersize=8, color='tab:green', label="Test Loss")
ax1.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax1.set_ylabel("Loss", fontsize=14, fontweight='bold')
ax1.set_title("Training, Validation & Test Loss", fontsize=16, fontweight='bold')
ax1.set_xticks(list(epochs))
ax1.legend(fontsize=12)

# Plot Accuracy Curves
ax2.plot(epochs, train_accs, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Accuracy")
ax2.plot(epochs, val_accs, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Accuracy")
ax2.plot(epochs, test_accs, marker='^', linestyle='-.', linewidth=2, markersize=8, color='tab:green', label="Test Accuracy")
ax2.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax2.set_title("Training, Validation & Test Accuracy", fontsize=16, fontweight='bold')
ax2.set_xticks(list(epochs))
ax2.set_ylim(0.0, 1.0)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()


ResNet50 Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- ResNet50 Model Definition -----
resnet_model = models.resnet50(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = False
num_features_resnet = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_features_resnet, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)
resnet_model = resnet_model.to(device)
criterion_resnet = nn.BCEWithLogitsLoss()
optimizer_resnet = optim.Adam(resnet_model.fc.parameters(), lr=1e-4)

# Define evaluation function for computing loss and accuracy
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = corrects.double() / total
    return avg_loss, avg_acc

# Modified training function for ResNet50 that tracks metrics and calculates test loss (for printing only)
def train_model_resnet(model, criterion, optimizer, train_loader, val_loader, test_loader, device, num_epochs):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    total_time = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

        # Calculate training metrics
        train_loss = running_loss / total
        train_acc = corrects.double() / total
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        # Evaluate on test set (for printing and test accuracy plot only)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc.item())

        epoch_time = time.time() - start_time
        total_time += epoch_time

        print(f"ResNet50 - Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Epoch Time: {epoch_time:.2f} sec")

    avg_epoch_time = total_time / num_epochs
    print(f"ResNet50 - Epochs per second: {1/avg_epoch_time:.4f}")

    return train_losses, train_accs, val_losses, val_accs, test_losses, test_accs

# Train the ResNet50 model and capture metrics
num_epochs = 10
train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = train_model_resnet(
    resnet_model, criterion_resnet, optimizer_resnet, train_loader, val_loader, test_loader, device, num_epochs
)

# Plotting the curves side by side with Seaborn style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
epochs = range(1, num_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot Loss Curves (only training and validation loss)
ax1.plot(epochs, train_losses, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Loss")
ax1.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Loss")
ax1.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax1.set_ylabel("Loss", fontsize=14, fontweight='bold')
ax1.set_title("Training vs. Validation Loss", fontsize=16, fontweight='bold')
ax1.set_xticks(list(epochs))
ax1.legend(fontsize=12)

# Plot Accuracy Curves (including test accuracy)
ax2.plot(epochs, train_accs, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Accuracy")
ax2.plot(epochs, val_accs, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Accuracy")
ax2.plot(epochs, test_accs, marker='^', linestyle='-.', linewidth=2, markersize=8, color='tab:green', label="Test Accuracy")
ax2.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax2.set_title("Training vs. Validation & Test Accuracy", fontsize=16, fontweight='bold')
ax2.set_xticks(list(epochs))
ax2.set_ylim(0.0, 1.0)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()


EfficientNetB3 Model


from torchvision.models import efficientnet_b3

# ----- EfficientNetB3 Model Setup -----
efficientnet_model = efficientnet_b3(pretrained=True)
for param in efficientnet_model.features.parameters():
    param.requires_grad = False
num_features_eff = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier = nn.Sequential(
    nn.Linear(num_features_eff, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)
efficientnet_model = efficientnet_model.to(device)
criterion_eff = nn.BCEWithLogitsLoss()
optimizer_eff = optim.Adam(efficientnet_model.classifier.parameters(), lr=1e-4)

# Evaluation function: computes average loss and accuracy over a dataloader.
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = corrects.double() / total
    return avg_loss, avg_acc

# Extended training function for EfficientNetB3:
def train_model_efficientnet_extended(model, criterion, optimizer, train_loader, val_loader, test_loader, device, num_epochs):
    # Lists to store metrics per epoch
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    total_time = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        # Training loop with progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

        # Calculate training metrics
        train_loss = running_loss / total
        train_acc = corrects.double() / total
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        # Evaluate on test set (calculated for printing and accuracy plotting)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc.item())

        epoch_time = time.time() - start_time
        total_time += epoch_time

        print(f"EfficientNetB3 - Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Epoch Time: {epoch_time:.2f} sec")

    avg_epoch_time = total_time / num_epochs
    print(f"EfficientNetB3 - Epochs per second: {1/avg_epoch_time:.4f}")

    return train_losses, train_accs, val_losses, val_accs, test_losses, test_accs

# Set number of epochs and train the model
num_epochs = 10
train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = train_model_efficientnet_extended(
    efficientnet_model, criterion_eff, optimizer_eff, train_loader, val_loader, test_loader, device, num_epochs
)

# Plotting the metrics using Seaborn style for a polished look
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
epochs = range(1, num_epochs + 1)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left subplot: Plot Training and Validation Loss (do not include test loss)
ax1.plot(epochs, train_losses, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Loss")
ax1.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Loss")
ax1.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax1.set_ylabel("Loss", fontsize=14, fontweight='bold')
ax1.set_title("EfficientNetB3: Training vs. Validation Loss", fontsize=16, fontweight='bold')
ax1.set_xticks(list(epochs))
ax1.legend(fontsize=12)

# Right subplot: Plot Training, Validation, and Test Accuracy
ax2.plot(epochs, train_accs, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Accuracy")
ax2.plot(epochs, val_accs, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Accuracy")
ax2.plot(epochs, test_accs, marker='^', linestyle='-.', linewidth=2, markersize=8, color='tab:green', label="Test Accuracy")
ax2.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax2.set_title("EfficientNetB3: Training vs. Validation & Test Accuracy", fontsize=16, fontweight='bold')
ax2.set_xticks(list(epochs))
ax2.set_ylim(0.0, 1.0)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()


 DenseNet121

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ----- DenseNet121 Model Setup -----
densenet_model = models.densenet121(pretrained=True)
for param in densenet_model.features.parameters():
    param.requires_grad = False
num_features_dense = densenet_model.classifier.in_features
densenet_model.classifier = nn.Sequential(
    nn.Linear(num_features_dense, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)
densenet_model = densenet_model.to(device)
criterion_dense = nn.BCEWithLogitsLoss()
optimizer_dense = optim.Adam(densenet_model.classifier.parameters(), lr=1e-4)

# Function to evaluate model performance
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = corrects.double() / total
    return avg_loss, avg_acc

# Extended training function for DenseNet121
def train_model_densenet_extended(model, criterion, optimizer, train_loader, val_loader, test_loader, device, num_epochs):
    # Lists to store metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    total_time = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        # Training loop with progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

        # Compute training metrics
        train_loss = running_loss / total
        train_acc = corrects.double() / total
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())

        # Evaluate on validation and test sets
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc.item())

        epoch_time = time.time() - start_time
        total_time += epoch_time

        print(f"DenseNet121 - Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Epoch Time: {epoch_time:.2f} sec")

    avg_epoch_time = total_time / num_epochs
    print(f"DenseNet121 - Epochs per second: {1/avg_epoch_time:.4f}")

    return train_losses, train_accs, val_losses, val_accs, test_losses, test_accs

# Set number of epochs and train the model
num_epochs = 10
train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = train_model_densenet_extended(
    densenet_model, criterion_dense, optimizer_dense, train_loader, val_loader, test_loader, device, num_epochs
)

# ---- Plot Metrics ----
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
epochs = range(1, num_epochs + 1)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left subplot: Plot Training and Validation Loss (do not include test loss)
ax1.plot(epochs, train_losses, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Loss")
ax1.plot(epochs, val_losses, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Loss")
ax1.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax1.set_ylabel("Loss", fontsize=14, fontweight='bold')
ax1.set_title("DenseNet121: Training vs. Validation Loss", fontsize=16, fontweight='bold')
ax1.set_xticks(list(epochs))
ax1.legend(fontsize=12)

# Right subplot: Plot Training, Validation, and Test Accuracy
ax2.plot(epochs, train_accs, marker='o', linestyle='--', linewidth=2, markersize=8, color='tab:blue', label="Train Accuracy")
ax2.plot(epochs, val_accs, marker='s', linestyle='-', linewidth=2, markersize=8, color='tab:orange', label="Validation Accuracy")
ax2.plot(epochs, test_accs, marker='^', linestyle='-.', linewidth=2, markersize=8, color='tab:green', label="Test Accuracy")
ax2.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax2.set_title("DenseNet121: Training vs. Validation & Test Accuracy", fontsize=16, fontweight='bold')
ax2.set_xticks(list(epochs))
ax2.set_ylim(0.0, 1.0)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()


Evaluating all the models

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Function to evaluate a model on the test set
def evaluate_model_metrics(model, test_loader, device, model_name):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Ensure labels are shaped correctly
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()  # Flatten for compatibility

            preds = (probs > 0.5).astype(int)  # Convert probabilities to binary labels
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds)
            y_scores.extend(probs)

    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.show()

    # ---- Classification Report ----
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()

# Evaluate all models on the test set
evaluate_model_metrics(vgg_model, test_loader, device, "VGG19")
evaluate_model_metrics(resnet_model, test_loader, device, "ResNet50")
evaluate_model_metrics(efficientnet_model, test_loader, device, "EfficientNetB3")
evaluate_model_metrics(densenet_model, test_loader, device, "DenseNet121")
evaluate_model_metrics(model, test_loader, device, "Custom CNN")


Ground Truth vs Predictions


import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize ground truth vs. predictions
def visualize_predictions(model, test_loader, device, model_name, num_images=16):
    model.eval()
    images, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            images.extend(inputs.cpu())
            true_labels.extend(labels.cpu().numpy().flatten())
            pred_labels.extend(preds)

            if len(images) >= num_images:
                break

    # Convert images to numpy format for visualization
    images = torch.stack(images[:num_images]).permute(0, 2, 3, 1).numpy()

    # Normalize images if required
    images = (images - images.min()) / (images.max() - images.min())

    # Plot the images with predictions
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap="gray")  # Adjust cmap if using color images
        true_label = "Pneumonia" if true_labels[i] == 1 else "Normal"
        pred_label = "Pneumonia" if pred_labels[i] == 1 else "Normal"

        # If prediction is incorrect, highlight in red
        color = "green" if true_label == pred_label else "red"

        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Ground Truth vs. Predictions - {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Visualize predictions for all models
visualize_predictions(vgg_model, test_loader, device, "VGG19")
visualize_predictions(resnet_model, test_loader, device, "ResNet50")
visualize_predictions(efficientnet_model, test_loader, device, "EfficientNetB3")
visualize_predictions(densenet_model, test_loader, device, "DenseNet121")
visualize_predictions(model, test_loader, device, "Custom CNN")


Finding the Best Model

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Dictionary to store model scores
model_performance = {}

# Function to evaluate models and return key metrics
def evaluate_model_metrics(model, test_loader, device, model_name):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds = (probs > 0.5).astype(int)  # Convert probabilities to binary labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_scores.extend(probs)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc_score = roc_auc_score(y_true, y_scores)
    f1 = classification_report(y_true, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate

    # Store metrics in dictionary
    model_performance[model_name] = {
        "Accuracy": round(accuracy * 100, 2),
        "AUC": round(auc_score, 3),
        "F1-Score": round(f1, 3),
        "FPR": round(fpr, 3),
        "FNR": round(fnr, 3)
    }

# Evaluate each model
evaluate_model_metrics(vgg_model, test_loader, device, "VGG19")
evaluate_model_metrics(resnet_model, test_loader, device, "ResNet50")
evaluate_model_metrics(efficientnet_model, test_loader, device, "EfficientNetB3")
evaluate_model_metrics(densenet_model, test_loader, device, "DenseNet121")
evaluate_model_metrics(model, test_loader, device, "Custom CNN")

# Print model performance table
import pandas as pd
df = pd.DataFrame(model_performance).T
print(df)

import pandas as pd

# Convert model_performance dictionary to a DataFrame
df = pd.DataFrame(model_performance).T

# Define the best model based on Accuracy, AUC, and F1-Score
best_model_acc = df["Accuracy"].idxmax()  # Model with highest Accuracy
best_model_auc = df["AUC"].idxmax()        # Model with highest AUC
best_model_f1 = df["F1-Score"].idxmax()    # Model with highest F1-score

# Print best models based on different criteria
print(f" Best Model based on Accuracy:{best_model_acc} ({df.loc[best_model_acc, 'Accuracy']}%)")
print(f" Best Model based on AUC:{best_model_auc} (AUC: {df.loc[best_model_auc, 'AUC']})")
print(f"Best Model based on F1-Score:{best_model_f1} (F1-Score: {df.loc[best_model_f1, 'F1-Score']})")

# Overall Best Model - weighted ranking (you can customize weights)
df["Overall Score"] = (df["Accuracy"] * 0.4) + (df["AUC"] * 0.4) + (df["F1-Score"] * 0.2)
best_overall_model = df["Overall Score"].idxmax()

print(f"\n Best Overall Model: {best_overall_model} ")



Hyperparameter tuning


!pip install optuna


import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Function to create models dynamically
def get_model(name, trial):
    if name == "custom_cnn":
        model = CustomCNN()
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 1)
    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet":
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    return model.to(device)

# Function to optimize hyperparameters
def objective(trial, model_name):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    model = get_model(model_name, trial)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return val_loss / len(val_loader)

# **Fix: Rename models list to model_names**
model_names = ["custom_cnn", "vgg19", "resnet50", "efficientnet", "densenet121"]
best_params = {}

for model_name in model_names:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=3)
    best_params[model_name] = study.best_params

print("\nBest Hyperparameters for Each Model:")
for model, params in best_params.items():
    print(f"{model}: {params}")

