import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error
import seaborn as sns
import pandas as pd

class L2DFrameDataset(Dataset):
    """
    A PyTorch dataset for loading L2D frames.
    """
    def __init__(self, root_dir, transform=None, max_images=None, task_type="reconstruction", annotations_dir=None):
        """
        Initializes the dataset.
        
        Args:
            root_dir (str): The root directory of the dataset.
            transform (torchvision.transforms.Compose, optional): The transform to apply to the images. Defaults to None.
            max_images (int, optional): The maximum number of images to load. Defaults to None.
            task_type (str, optional): The type of task to perform. Defaults to "reconstruction".
            annotations_dir (str, optional): The directory containing the annotations. Defaults to None.
        """
        # Step 1: Initialize the dataset
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images
        self.task_type = task_type
        self.annotations_dir = annotations_dir
        self.image_paths = []
        self.labels = []

        # Step 2: Load the image paths and labels
        for episode_dir in sorted(os.listdir(root_dir)):
            episode_path = os.path.join(root_dir, episode_dir)
            if os.path.isdir(episode_path):
                camera_path = os.path.join(episode_path, "observation.images.front_left")
                if os.path.isdir(camera_path):
                    for img_name in sorted(os.listdir(camera_path)):
                        if img_name.endswith(".jpg"):
                            self.image_paths.append(os.path.join(camera_path, img_name))
                            
                            if self.task_type in ["vehicle_presence", "vehicle_count"]:
                                annotation_episode_path = os.path.join(annotations_dir, episode_dir)
                                annotation_file_name = img_name.replace(".jpg", ".json")
                                annotation_file_path = os.path.join(annotation_episode_path, annotation_file_name)
                                
                                if os.path.exists(annotation_file_path):
                                    with open(annotation_file_path, 'r') as f:
                                        annotations_data = json.load(f)
                                    
                                    vehicle_count = 0
                                    for ann in annotations_data.get("annotations", []):
                                        if ann.get("category_id") == 1:
                                            vehicle_count += 1
                                    
                                    if self.task_type == "vehicle_presence":
                                        self.labels.append(1.0 if vehicle_count > 0 else 0.0)
                                    else:
                                        self.labels.append(float(vehicle_count))
                                else:
                                    self.labels.append(0.0)

                            if max_images is not None and len(self.image_paths) >= max_images:
                                break
                    if max_images is not None and len(self.image_paths) >= max_images:
                        break
            if max_images is not None and len(self.image_paths) >= max_images:
                break

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
            if self.task_type in ["vehicle_presence", "vehicle_count"]:
                self.labels = self.labels[:max_images]

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        
        Args:
            idx (int): The index of the item to return.
            
        Returns:
            tuple: A tuple containing the image and the label.
        """
        # Step 1: Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Step 2: Return the image and label
        if self.task_type in ["vehicle_presence", "vehicle_count"]:
            return image, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return image, image

class FrameAutoencoder(nn.Module):
    """
    A frame autoencoder.
    """
    def __init__(self):
        """
        Initializes the autoencoder.
        """
        super(FrameAutoencoder, self).__init__()
        # Step 1: Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Step 2: Define the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        """
        Performs the forward pass.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PretrainedAutoencoder(nn.Module):
    """
    A pretrained autoencoder with a VGG16 encoder and a symmetrical decoder.
    This implementation uses a more explicit block-based architecture for clarity and robustness.
    """
    def __init__(self):
        """
        Initializes the autoencoder.
        """
        super(PretrainedAutoencoder, self).__init__()
        
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg16.features)

        self.encoder_block1 = nn.Sequential(*features[:4]) # Includes Conv2d, ReLU, Conv2d, ReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encoder_block2 = nn.Sequential(*features[5:9]) # Includes Conv2d, ReLU, Conv2d, ReLU
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encoder_block3 = nn.Sequential(*features[10:16]) # Includes Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encoder_block4 = nn.Sequential(*features[17:23]) # Includes Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encoder_block5 = nn.Sequential(*features[24:30]) # Includes Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.decoder_block5 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder_block4 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder_block3 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder_block2 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder_block1 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_block1(x)
        x, p1_indices = self.pool1(x)
        
        x = self.encoder_block2(x)
        x, p2_indices = self.pool2(x)
        
        x = self.encoder_block3(x)
        x, p3_indices = self.pool3(x)
        
        x = self.encoder_block4(x)
        x, p4_indices = self.pool4(x)
        
        x = self.encoder_block5(x)
        x, p5_indices = self.pool5(x)

        x = self.decoder_block5[0](x, p5_indices)
        x = self.decoder_block5[1:](x)
        
        x = self.decoder_block4[0](x, p4_indices)
        x = self.decoder_block4[1:](x)
        
        x = self.decoder_block3[0](x, p3_indices)
        x = self.decoder_block3[1:](x)
        
        x = self.decoder_block2[0](x, p2_indices)
        x = self.decoder_block2[1:](x)
        
        x = self.decoder_block1[0](x, p1_indices)
        x = self.decoder_block1[1:](x)
        
        return x

class FrameInfoPredictor(nn.Module):
    """
    A frame information predictor.
    """
    def __init__(self):
        """
        Initializes the predictor.
        """
        super(FrameInfoPredictor, self).__init__()
        # Step 1: Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Step 2: Define the prediction head
        self.prediction_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Performs the forward pass.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.prediction_head(x)
        return x

class FrameInfoCounter(nn.Module):
    """
    A frame information counter.
    """
    def __init__(self):
        """
        Initializes the counter.
        """
        super(FrameInfoCounter, self).__init__()
        # Step 1: Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Step 2: Define the regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Performs the forward pass.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.regression_head(x)
        return x

def train_model(model, train_dataloader, epochs, lr, model_path, task_type, device, fine_tune=False, fine_tune_layers=0, freeze_encoder=False):
    print(f"========== Training {task_type.replace('_', ' ').title()} Model ==========")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if fine_tune and isinstance(model, PretrainedAutoencoder):
        print(f"Fine-tuning pretrained model. Unfreezing last {fine_tune_layers} encoder blocks.")
        
        # Freeze all encoder blocks first
        encoder_blocks = [model.encoder_block1, model.encoder_block2, model.encoder_block3, model.encoder_block4, model.encoder_block5]
        for block in encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
            
        # Unfreeze the last N encoder blocks
        for block in encoder_blocks[-fine_tune_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Set up optimizer with differential learning rates
        fine_tune_params = [p for block in encoder_blocks[-fine_tune_layers:] for p in block.parameters()]
        decoder_params = list(model.decoder_block1.parameters()) + list(model.decoder_block2.parameters()) + list(model.decoder_block3.parameters()) + list(model.decoder_block4.parameters()) + list(model.decoder_block5.parameters())

        optimizer = optim.Adam([
            {'params': fine_tune_params, 'lr': lr * 0.1},
            {'params': decoder_params, 'lr': lr}
        ], lr=lr)

    elif freeze_encoder and isinstance(model, PretrainedAutoencoder):
        print("Freezing encoder. Training decoder only.")
        encoder_blocks = [model.encoder_block1, model.encoder_block2, model.encoder_block3, model.encoder_block4, model.encoder_block5]
        for block in encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        decoder_params = list(model.decoder_block1.parameters()) + list(model.decoder_block2.parameters()) + list(model.decoder_block3.parameters()) + list(model.decoder_block4.parameters()) + list(model.decoder_block5.parameters())
        optimizer = optim.Adam(decoder_params, lr=lr)
        
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCEWithLogitsLoss() if task_type == "vehicle_presence" else nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for data, target in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(img)
            
            if task_type in ["vehicle_presence", "vehicle_count"]:
                loss = criterion(output.squeeze(1), target)
            else:
                loss = criterion(output, img)
            
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def test_model(model, test_dataloader, model_path, task_type, device, output_dir):
    print(f"========== Testing {task_type.replace('_', ' ').title()} Model ==========")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in tqdm(test_dataloader, desc="Testing"):
            img, target = data.to(device), target.to(device)
            output = model(img)
            if task_type == "vehicle_presence":
                preds = (torch.sigmoid(output) >= 0.5).float()
                all_preds.extend(preds.squeeze(1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            elif task_type == "vehicle_count":
                all_preds.extend(output.squeeze(1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    if task_type == "vehicle_presence":
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Vehicle', 'Vehicle'], yticklabels=['No Vehicle', 'Vehicle'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        print("Confusion matrix saved.")
    elif task_type == "vehicle_count":
        mae = mean_absolute_error(all_targets, all_preds)
        print(f"Mean Absolute Error: {mae:.4f}")
        bins = [0, 1, 2, 3, 4, 5, np.inf]
        bin_labels = ["0", "1", "2", "3", "4", "5+"]
        true_binned = pd.cut(all_targets, bins=bins, labels=bin_labels, right=False)
        pred_binned = pd.cut(all_preds, bins=bins, labels=bin_labels, right=False)
        cm = confusion_matrix(true_binned, pred_binned, labels=bin_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=bin_labels, yticklabels=bin_labels)
        plt.xlabel('Predicted Count')
        plt.ylabel('True Count')
        plt.title(f'Binned Confusion Matrix (MAE: {mae:.2f})')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        print("Binned confusion matrix saved.")

def visualize_results(model, test_dataset, model_path, task_type, device, output_dir, batch_size):
    print(f"========== Visualizing {task_type.replace('_', ' ').title()} Results ==========")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    vis_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if task_type == "vehicle_presence":
        pass
    elif task_type == "vehicle_count":
        correct_examples, incorrect_examples = [], []
        with torch.no_grad():
            for data, target in tqdm(vis_dataloader, desc="Collecting visualizations"):
                img, target = data.to(device), target.to(device)
                output = model(img)
                preds = output.squeeze(1)
                
                for i in range(len(preds)):
                    pred_count, true_count = preds[i].item(), target[i].item()
                    img_display = img[i].cpu().permute(1, 2, 0).numpy()
                    error = abs(pred_count - true_count)
                    
                    if error < 0.5:
                        correct_examples.append((img_display, pred_count, true_count))
                    else:
                        incorrect_examples.append((img_display, pred_count, true_count))
        
        def save_vis(examples, category):
            for i in range(min(4, len(examples))):
                img_np, pred, true = examples[i]
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img_np)
                ax.set_title(f"{category.title()} Example {i+1}\nTrue: {int(true)}, Pred: {pred:.2f}")
                ax.axis('off')
                plt.savefig(os.path.join(output_dir, f"{category}_example_{i+1}.png"))
                plt.close(fig)
        
        save_vis(correct_examples, "correct")
        save_vis(incorrect_examples, "incorrect")
        print("Visualization complete.")

def get_dataloaders(
    data_dir,
    annotations_dir,
    task_type,
    model_choice,
    batch_size,
    max_images=None,
    balanced=False,
):
    """Prepares and returns data loaders for the specified task."""
    if model_choice == "pretrained":
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize_transform = transforms.Lambda(lambda x: x)

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize_transform,
        ]
    )

    try:
        full_dataset = L2DFrameDataset(
            root_dir=data_dir,
            transform=transform,
            task_type=task_type,
            annotations_dir=annotations_dir,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    if balanced and task_type in ["vehicle_presence", "vehicle_count"]:
        return get_balanced_dataloaders(full_dataset, task_type, batch_size)
    else:
        return get_standard_dataloaders(full_dataset, batch_size, max_images)


def get_balanced_dataloaders(full_dataset, task_type, batch_size):
    """Creates balanced datasets for classification/counting tasks."""
    print(f"Creating balanced dataset for {task_type}...")
    if task_type == "vehicle_presence":
        positive_indices = [i for i, (_, label) in enumerate(tqdm(full_dataset, desc="Scanning dataset")) if label == 1]
        negative_indices = [i for i, (_, label) in enumerate(tqdm(full_dataset, desc="Scanning dataset")) if label == 0]
        
        train_pos_indices = np.random.choice(positive_indices, 2000, replace=False)
        train_neg_indices = np.random.choice(negative_indices, 2000, replace=False)
        train_indices = np.concatenate([train_pos_indices, train_neg_indices])

        remaining_pos = list(set(positive_indices) - set(train_pos_indices))
        remaining_neg = list(set(negative_indices) - set(train_neg_indices))
        test_pos_indices = np.random.choice(remaining_pos, 100, replace=False)
        test_neg_indices = np.random.choice(remaining_neg, 100, replace=False)
        test_indices = np.concatenate([test_pos_indices, test_neg_indices])

    elif task_type == "vehicle_count":
        bins = {i: [] for i in range(6)}
        for i, (_, label) in enumerate(tqdm(full_dataset, desc="Scanning and binning dataset")):
            bins[min(int(label), 5)].append(i)

        train_indices, test_indices = [], []
        for _, indices in bins.items():
            np.random.shuffle(indices)
            train_count = min(len(indices), 400)
            test_count = min(len(indices) - train_count, 20)
            train_indices.extend(indices[:train_count])
            test_indices.extend(indices[train_count : train_count + test_count])

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    print(f"Balanced training set: {len(train_dataset)} samples. Test set: {len(test_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_standard_dataloaders(full_dataset, batch_size, max_images):
    """Creates standard split datasets."""
    dataset = Subset(full_dataset, range(max_images)) if max_images else full_dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
