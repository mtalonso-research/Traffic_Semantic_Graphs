import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class L2DFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=None, task_type="reconstruction", annotations_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images
        self.task_type = task_type
        self.annotations_dir = annotations_dir
        self.image_paths = []
        self.labels = [] # For vehicle presence task

        # Collect image paths and labels
        for episode_dir in sorted(os.listdir(root_dir)):
            episode_path = os.path.join(root_dir, episode_dir)
            if os.path.isdir(episode_path):
                camera_path = os.path.join(episode_path, "observation.images.front_left") # Corrected camera path
                if os.path.isdir(camera_path):
                    for img_name in sorted(os.listdir(camera_path)):
                        if img_name.endswith(".jpg"):
                            self.image_paths.append(os.path.join(camera_path, img_name))
                            if task_type == "vehicle_presence":
                                # Construct annotation file path
                                annotation_episode_path = os.path.join(annotations_dir, episode_dir)
                                annotation_camera_path = os.path.join(annotation_episode_path, "front_left_Annotations")
                                annotation_file_name = img_name.replace(".jpg", ".json")
                                annotation_file_path = os.path.join(annotation_camera_path, annotation_file_name)
                                
                                # Load and parse annotation to get vehicle presence label
                                if os.path.exists(annotation_file_path):
                                    with open(annotation_file_path, 'r') as f:
                                        annotations_data = json.load(f)
                                    has_vehicle = False
                                    for ann in annotations_data.get("annotations", []):
                                        if ann.get("category_id") == 1: # Assuming category_id 1 is 'vehicle'
                                            has_vehicle = True
                                            break
                                    self.labels.append(1.0 if has_vehicle else 0.0) # Store as float for BCEWithLogitsLoss
                                else:
                                    self.labels.append(0.0) # Default to no vehicle if annotation missing

                            if max_images is not None and len(self.image_paths) >= max_images:
                                break
                    if max_images is not None and len(self.image_paths) >= max_images:
                        break
            if max_images is not None and len(self.image_paths) >= max_images:
                break

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
            if task_type == "vehicle_presence":
                self.labels = self.labels[:max_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.task_type == "vehicle_presence":
            return image, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return image, image

class FrameAutoencoder(nn.Module):
    def __init__(self):
        super(FrameAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PretrainedAutoencoder(nn.Module):
    def __init__(self):
        super(PretrainedAutoencoder, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use features as encoder, up to the last pooling layer
        self.encoder = nn.Sequential(*list(vgg16.features)[:30]) # Corresponds to output before last MaxPool (512 channels, 4x4 for 128x128 input)

        # Custom decoder to upsample to 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FrameInfoPredictor(nn.Module):
    def __init__(self):
        super(FrameInfoPredictor, self).__init__()
        # Encoder - Reusing the encoder part from FrameAutoencoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU()
        )
        # Prediction Head for binary classification (e.g., vehicle presence)
        # The output of the encoder is 512 channels, 4x4 spatial dimensions
        self.prediction_head = nn.Sequential(
            nn.Flatten(), # Flatten the 512x4x4 tensor
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output a single logit for binary classification
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.prediction_head(x)
        return x