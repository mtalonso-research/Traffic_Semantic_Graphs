import torch
import numpy as np
import sys
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add the cloned repository to the path
sys.path.append(os.path.abspath('lib/deeplabs'))

from modeling.deeplab import DeepLab

class DummyArgs:
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.norm = 'bn'

class PretrainedEncoder:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        # Instantiate the model
        args = DummyArgs()
        model = DeepLab(args, num_classes=19, freeze_bn=False, abn=False, deep_dec=True)
        
        # Load the pretrained weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        # The state dict might have a 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # The keys in the state dict might have a 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_frames(self, frame_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for episode_id, (image_dir, image_files) in tqdm(frame_paths.items(), desc="Processing episodes"):
            episode_embeddings = []
            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    # Get the feature map from the backbone
                    embedding, _ = self.model.backbone(image_tensor)
                episode_embeddings.append(embedding.cpu())
            
            output_path = os.path.join(output_dir, f"{episode_id}.pt")
            torch.save(torch.stack(episode_embeddings), output_path)