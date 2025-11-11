import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frame_autoencoder import (
    L2DFrameDataset,
    FrameAutoencoder,
    PretrainedAutoencoder,
    FrameInfoPredictor,
    FrameInfoCounter,
    train_model,
    test_model,
    visualize_results,
    get_dataloaders,
    get_standard_dataloaders,
)


TASK_CONFIGS = {
    "reconstruction": {
        "homemade": {
            "model_class": FrameAutoencoder,
            "model_path": "./models/frame_autoencoder_homemade.pth",
            "output_dir": "figures/frame_autoencoder/homemade_image_reconstruction",
        },
        "pretrained": {
            "model_class": PretrainedAutoencoder,
            "model_path": "./models/frame_autoencoder_pretrained.pth",
            "output_dir": "figures/frame_autoencoder/pretrained_image_reconstruction",
        },
    },
    "vehicle_presence": {
        "model_class": FrameInfoPredictor,
        "model_path": "./models/frame_vehicle_presence.pth",
        "output_dir": "figures/frame_autoencoder/vehicle_presence_prediction",
    },
    "vehicle_count": {
        "model_class": FrameInfoCounter,
        "model_path": "./models/frame_vehicle_count.pth",
        "output_dir": "figures/frame_autoencoder/vehicle_counter_prediction",
    },
}


def run_task(args):
    if args.task_type == "reconstruction":
        config = TASK_CONFIGS["reconstruction"][args.model_choice]
    else:
        config = TASK_CONFIGS[args.task_type]

    model_class = config["model_class"]
    model_path = args.model_path or config["model_path"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading ---
    train_loader, test_loader = get_dataloaders(
        args.data_dir,
        args.annotations_dir,
        args.task_type,
        args.model_choice,
        args.batch_size,
        args.max_images,
        args.balanced,
    )
    if not train_loader or not test_loader:
        return

    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)

    # --- Task Execution ---
    if args.train:
        print(f"--- Training for {args.task_type} ({args.model_choice if args.task_type == 'reconstruction' else 'default'}) ---")
        train_model(
            model,
            train_loader,
            args.epochs,
            args.lr,
            model_path,
            args.task_type,
            device,
            fine_tune=args.fine_tune,
            fine_tune_layers=args.fine_tune_layers,
            freeze_encoder=args.freeze_encoder,
        )

    if args.test:
        print(f"--- Testing for {args.task_type} ({args.model_choice if args.task_type == 'reconstruction' else 'default'}) ---")
        test_model(model, test_loader, model_path, args.task_type, device, output_dir)

    if args.visualize:
        print(f"--- Visualizing for {args.task_type} ({args.model_choice if args.task_type == 'reconstruction' else 'default'}) ---")
        # For visualization, we need the dataset, not the dataloader
        _, test_dataset = get_standard_dataloaders(train_loader.dataset.dataset, args.batch_size, args.max_images)
        visualize_results(model, test_dataset, model_path, args.task_type, device, output_dir, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models for frame analysis.")
    
    # Task selection
    parser.add_argument(
        "--task_type",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "vehicle_presence", "vehicle_count"],
        help="Choose the task type.",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        default="homemade",
        choices=["homemade", "pretrained"],
        help="For reconstruction, choose between 'homemade' and 'pretrained' autoencoder.",
    )

    # Actions
    parser.add_argument("--train", action="store_true", help="Run the model training step.")
    parser.add_argument("--test", action="store_true", help="Run the model testing/evaluation step.")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of model results.")
    parser.add_argument("--all", action="store_true", help="Run all steps: train, test, and visualize.")

    # Model and Data paths
    parser.add_argument("--model_path", type=str, default=None, help="Override default path to save/load the model.")
    parser.add_argument("--data_dir", type=str, default="./data/raw/L2D/frames", help="Directory for raw frame data.")
    parser.add_argument("--annotations_dir", type=str, default="./data/annotations/L2D", help="Directory for frame annotations.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    
    # Dataset options
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to use.")
    parser.add_argument("--balanced", action="store_true", help="Use a balanced dataset for classification/counting tasks.")
    
    # Model-specific options
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder layers for pretrained models.")
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning for the pretrained model.")
    parser.add_argument("--fine_tune_layers", type=int, default=0, help="Number of final encoder layers to fine-tune.")

    args = parser.parse_args()

    if args.all:
        args.train = args.test = args.visualize = True
    
    if not any([args.train, args.test, args.visualize]):
        print("No action specified. Please select --train, --test, --visualize, or --all.")
    else:
        run_task(args)