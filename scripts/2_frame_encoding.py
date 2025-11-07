import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.frame_autoencoder import FrameAutoencoder, PretrainedAutoencoder, FrameInfoPredictor, L2DFrameDataset

parser = argparse.ArgumentParser(description="Train and test a frame autoencoder.")
parser.add_argument("--train", action="store_true", help="Run the autoencoder training step.")
parser.add_argument("--test", action="store_true", help="Run the autoencoder testing/evaluation step.")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
parser.add_argument("--model_path", type=str, default="./models/frame_autoencoder.pth", help="Path to save/load the autoencoder model.")
parser.add_argument("--data_dir", type=str, default="./data/raw/L2D/frames", help="Directory containing the raw frame data.")
parser.add_argument("--annotations_dir", type=str, default="./data/processed_frames/L2D", help="Directory containing the frame annotations.")
parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to use from the dataset.")
parser.add_argument("--visualize", action="store_true", help="Save original and reconstructed images for visualization.")
parser.add_argument("--model_choice", type=str, default="homemade", choices=["homemade", "pretrained"], help="Choose between 'homemade' and 'pretrained' autoencoder.")
parser.add_argument("--task_type", type=str, default="reconstruction", choices=["reconstruction", "vehicle_presence"], help="Choose between 'reconstruction' and 'vehicle_presence' tasks.")
parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the encoder layers when using a pretrained model.")
parser.add_argument("--all", action="store_true", help="Run all steps (default if no flags are set).")

args = parser.parse_args()

def run_frame_encoding(run_train=False, run_test=False, epochs=10, batch_size=32, lr=1e-4, model_path="./models/frame_autoencoder.pth", data_dir="./data/raw/L2D/frames", annotations_dir="./data/processed_frames/L2D", max_images=None, visualize=False, model_choice="homemade", task_type="reconstruction", freeze_encoder=False):
    if not any([run_train, run_test, visualize]):
        run_train = run_test = True

    # Define transformations for the images
    if model_choice == "pretrained":
        # VGG-specific normalization
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        # Our homemade model expects values in [0, 1]
        # No normalization for homemade model for now, as it was before
        normalize_transform = transforms.Lambda(lambda x: x) # No-op transform

    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize images to a fixed size
        transforms.ToTensor(),         # Convert images to PyTorch tensors
        normalize_transform
    ])

    # Load dataset
    try:
        full_dataset = L2DFrameDataset(root_dir=data_dir, transform=transform, task_type=task_type, annotations_dir=annotations_dir)
        if max_images is not None and max_images < len(full_dataset):
            dataset = Subset(full_dataset, range(max_images))
            print(f"Using a subset of {len(dataset)} images from {data_dir}")
        else:
            dataset = full_dataset
            print(f"Found {len(dataset)} images in {data_dir}")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        print("Please ensure the data directory exists and contains images.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task_type == "vehicle_presence":
        model = FrameInfoPredictor().to(device)
        print("Using FrameInfoPredictor for vehicle presence detection.")
        criterion = nn.BCEWithLogitsLoss()
    elif model_choice == "homemade":
        model = FrameAutoencoder().to(device)
        print("Using homemade FrameAutoencoder.")
        criterion = nn.MSELoss()
    elif model_choice == "pretrained":
        model = PretrainedAutoencoder().to(device)
        print("Using pretrained VGG16-based Autoencoder.")
        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder layers frozen.")
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if run_train:
        print("========== Training Frame Autoencoder ==========")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        for epoch in range(epochs):
            # Wrap dataloader with tqdm for a progress bar
            for data, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                img = data.to(device)
                target = target.to(device) # Move target to device

                optimizer.zero_grad()
                output = model(img)

                if task_type == "vehicle_presence":
                    loss = criterion(output.squeeze(1), target) # Squeeze output for BCEWithLogitsLoss
                else: # Reconstruction task
                    # Apply inverse normalization before calculating loss for pretrained model
                    if model_choice == "pretrained":
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                        img_denorm = img * std + mean
                        output_denorm = output * std + mean
                        loss = criterion(output_denorm, img_denorm)
                    else:
                        loss = criterion(output, img)

                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if run_test:
        print("========== Testing Frame Autoencoder ==========")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for data, target in tqdm(dataloader, desc="Testing"):
                    img = data.to(device)
                    target = target.to(device)
                    output = model(img)

                    if task_type == "vehicle_presence":
                        loss = criterion(output.squeeze(1), target)
                    else: # Reconstruction task
                        # Apply inverse normalization before calculating loss for pretrained model
                        if model_choice == "pretrained":
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                            img_denorm = img * std + mean
                            output_denorm = output * std + mean
                            loss = criterion(output_denorm, img_denorm)
                        else:
                            loss = criterion(output, img)
                    total_loss += loss.item()
            print(f'Average Test Loss: {total_loss / len(dataloader):.4f}')
        else:
            print(f"Model not found at {model_path}. Please train the model first.")

    if visualize:
        print("========== Visualizing Results ==========")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train the model first.")
            return

        model.load_state_dict(torch.load(model_path))
        model.eval()

        if task_type == "reconstruction":
            if model_choice == "homemade":
                output_dir = "figures/frame_autoencoder/homemade_image_reconstruction"
            elif model_choice == "pretrained":
                output_dir = "figures/frame_autoencoder/pretrained_image_reconstruction"
            else:
                output_dir = "figures/frame_autoencoder/unknown_image_reconstruction"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving visualizations to {output_dir}")

            vis_dataset_size = min(len(dataset), 100) # Visualize up to 100 images
            vis_dataloader = DataLoader(Subset(dataset, range(vis_dataset_size)), batch_size=1, shuffle=False)

            results = [] # Store (loss, original_img, reconstructed_img)

            with torch.no_grad():
                for i, (data, _) in tqdm(enumerate(vis_dataloader), desc="Collecting visualizations", total=len(vis_dataloader)):
                    img = data.to(device)
                    output = model(img)

                    # Denormalize for display and loss calculation
                    if model_choice == "pretrained":
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                        original_img_display = img * std + mean
                        reconstructed_img_display = output * std + mean
                    else:
                        original_img_display = img
                        reconstructed_img_display = output

                    loss = nn.MSELoss(reduction='none')(reconstructed_img_display, original_img_display).mean().item() # Per-image loss

                    original_img_np = original_img_display.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    reconstructed_img_np = reconstructed_img_display.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

                    results.append((loss, original_img_np, reconstructed_img_np))

            results.sort(key=lambda x: x[0]) # Sort by loss (ascending)

            # Save 4 'good' examples (lowest loss)
            print("Saving 4 'good' examples...")
            for i in range(min(4, len(results))):
                loss, original, reconstructed = results[i]
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(original)
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(reconstructed)
                axes[1].set_title(f"Reconstructed (Loss: {loss:.4f})")
                axes[1].axis('off')
                plt.savefig(os.path.join(output_dir, f"good_example_{i+1}.png"))
                plt.close(fig)

            # Save 4 'bad' examples (highest loss)
            print("Saving 4 'bad' examples...")
            for i in range(min(4, len(results))):
                loss, original, reconstructed = results[-(i+1)]
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(original)
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(reconstructed)
                axes[1].set_title(f"Reconstructed (Loss: {loss:.4f})")
                axes[1].axis('off')
                plt.savefig(os.path.join(output_dir, f"bad_example_{i+1}.png"))
                plt.close(fig)
            print("Visualization complete.")

        elif task_type == "vehicle_presence":
            output_dir = "figures/frame_autoencoder/vehicle_presence_prediction"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving visualizations to {output_dir}")

            vis_dataset_size = min(len(dataset), 100) # Visualize up to 100 images
            vis_dataloader = DataLoader(Subset(dataset, range(vis_dataset_size)), batch_size=1, shuffle=False)

            correct_predictions = []
            incorrect_predictions = []

            with torch.no_grad():
                for i, (data, target) in tqdm(enumerate(vis_dataloader), desc="Collecting visualizations", total=len(vis_dataloader)):
                    img = data.to(device) # Image tensor
                    
                    # Determine the effective index in the original L2DFrameDataset
                    # vis_dataloader.dataset is a Subset; its indices map to `dataset`.
                    # `dataset` might be L2DFrameDataset or a Subset of L2DFrameDataset.
                    current_dataset_idx = vis_dataloader.dataset.indices[i]
                    if isinstance(dataset, Subset):
                        # If `dataset` is a Subset, then current_dataset_idx is an index into `dataset.indices`.
                        # We need to map it back to the original full_dataset.
                        original_dataset_idx = dataset.indices[current_dataset_idx]
                    else:
                        # If `dataset` is the original L2DFrameDataset, then current_dataset_idx is directly the index.
                        original_dataset_idx = current_dataset_idx

                    img_path = full_dataset.image_paths[original_dataset_idx] # Original image path

                    target = target.to(device)
                    output = model(img)
                    
                    predicted_prob = torch.sigmoid(output).item()
                    predicted_label = 1 if predicted_prob >= 0.5 else 0
                    true_label = int(target.item())

                    # Denormalize image for display
                    if model_choice == "pretrained":
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                        img_display = img * std + mean
                    else:
                        img_display = img
                    img_np = img_display.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

                    if predicted_label == true_label:
                        correct_predictions.append((img_np, predicted_prob, true_label, img_path))
                    else:
                        incorrect_predictions.append((img_np, predicted_prob, true_label, img_path))
            
            # Save 4 correct examples
            print("Saving 4 correctly predicted examples...")
            for i in range(min(4, len(correct_predictions))):
                img_np, predicted_prob, true_label, img_path = correct_predictions[i]
                print(f"Correct Example {i+1} (Path: {img_path})")
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img_np)
                ax.set_title(f"True: {true_label}, Pred: {predicted_prob:.2f} (Correct)")
                ax.axis('off')
                plt.savefig(os.path.join(output_dir, f"correct_example_{i+1}.png"))
                plt.close(fig)

            # Save 4 incorrect examples
            print("Saving 4 incorrectly predicted examples...")
            for i in range(min(4, len(incorrect_predictions))):
                img_np, predicted_prob, true_label, img_path = incorrect_predictions[i]
                print(f"Incorrect Example {i+1} (Path: {img_path})")
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img_np)
                ax.set_title(f"True: {true_label}, Pred: {predicted_prob:.2f} (Incorrect)")
                ax.axis('off')
                plt.savefig(os.path.join(output_dir, f"incorrect_example_{i+1}.png"))
                plt.close(fig)
            print("Visualization complete.")

if __name__ == "__main__":
    run_frame_encoding(
        run_train=args.train or args.all,
        run_test=args.test or args.all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_path=args.model_path,
        data_dir=args.data_dir,
        annotations_dir=args.annotations_dir,
        max_images=args.max_images,
        visualize=args.visualize or args.all,
        model_choice=args.model_choice,
        task_type=args.task_type,
        freeze_encoder=args.freeze_encoder
    )