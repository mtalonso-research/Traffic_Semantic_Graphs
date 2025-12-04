import argparse
import os
import torch
from tqdm import tqdm

from src.frame_encoding.pretrained_encoder import PretrainedEncoder

def get_frame_paths(root_dir):
    frame_paths = {}
    for episode_dir in os.listdir(root_dir):
        episode_path = os.path.join(root_dir, episode_dir)
        if not os.path.isdir(episode_path):
            continue
        
        image_dir = os.path.join(episode_path, "observation.images.front_left")
        if not os.path.isdir(image_dir):
            continue

        images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        frame_paths[image_dir] = images
    return frame_paths

def default_frame_encoding_processing(model_path, output_path, frames_root, run_encoding=False):
    if not run_encoding:
        run_encoding = True

    if run_encoding:
        print("========== Encoding Frames ==========")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frame_paths = get_frame_paths(frames_root)
        encoder = PretrainedEncoder(model_path)
        embeddings = encoder.encode_frames(frame_paths)
        torch.save(embeddings, output_path)
        print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode frames using a pretrained model.")
    parser.add_argument("--model_path", type=str, default="models/frame_encoder/deeplabv3+_r101-d8_512x1024_80k_sem_seg_bdd100k.pth", help="Path to the pretrained model.")
    parser.add_argument("--output_path", type=str, default="data/frame_embeddings/L2D/l2d_frame_embeddings.pt", help="Path to save the embeddings.")
    parser.add_argument("--frames_root", type=str, default="data/raw/L2D/frames", help="Root directory of the frames.")
    parser.add_argument("--run_encoding", action="store_true", help="Run the frame encoding step.")
    parser.add_argument("--all", action="store_true", help="Run all steps (default if no flags are set).")

    args = parser.parse_args()

    default_frame_encoding_processing(
        args.model_path,
        args.output_path,
        args.frames_root,
        run_encoding=args.run_encoding or args.all
    )