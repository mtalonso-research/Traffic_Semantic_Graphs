import argparse
import os
import torch
from tqdm import tqdm
import itertools

from src.frame_encoding.pretrained_encoder import PretrainedEncoder

def get_frame_paths(root_dir, min_ep=None, max_ep=None):
    frame_paths = {}
    
    # Use sorted list to ensure consistent order
    episode_dirs = sorted(os.listdir(root_dir))
    
    filtered_episode_dirs = []
    for episode_dir in episode_dirs:
        try:
            # Assumes format "Episode" + number, e.g. "Episode000001"
            episode_id = int(episode_dir.replace("Episode", "").replace("/", ""))
            if (min_ep is None or episode_id >= min_ep) and \
               (max_ep is None or episode_id <= max_ep):
                filtered_episode_dirs.append(episode_dir)
        except ValueError:
            # Not a numeric episode ID, skip or handle as needed
            continue

    for episode_dir in filtered_episode_dirs:
        episode_path = os.path.join(root_dir, episode_dir)
        if not os.path.isdir(episode_path):
            continue
        
        image_dir = os.path.join(episode_path, "observation.images.front_left")
        if not os.path.isdir(image_dir):
            continue

        images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        frame_paths[episode_dir] = (image_dir, images)
    return frame_paths

def default_frame_encoding_processing(model_path, output_dir, frames_root, run_encoding=False, min_ep=None, max_ep=None):
    if not run_encoding:
        run_encoding = True

    if run_encoding:
        print("========== Encoding Frames ==========")
        if min_ep is not None or max_ep is not None:
            print(f"--- Running in test mode: processing episodes from {min_ep if min_ep is not None else 'start'} to {max_ep if max_ep is not None else 'end'} ---")
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = get_frame_paths(frames_root, min_ep=min_ep, max_ep=max_ep)
        encoder = PretrainedEncoder(model_path)
        encoder.encode_frames(frame_paths, output_dir)
        print(f"Embeddings saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode frames using a pretrained model.")
    parser.add_argument("--model_path", type=str, default="models/frame_encoder/deeplabv3+_r101-d8_512x1024_80k_sem_seg_bdd100k.pth", help="Path to the pretrained model.")
    parser.add_argument("--output_dir", type=str, default="data/frame_embeddings/L2D/", help="Directory to save the embeddings.")
    parser.add_argument("--frames_root", type=str, default="data/raw/L2D/frames", help="Root directory of the frames.")
    parser.add_argument("--run_encoding", action="store_true", help="Run the frame encoding step.")
    parser.add_argument("--all", action="store_true", help="Run all steps (default if no flags are set).")
    parser.add_argument("--min_ep", type=int, default=None, help="Minimum episode ID to process.")
    parser.add_argument("--max_ep", type=int, default=None, help="Maximum episode ID to process.")

    args = parser.parse_args()

    default_frame_encoding_processing(
        args.model_path,
        args.output_dir,
        args.frames_root,
        run_encoding=args.run_encoding or args.all,
        min_ep=args.min_ep,
        max_ep=args.max_ep
    )