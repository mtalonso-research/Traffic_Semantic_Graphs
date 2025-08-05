import os
from huggingface_hub import hf_hub_download
import cv2
import shutil
from functions.utils_L2D import get_chunk_num
import pandas as pd

def data_downloader(min_ep,max_ep=-1,
                    tabular_data_dir='../data/raw/L2D/tabular',
                    frames_dir='../data/raw/L2D/frames',
                    features=None, n_secs=3):
    
    if max_ep == -1: max_ep = min_ep + 1
    if features is None:
        features = {
            "tabular": True,
            "frames": {
                'observation.images.front_left': True,
                'observation.images.left_backward': True,
                'observation.images.left_forward': True,
                'observation.images.map': True,
                'observation.images.rear': True,
                'observation.images.right_backward': True,
                'observation.images.right_forward': True,
            }
        }

    for ep_num in range(min_ep, max_ep):
        chunk = get_chunk_num(ep_num)
        
        if features.get("tabular", False):
            download_tabular_data(chunk, n_secs, ep_num, tabular_data_dir)
        
        if "frames" in features:
            for vid_key, enabled in features["frames"].items():
                if enabled:
                    extract_frames_from_hf_video(chunk, vid_key, ep_num, frames_dir, n_seconds=n_secs)

def download_tabular_data(chunk,n_sec,episode_num,output_dir):
    file_path = hf_hub_download(
        repo_id="yaak-ai/L2D",
        filename=f"data/chunk-{chunk:03d}/episode_{episode_num:06d}.parquet",
        repo_type="dataset",
    )
    step = int(10*n_sec)
    df = pd.read_parquet(file_path)
    df = df.iloc[::step]
    target_file = os.path.join(output_dir, f"episode_{episode_num:06d}.parquet")
    df.to_parquet(target_file, index=False)

def extract_frames_from_hf_video(chunk,video_key,episode_num,
                                 output_dir,n_seconds=1,filename_prefix="frame"):
    # Get video file from Hugging Face Hub
    file_path = hf_hub_download(
        repo_id="yaak-ai/L2D",
        filename=f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_num:06d}.mp4",
        repo_type="dataset"
    )

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open video with OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Failed to open video: {file_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("FPS is 0, cannot proceed.")
        return

    frame_interval = int(fps * n_seconds)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir,f"Episode{episode_num:06d}",video_key, f"{filename_prefix}_{saved_count:05d}.jpg")
            os.makedirs(os.path.dirname(frame_filename), exist_ok=True)
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count