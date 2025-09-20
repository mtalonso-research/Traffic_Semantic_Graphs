import os
from huggingface_hub import hf_hub_download
import cv2
import shutil
from functions.utils import get_chunk_num
import pandas as pd
import subprocess
import imageio_ffmpeg

def data_downloader(min_ep,max_ep=-1,
                    tabular_data_dir='../data/raw/L2D/tabular',
                    frames_dir='../data/raw/L2D/frames',
                    features=None, n_secs=3):
    
    # Ensure target directories exist
    os.makedirs(tabular_data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

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

    for ep_num in iterable:
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

def extract_frames_from_hf_video(chunk, video_key, episode_num,
                                 output_dir, n_seconds=1, filename_prefix="frame"):
    """
    Downloads an AV1-encoded video from Hugging Face, converts it to H.264, and extracts frames
    every `n_seconds` into JPGs using OpenCV. Output is saved in the structure:
    output_dir/Episode{episode_num:06d}/{video_key}/frame_00000.jpg ...
    Returns: number of frames saved
    """

    # Step 1: Download original video (likely AV1)
    file_path = hf_hub_download(
        repo_id="yaak-ai/L2D",
        filename=f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_num:06d}.mp4",
        repo_type="dataset"
    )

    # Step 2: Convert to H.264 if not already
    h264_path = file_path.replace(".mp4", "_h264.mp4")
    if not os.path.exists(h264_path):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_path, "-y",
            "-hwaccel", "none",
            "-i", file_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-an",
            h264_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("ffmpeg conversion failed:\n", e.stderr)
            return 0

    # Step 3: Extract frames
    cap = cv2.VideoCapture(h264_path)
    if not cap.isOpened():
        print(f"Failed to open converted video: {h264_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Invalid FPS. Cannot proceed.")
        cap.release()
        return 0

    frame_interval = max(1, int(fps * n_seconds))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_dir, f"Episode{episode_num:06d}", video_key,
                f"{filename_prefix}_{saved_count:05d}.jpg"
            )
            os.makedirs(os.path.dirname(frame_filename), exist_ok=True)
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count
