import os
from huggingface_hub import hf_hub_download
import cv2
import shutil
from functions.utils import get_chunk_num
import pandas as pd
import subprocess
import imageio_ffmpeg

def download_metadata(metadata_dir="../data/raw/L2D/metadata"):
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, "metadata.parquet")

    if not os.path.exists(metadata_path):
        print("Downloading global metadata parquet...")
        file_path = hf_hub_download(
            repo_id="yaak-ai/L2D",
            filename="meta/episodes/chunk-000/file-000.parquet",
            repo_type="dataset",
        )
        shutil.copy(file_path, metadata_path)
        print(f"Saved metadata to {metadata_path}")
    else:
        print(f"Using cached metadata at {metadata_path}")

    return metadata_path


def data_downloader(min_ep, max_ep=-1,
                    tabular_data_dir="../data/raw/L2D/tabular",
                    frames_dir="../data/raw/L2D/frames",
                    metadata_dir="../data/raw/L2D/metadata",
                    features=None, n_secs=3):

    # Ensure target directories exist
    os.makedirs(tabular_data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Determine which episodes to process
    if not isinstance(min_ep, list):
        if max_ep == -1:
            max_ep = min_ep + 1
        iterable = range(min_ep, max_ep)
    else:
        iterable = min_ep

    # Default features if none specified
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

    # Load metadata once
    metadata_path = download_metadata(metadata_dir)
    metadata_df = pd.read_parquet(metadata_path)

    # Initialize ephemeral cache for chunk/file reuse
    cached_files = {"parquet": {}, "video": {}}

    for ep_num in iterable:
        ep_row = metadata_df.loc[metadata_df["episode_index"] == ep_num]
        if ep_row.empty:
            print(f"⚠️ Episode {ep_num} not found in metadata, skipping.")
            continue
        ep_row = ep_row.iloc[0]
        chunk = int(ep_row["data/chunk_index"])
        file_index = int(ep_row["data/file_index"])

        if features.get("tabular", False):
            download_tabular_data(ep_row, cached_files, n_secs, ep_num, tabular_data_dir)

        if "frames" in features:
            for vid_key, enabled in features["frames"].items():
                if enabled:
                    extract_frames_from_hf_video(ep_row, vid_key, cached_files, frames_dir, n_seconds=n_secs)

    # Cleanup cache (delete any remaining downloaded master files)
    for f in list(cached_files["parquet"].values()) + list(cached_files["video"].values()):
        if os.path.exists(f):
            os.remove(f)
    cached_files.clear()
    print("✅ All done. Temporary files cleaned up.")


def download_tabular_data(ep_row, cached_files, n_sec, episode_num, output_dir):
    chunk = int(ep_row["data/chunk_index"])
    file_index = int(ep_row["data/file_index"])
    from_idx = int(ep_row["dataset_from_index"])
    to_idx = int(ep_row["dataset_to_index"])

    cache_key = f"{chunk}_{file_index}"

    # Download master parquet if not cached
    if cache_key not in cached_files["parquet"]:
        print(f"Downloading tabular chunk {chunk}, file {file_index}...")
        file_path = hf_hub_download(
            repo_id="yaak-ai/L2D",
            filename=f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet",
            repo_type="dataset",
        )
        cached_files["parquet"][cache_key] = file_path
    else:
        file_path = cached_files["parquet"][cache_key]

    df = pd.read_parquet(file_path)
    step = int(10 * n_sec)
    sliced_df = df.iloc[from_idx:to_idx:step]
    target_file = os.path.join(output_dir, f"episode_{episode_num:06d}.parquet")
    sliced_df.to_parquet(target_file, index=False)
    print(f"✅ Saved tabular episode {episode_num} → {target_file}")


def extract_frames_from_hf_video(ep_row, video_key, cached_files,
                                 output_dir, n_seconds=1, filename_prefix="frame"):

    chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
    file_index = int(ep_row[f"videos/{video_key}/file_index"])
    from_ts = ep_row[f"videos/{video_key}/from_timestamp"]
    to_ts = ep_row[f"videos/{video_key}/to_timestamp"]
    episode_num = int(ep_row["episode_index"])

    if pd.isna(chunk) or pd.isna(file_index):
        print(f"Skipping {video_key} for episode {episode_num} (missing metadata).")
        return 0

    cache_key = f"{chunk}_{file_index}"

    # Step 1: Download master video if not cached
    if cache_key not in cached_files["video"]:
        print(f"Downloading video chunk {chunk}, file {file_index} for {video_key}...")
        file_path = hf_hub_download(
            repo_id="yaak-ai/L2D",
            filename=f"videos/{video_key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4",
            repo_type="dataset",
        )
        cached_files["video"][cache_key] = file_path
    else:
        file_path = cached_files["video"][cache_key]

    # Step 2: Cut relevant portion with ffmpeg (avoiding full decode)
    h264_path = file_path.replace(".mp4", f"_{episode_num:06d}_{video_key.replace('/', '_')}_cut.mp4")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, "-y",
        "-ss", str(from_ts),
        "-to", str(to_ts),
        "-i", file_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        h264_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ ffmpeg cut failed for {video_key} episode {episode_num}:\n", e.stderr)
        return 0

    # Step 3: Extract frames
    cap = cv2.VideoCapture(h264_path)
    if not cap.isOpened():
        print(f"Failed to open cut video for {video_key}, episode {episode_num}.")
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
    os.remove(h264_path)  # cleanup
    print(f"✅ Extracted {saved_count} frames from {video_key} for episode {episode_num}.")
    return saved_count
