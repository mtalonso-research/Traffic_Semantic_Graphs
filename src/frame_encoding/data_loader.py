import os
import json
import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        labels_json_path,
        transform=None,
        episode_dir_format: str = "Episode{episode_id:06d}",
        # NEW args (all defaulted so L2D behavior is unchanged)
        frames_per_sample: Optional[int] = None,
        samples_per_episode: int = 1,
        min_frame_gap: int = 1,
        # NEW: optional domain name to prefix episode IDs (for multi-domain)
        domain_name: Optional[str] = None,
    ):
        """
        Args:
            root_dir: directory containing episode subfolders.
            labels_json_path: JSON mapping {episode_id_str: [label_0, ..., label_K-1]}.
            transform: torchvision-style transform applied to each frame.
            episode_dir_format: format string to map int episode_id -> subdir name.
                E.g. "Episode{episode_id:06d}" or
                     "Episode{episode_id:06d}/observation.images.front_left".

            frames_per_sample:
                - None (default): use ALL frames in the episode (L2D behavior).
                - N: each dataset item will contain exactly N frames from the episode
                     (subsampled / spaced out in time).

            samples_per_episode:
                - 1 (default): one item per episode (L2D behavior).
                - >1: create multiple different frame subsets per episode, for data augmentation.

            min_frame_gap:
                - Minimum gap in frame indices between consecutive frames in a subsampled set.
                - E.g., if frames are ~10 Hz and you want ~3s spacing, set ~30.

            domain_name:
                - Optional string tag, e.g. "L2D" or "NuPlan".
                - If provided, episode_id returned by __getitem__ becomes "L2D:<eid>" etc.
                  This makes IDs unique and domain-aware when mixing datasets.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.episode_dir_format = episode_dir_format
        self.frames_per_sample = frames_per_sample
        self.samples_per_episode = samples_per_episode
        self.min_frame_gap = max(1, min_frame_gap)
        self.domain_name = domain_name

        with open(labels_json_path, "r") as f:
            raw_labels = json.load(f)

        # sort keys numerically for reproducibility
        self.episode_ids: List[str] = sorted(raw_labels.keys(), key=lambda x: int(x))
        self.samples = []

        skipped_missing_dir = 0
        skipped_no_frames = 0

        for eid in self.episode_ids:
            int_id = int(eid)

            episode_rel = episode_dir_format.format(episode_id=int_id)
            episode_dir = os.path.join(root_dir, episode_rel)

            if not os.path.isdir(episode_dir):
                print(
                    f"[EpisodeDataset] WARNING: episode dir not found, skipping: "
                    f"{episode_dir} (episode id {eid}, mapped '{episode_rel}')"
                )
                skipped_missing_dir += 1
                continue

            frame_paths = sorted(glob.glob(os.path.join(episode_dir, "*.jpg")))
            if len(frame_paths) == 0:
                print(
                    f"[EpisodeDataset] WARNING: no .jpg frames in {episode_dir}, skipping episode {eid}"
                )
                skipped_no_frames += 1
                continue

            target = torch.tensor(raw_labels[eid], dtype=torch.long)
            T = len(frame_paths)

            # Build a domain-aware episode_id string
            if self.domain_name is not None:
                ep_id_str = f"{self.domain_name}:{eid}"
            else:
                ep_id_str = eid

            # --- OLD BEHAVIOR: no subsampling, no augmentation ---
            if self.frames_per_sample is None and self.samples_per_episode == 1:
                self.samples.append(
                    {
                        "episode_id": ep_id_str,
                        "frame_paths": frame_paths,
                        "target": target,
                        # no "frame_indices" key -> use all frames
                    }
                )
                continue

            # --- NEW BEHAVIOR: subsampling and/or augmentation ---
            # We will create `samples_per_episode` entries per episode, each
            # with a different frame_indices subset.
            for _ in range(self.samples_per_episode):
                indices = self._select_frame_indices(T)
                self.samples.append(
                    {
                        "episode_id": ep_id_str,
                        "frame_paths": frame_paths,
                        "target": target,
                        "frame_indices": indices,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "EpisodeDataset found no valid episodes with frames. "
                f"Skipped {skipped_missing_dir} with missing dirs and "
                f"{skipped_no_frames} no-frames. "
                "Check your root_dir / episode_dir_format / labels_json_path."
            )

        print(
            f"[EpisodeDataset] Loaded {len(self.samples)} items "
            f"(from {len(self.episode_ids)} episodes; "
            f"skipped {skipped_missing_dir} missing-dir, "
            f"{skipped_no_frames} no-frames)."
        )

    def _select_frame_indices(self, T: int) -> List[int]:
        """
        Select a subset of frame indices for one sample of an episode.

        Guarantees:
          - length(indices) == frames_per_sample (if possible)
          - indices are sorted
          - consecutive indices differ by at least min_frame_gap,
            if T is large enough to allow that; otherwise falls back
            to approximately evenly spaced indices.
        """
        # If frames_per_sample is None, use all frames
        if self.frames_per_sample is None or self.frames_per_sample >= T:
            return list(range(T))

        N = self.frames_per_sample
        gap = self.min_frame_gap

        # Check if we can enforce the gap without exceeding T
        max_needed = (N - 1) * gap + 1
        if max_needed <= T:
            # We can pick a random starting point and then step by gap
            start_max = T - max_needed
            if start_max <= 0:
                start = 0
            else:
                start = np.random.randint(0, start_max + 1)
            indices = [start + i * gap for i in range(N)]
            return indices

        # Otherwise, fall back to roughly evenly spaced indices across [0, T-1]
        indices = np.linspace(0, T - 1, N).round().astype(int).tolist()
        return sorted(set(indices))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        target = sample["target"]
        episode_id = sample["episode_id"]

        # If we stored frame_indices, only load those; otherwise load all.
        frame_indices = sample.get("frame_indices", None)
        if frame_indices is None:
            paths_to_load = frame_paths
        else:
            paths_to_load = [frame_paths[i] for i in frame_indices]

        frames = []
        for fp in paths_to_load:
            img = Image.open(fp).convert("RGB")
            if self.transform is None:
                raise RuntimeError(
                    "EpisodeDataset requires a transform (e.g. Resize+ToTensor+Normalize)."
                )
            img = self.transform(img)
            frames.append(img)

        frames_tensor = torch.stack(frames, dim=0)  # (T_sub, C, H, W)
        return frames_tensor, target, episode_id


from typing import List, Tuple

def collate_episodes(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    """
    batch: list of (frames, target, episode_id)
        frames: (T_i, C, H, W)
        target: (K,)
        episode_id: str

    returns:
        frames_list: list of length B, each (T_i, C, H, W)
        targets: (B, K)
        episode_ids: list[str]
    """
    frames_list, targets_list, ids_list = zip(*batch)

    frames_list = list(frames_list)
    targets = torch.stack(targets_list, dim=0)  # (B, K)
    episode_ids = list(ids_list)

    return frames_list, targets, episode_ids
