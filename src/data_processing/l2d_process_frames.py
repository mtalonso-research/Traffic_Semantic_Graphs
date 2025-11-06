import os
import cv2
import numpy as np
import json
import sys
import torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from collections import defaultdict, deque
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import sys
import math
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
CAMERA_CONFIG = {'fps': 10, 'time_step': 2}
camera_trackers = {}
last_detections_cache = {}

VERBOSE_MODE = False

# --- Warning handling (quiet known noisy third-party warnings) ---
import os, warnings
if not os.environ.get('STRICT_WARNINGS'):
    try:
        from torch.jit import TracerWarning
        warnings.filterwarnings('ignore', category=TracerWarning)
    except Exception:
        pass
    warnings.filterwarnings('ignore', message=r'.*torch\.meshgrid.*indexing.*', category=UserWarning)
    warnings.filterwarnings('ignore', message=r'.*Converting a tensor to a Python boolean.*', category=UserWarning)
    warnings.filterwarnings('ignore', message=r'.*results are registered as constants in the trace.*', category=UserWarning)
    warnings.filterwarnings('ignore', message=r'.*Iterating over a tensor might cause the trace to be incorrect.*', category=UserWarning)
    warnings.filterwarnings('ignore', message=r'.*loss_type=None.*ForCausalLMLoss.*', category=UserWarning)
# --- end warning handling ---

def quick_setup_depth_pro(verbose):
    """Quick setup function to download Depth Pro model"""
    if not os.path.exists('./ml-depth-pro'):
        return False
    checkpoint_dir = './ml-depth-pro/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id='apple/DepthPro', filename='depth_pro.pt', local_dir=checkpoint_dir)
        return True
    except Exception as e:
        return False

def initialize_depth_pro(DEPTH_PRO_AVAILABLE, depth_pro, verbose):
    """Initialize the Depth Pro model"""
    if not DEPTH_PRO_AVAILABLE:
        return (None, None, None)
    try:
        if not download_depth_pro_model(verbose):
            return (None, None, None)
        original_dir = os.getcwd()
        depth_pro_dir = './ml-depth-pro'
        try:
            os.chdir(depth_pro_dir)
            model, transform = depth_pro.create_model_and_transforms()
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return (model, transform, device)
        finally:
            os.chdir(original_dir)
    except FileNotFoundError as e:
        return try_alternative_download(verbose)
    except Exception as e:
        return (None, None, None)

def download_depth_pro_model(verbose):
    """Download Depth Pro model if not exists"""
    checkpoint_dir = './ml-depth-pro/checkpoints'
    model_path = os.path.join(checkpoint_dir, 'depth_pro.pt')
    if os.path.exists(model_path):
        return True
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(repo_id='apple/DepthPro', filename='depth_pro.pt', local_dir=checkpoint_dir)
        return True
    except Exception as e:
        return False

def try_alternative_download(verbose):
    """Try alternative method to get Depth Pro working"""
    try:
        import subprocess
        script_path = './ml-depth-pro/get_pretrained_models.sh'
        if os.path.exists(script_path):
            result = subprocess.run(['bash', script_path], cwd='./ml-depth-pro', capture_output=True, text=True)
            if result.returncode == 0:
                original_dir = os.getcwd()
                try:
                    os.chdir('./ml-depth-pro')
                    import depth_pro
                    model, transform = depth_pro.create_model_and_transforms()
                    model.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    return (model, transform, device)
                finally:
                    os.chdir(original_dir)
        return (None, None, None)
    except Exception as e:
        return (None, None, None)

def estimate_depth(image_path, model, transform, device):
    """Estimate depth using Depth Pro with proper focal length"""
    if model is None:
        return None, None
    try:
        import depth_pro
        # Use depth_pro's load_rgb to get focal length
        image, _, f_px = depth_pro.load_rgb(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px)  # Pass f_px!
            depth = prediction["depth"]
        # Convert to numpy
        depth_map = depth.squeeze().cpu().numpy()
        # Create colormap
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return depth_map, depth_colormap
    except Exception as e:
        print(f"Error in depth estimation: {e}")
        return None, None
        

# def estimate_depth(image, model, transform, device):
#     """Estimate depth using Depth Pro"""
#     if model is None:
#         return (None, None)
#     try:
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(rgb_image)
#         image_tensor = transform(pil_image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             prediction = model.infer(image_tensor)
#             depth = prediction['depth']
#         depth_map = depth.squeeze().cpu().numpy()
#         depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#         depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
#         return (depth_map, depth_colormap)
#     except Exception as e:
#         return (None, None)

def extract_depth_from_bbox(depth_map, bbox):
    """Extract depth statistics from bounding box region"""
    if depth_map is None:
        return None
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        x1, y1 = (max(0, x1), max(0, y1))
        x2, y2 = (min(w, x2), min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[depth_roi > 0]
        if len(valid_depths) == 0:
            return None
        depth_stats = {'mean_depth': float(np.mean(valid_depths)), 'median_depth': float(np.median(valid_depths)), 'min_depth': float(np.min(valid_depths)), 'max_depth': float(np.max(valid_depths)), 'std_depth': float(np.std(valid_depths)), 'percentile_25': float(np.percentile(valid_depths, 25)), 'percentile_75': float(np.percentile(valid_depths, 75)), 'valid_pixel_ratio': float(len(valid_depths) / depth_roi.size), 'center_depth': None}
        center_x, center_y = ((x1 + x2) // 2, (y1 + y2) // 2)
        if 0 <= center_y < h and 0 <= center_x < w:
            center_depth = depth_map[center_y, center_x]
            if center_depth > 0:
                depth_stats['center_depth'] = float(center_depth)
        return depth_stats
    except Exception as e:
        return None

def setting_up(verbose=True):
    checkpoint_dir = './ml-depth-pro/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        model_path = hf_hub_download(repo_id='apple/DepthPro', filename='depth_pro.pt', local_dir=checkpoint_dir)
    except Exception as e: pass
    if os.path.exists('../ml-depth-pro'): 
        sys.path.insert(0, '../ml-depth-pro')
    try:
        import depth_pro
        DEPTH_PRO_AVAILABLE = True
    except ImportError as e:
        DEPTH_PRO_AVAILABLE = False
    if DEPTH_PRO_AVAILABLE:
        model_path = './ml-depth-pro/checkpoints/depth_pro.pt'
        if not os.path.exists(model_path):
            quick_setup_depth_pro(verbose)
    depth_model, depth_transform, depth_device = initialize_depth_pro(DEPTH_PRO_AVAILABLE, depth_pro, verbose)
    return (DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device)


class SpeedEstimator:
    """
    Minimal speed estimator with optional debug output
    Uses depth directly for Z, simple approximation for X,Y
    """ 
    def __init__(self, n_seconds=3, image_width=1920, debug=False):
        """
        Args:
            n_seconds: Time between frames (MUST match your data extraction!)
            image_width: Image width in pixels
            debug: Enable debug output
        """
        self.frame_time = n_seconds
        self.image_width = image_width
        self.debug = debug
        
        if self.debug:
            print(f"\n{'='*70}")
            print(f"MinimalSpeedEstimator Initialized")
            print(f"{'='*70}")
            print(f"  Frame time: {self.frame_time} seconds")
            print(f"  Image width: {self.image_width} pixels")
            print(f"  FOV assumption: ~90° horizontal")
            print(f"{'='*70}\n")
    
    def calculate_speed_from_positions(self, pos_history, depth_history, 
                                      bbox_heights=None, class_name=None):
        """
        Calculate speed with optional debug output
        
        Args:
            pos_history: List of (x, y) pixel positions
            depth_history: List of depth stats dicts
            bbox_heights: Not used (for compatibility)
            class_name: Not used (for compatibility)
        Returns:
            (speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh)
        """
        if len(pos_history) < 2 or len(depth_history) < 2:
            return (0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        
        current_pos = pos_history[-1]
        prev_pos = pos_history[-2]
        current_depth_stats = depth_history[-1]
        prev_depth_stats = depth_history[-2]
        if not current_depth_stats or not prev_depth_stats:
            return (0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        current_depth = current_depth_stats.get('median_depth')
        prev_depth = prev_depth_stats.get('median_depth')
        if current_depth is None or prev_depth is None or current_depth <= 0 or prev_depth <= 0:
            return (0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # ==================== DEBUG OUTPUT START ====================
        if self.debug:
            print(f"\n{'='*70}")
            print(f"SPEED CALCULATION - MinimalSpeedEstimator")
            print(f"{'='*70}")
            print(f"Frame time: {self.frame_time}s")
            if class_name:
                print(f"Object class: {class_name}")
        # Pixel displacement
        dx_pixels = current_pos[0] - prev_pos[0]
        dy_pixels = current_pos[1] - prev_pos[1]
        
        if self.debug:
            print(f"\n📍 Position Data:")
            print(f"  Previous: ({prev_pos[0]:.1f}, {prev_pos[1]:.1f}) pixels")
            print(f"  Current:  ({current_pos[0]:.1f}, {current_pos[1]:.1f}) pixels")
            print(f"  Change:   ({dx_pixels:+.1f}, {dy_pixels:+.1f}) pixels")
        
        # Depth displacement (already in meters!)
        dz_meters = current_depth - prev_depth
        
        if self.debug:
            print(f"\n📏 Depth Data:")
            print(f"  Previous: {prev_depth:.2f}m")
            print(f"  Current:  {current_depth:.2f}m")
            print(f"  Change:   {dz_meters:+.2f}m")
            print(f"  Status:   {'⬇ APPROACHING' if dz_meters < 0 else '⬆ DEPARTING' if dz_meters > 0 else '↔ STATIONARY'}")
        
        # Approximate pixel-to-meter conversion using depth
        # Assumes ~90° horizontal FOV
        avg_depth = (current_depth + prev_depth) / 2.0
        pixel_to_meter = (2.0 * avg_depth) / self.image_width
        
        if self.debug:
            print(f"\n🔄 Pixel-to-Meter Conversion:")
            print(f"  Average depth: {avg_depth:.2f}m")
            print(f"  Visible width: {2.0 * avg_depth:.2f}m (assumes 90° FOV)")
            print(f"  Ratio: {pixel_to_meter:.6f} m/pixel")
            print(f"  (i.e., 1 pixel = {pixel_to_meter * 100:.2f} cm at this depth)")
        
        dx_meters = dx_pixels * pixel_to_meter
        dy_meters = dy_pixels * pixel_to_meter
        
        if self.debug:
            print(f"\n📐 Lateral Displacement:")
            print(f"  X: {dx_pixels:+.1f} pixels = {dx_meters:+.3f}m")
            print(f"  Y: {dy_pixels:+.1f} pixels = {dy_meters:+.3f}m")
            print(f"  Z: {dz_meters:+.3f}m (from depth)")
        
        # Calculate velocities (m/s)
        velocity_x_ms = dx_meters / self.frame_time
        velocity_y_ms = dy_meters / self.frame_time
        velocity_z_ms = dz_meters / self.frame_time
        
        if self.debug:
            print(f"\n⚡ Velocity Components:")
            print(f"  Vx (lateral):  {velocity_x_ms:+.3f} m/s = {velocity_x_ms * 3.6:+.2f} km/h")
            print(f"  Vy (vertical): {velocity_y_ms:+.3f} m/s = {velocity_y_ms * 3.6:+.2f} km/h")
            print(f"  Vz (depth):    {velocity_z_ms:+.3f} m/s = {velocity_z_ms * 3.6:+.2f} km/h")
        
        # Calculate scalar speed (3D magnitude)
        displacement_3d = math.sqrt(dx_meters**2 + dy_meters**2 + dz_meters**2)
        speed_ms = displacement_3d / self.frame_time
        speed_kmh = speed_ms * 3.6
        
        if self.debug:
            print(f"\n🎯 Total Speed:")
            print(f"  3D displacement: {displacement_3d:.3f}m")
            print(f"  Speed: {speed_ms:.3f} m/s = {speed_kmh:.2f} km/h")
        
        # Velocity vectors
        velocity_vector_ms = (velocity_x_ms, velocity_y_ms, velocity_z_ms)
        velocity_vector_kmh = (velocity_x_ms * 3.6, velocity_y_ms * 3.6, velocity_z_ms * 3.6)
        
        # Sanity checks and clamping
        speed_kmh_original = speed_kmh
        speed_kmh = max(0, min(speed_kmh, 300))
        speed_ms = speed_kmh / 3.6
        
        velocity_vector_ms = tuple(v / 3.6 for v in velocity_vector_kmh)
        
       
        
        # ==================== DEBUG OUTPUT END ====================
        return (speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh)

# class SpeedEstimator:
#     """
#     Class to handle 3D vector speed estimation for vehicles
#     """

#     def __init__(self, fps=10, time_step=2):
#         self.fps = fps
#         self.time_step = time_step
#         self.frame_time = time_step / fps

#     def calculate_pixel_to_meter_ratio(self, depth_stats, bbox_height):
#         """
#         Estimate pixel to meter ratio using depth information
#         """
#         if depth_stats is None or depth_stats.get('median_depth') is None:
#             estimated_depth = 20.0
#             return (1.5 / bbox_height, estimated_depth)
#         median_depth = depth_stats['median_depth']
#         vehicle_height_m = 1.5
#         pixel_to_meter = vehicle_height_m / bbox_height
#         return (pixel_to_meter, median_depth)

#     def calculate_speed_from_positions(self, pos_history, depth_history, bbox_heights):
#         """
#         Calculate relative 3D velocity vector using position history and depth information.
#         """
#         if len(pos_history) < 2:
#             return (0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
#         current_pos = pos_history[-1]
#         prev_pos = pos_history[-2] if len(pos_history) >= 2 else pos_history[-1]
#         dx_pixels = current_pos[0] - prev_pos[0]
#         dy_pixels = current_pos[1] - prev_pos[1]
#         current_depth = depth_history[-1] if len(depth_history) > 0 else None
#         prev_depth = depth_history[-2] if len(depth_history) > 1 else current_depth
#         current_bbox_height = bbox_heights[-1] if bbox_heights else 50
#         pixel_to_meter, depth = self.calculate_pixel_to_meter_ratio(current_depth, current_bbox_height)
#         velocity_x_ms = dx_pixels * pixel_to_meter / self.frame_time
#         velocity_y_ms = -dy_pixels * pixel_to_meter / self.frame_time
#         velocity_z_ms = 0.0
#         if current_depth and prev_depth and current_depth.get('median_depth') and prev_depth.get('median_depth'):
#             depth_change = current_depth['median_depth'] - prev_depth['median_depth']
#             velocity_z_ms = depth_change / self.frame_time
#         pixel_displacement = math.sqrt(dx_pixels * dx_pixels + dy_pixels * dy_pixels)
#         displacement_m = pixel_displacement * pixel_to_meter
#         speed_ms = displacement_m / self.frame_time
#         speed_kmh = speed_ms * 3.6
#         velocity_vector_ms = (velocity_x_ms, velocity_y_ms, velocity_z_ms)
#         velocity_vector_kmh = (velocity_x_ms * 3.6, velocity_y_ms * 3.6, velocity_z_ms * 3.6)
#         speed_kmh = max(0, min(speed_kmh, 300))
#         speed_ms = speed_kmh / 3.6
#         max_component_kmh = 150
#         velocity_vector_kmh = tuple((max(-max_component_kmh, min(max_component_kmh, v)) for v in velocity_vector_kmh))
#         velocity_vector_ms = tuple((v / 3.6 for v in velocity_vector_kmh))
#         return (speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh)

class EnhancedRobustTracker:
    """
    Enhanced tracker with speed estimation capabilities (no lane detection)
    """

    def __init__(self, camera_code, max_history=30, max_disappeared=15):
        self.id_label_map = {}
        self.track_history = {}
        self.disappeared_tracks = {}
        self.vehicle_counter = 0
        self.pedestrian_counter = 0
        self.max_history = max_history
        self.max_disappeared = max_disappeared
        self.camera_code = camera_code
        #self.speed_estimator = SpeedEstimator(fps=CAMERA_CONFIG['fps'], time_step=CAMERA_CONFIG['time_step'])
        self.speed_estimator = SpeedEstimator()

    def get_camera_prefix(self):
        """Map camera folder names to their corresponding prefix codes"""
        camera_prefixes = {'front_left': 'FL', 'left_forward': 'LF', 'right_forward': 'RF', 'right_backward': 'RB', 'rear': 'RE', 'left_backward': 'LB', 'map': 'MP'}
        return camera_prefixes.get(self.camera_code, 'XX')

    def assign_label(self, cls_name, vehicle_classes):
        """Assign a new label for a track with camera-specific prefix"""
        prefix = self.get_camera_prefix()
        if cls_name in vehicle_classes:
            letter_id = chr(ord('A') + self.vehicle_counter % 26)
            label = f'Veh_{prefix}_{letter_id}'
            self.vehicle_counter += 1
        else:
            letter_id = chr(ord('A') + self.pedestrian_counter % 26)
            label = f'Ped_{prefix}_{letter_id}'
            self.pedestrian_counter += 1
        return label

    def calculate_features(self, image, box):
        """Enhanced feature extraction using HSV color space"""
        if image is None:
            return None
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = (max(0, x1), max(0, y1))
        x2, y2 = (min(w, x2), min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            height = roi.shape[0]
            if height < 2:
                return None
            upper_roi = hsv_roi[:height // 2, :]
            lower_roi = hsv_roi[height // 2:, :]
            hist_bins = [8, 8, 8]
            hist_ranges = [180, 256, 256]
            features = {}
            for i, channel in enumerate(['h', 's', 'v']):
                hist = cv2.calcHist([upper_roi], [i], None, [hist_bins[i]], [0, hist_ranges[i]])
                hist = cv2.normalize(hist, hist).flatten()
                features[f'upper_{channel}'] = hist.tolist()
            for i, channel in enumerate(['h', 's', 'v']):
                hist = cv2.calcHist([lower_roi], [i], None, [hist_bins[i]], [0, hist_ranges[i]])
                hist = cv2.normalize(hist, hist).flatten()
                features[f'lower_{channel}'] = hist.tolist()
            pixels = hsv_roi.reshape(-1, 3)
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3
            if len(pixels) < K:
                K = max(1, len(pixels) - 1)
            if K > 0:
                _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                labels_unique, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(labels)
                dominant_colors = []
                for i, (center, percentage) in enumerate(zip(centers, percentages)):
                    dominant_colors.append({'color': center.tolist(), 'percentage': float(percentage)})
                features['dominant_colors'] = dominant_colors
            return features
        except Exception as e:
            return None

    def calculate_similarity_score(self, hist1, hist2):
        """Calculate similarity between two feature sets"""
        if hist1 is None or hist2 is None:
            return 0
        try:
            total_score = 0
            processed_channels = 0
            weight = 1.0 / 6
            for region in ['upper', 'lower']:
                for channel in ['h', 's', 'v']:
                    key = f'{region}_{channel}'
                    if key not in hist1 or key not in hist2:
                        continue
                    try:
                        hist1_array = np.array(hist1[key], dtype=np.float32).reshape(-1, 1)
                        hist2_array = np.array(hist2[key], dtype=np.float32).reshape(-1, 1)
                        if hist1_array.size > 0 and hist2_array.size > 0:
                            score = cv2.compareHist(hist1_array, hist2_array, cv2.HISTCMP_CORREL)
                            total_score += weight * max(0, score)
                            processed_channels += 1
                    except ValueError:
                        continue
            if processed_channels > 0:
                total_score = total_score * (6 / processed_channels)
            if 'dominant_colors' in hist1 and 'dominant_colors' in hist2:
                color_sim = self.compare_dominant_colors(hist1['dominant_colors'], hist2['dominant_colors'])
                total_score = 0.7 * total_score + 0.3 * color_sim
            return total_score
        except Exception as e:
            return 0

    def compare_dominant_colors(self, colors1, colors2):
        """Compare dominant color sets"""
        if not colors1 or not colors2:
            return 0
        total_sim = 0
        for c1 in colors1:
            color1 = np.array(c1['color'])
            pct1 = c1['percentage']
            max_color_sim = 0
            for c2 in colors2:
                color2 = np.array(c2['color'])
                pct2 = c2['percentage']
                color_dist = np.exp(-np.sum(np.abs(color1 - color2)) / 255.0)
                sim = color_dist * min(pct1, pct2)
                max_color_sim = max(max_color_sim, sim)
            total_sim += max_color_sim
        return total_sim / len(colors1)

    def predict_next_position(self, positions):
        """Predict next position based on velocity history"""
        if len(positions) < 2:
            return None
        recent_positions = list(positions)[-5:]
        if len(recent_positions) < 2:
            return recent_positions[-1]
        velocities = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i - 1][0]
            dy = recent_positions[i][1] - recent_positions[i - 1][1]
            velocities.append((dx, dy))
        avg_velocity = np.mean(velocities, axis=0)
        last_pos = recent_positions[-1]
        predicted_x = last_pos[0] + avg_velocity[0]
        predicted_y = last_pos[1] + avg_velocity[1]
        return (predicted_x, predicted_y)

    def update(self, image, current_detections, vehicle_classes):
        """Update tracking for current frame with speed estimation"""
        current_track_ids = set()
        for track_id, (box, cls_name, depth_stats) in current_detections.items():
            current_track_ids.add(track_id)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            features = self.calculate_features(image, box)
            bbox_height = box[3] - box[1]
            if track_id not in self.track_history:
                self.track_history[track_id] = {'positions': deque(maxlen=self.max_history), 'features': features, 'class': cls_name, 'depth_history': deque(maxlen=self.max_history), 'bbox_heights': deque(maxlen=self.max_history), 'speeds': deque(maxlen=10)}
                best_match_id = None
                best_match_score = 0
                for lost_id, lost_data in list(self.disappeared_tracks.items()):
                    if lost_data['class'] != cls_name:
                        continue
                    predicted_pos = self.predict_next_position(lost_data['positions'])
                    if predicted_pos is None:
                        continue
                    dist = np.sqrt((center[0] - predicted_pos[0]) ** 2 + (center[1] - predicted_pos[1]) ** 2)
                    pos_score = np.exp(-dist / 100)
                    appear_score = self.calculate_similarity_score(features, lost_data['features'])
                    if cls_name == 'person':
                        total_score = 0.7 * appear_score + 0.3 * pos_score
                    else:
                        total_score = 0.6 * appear_score + 0.4 * pos_score
                    if total_score > best_match_score and total_score > 0.4:
                        best_match_score = total_score
                        best_match_id = lost_id
                if best_match_id is not None:
                    self.id_label_map[track_id] = self.disappeared_tracks[best_match_id]['label']
                    self.track_history[track_id]['positions'] = self.disappeared_tracks[best_match_id]['positions']
                    self.track_history[track_id]['depth_history'] = self.disappeared_tracks[best_match_id].get('depth_history', deque(maxlen=self.max_history))
                    self.track_history[track_id]['bbox_heights'] = self.disappeared_tracks[best_match_id].get('bbox_heights', deque(maxlen=self.max_history))
                    self.track_history[track_id]['speeds'] = self.disappeared_tracks[best_match_id].get('speeds', deque(maxlen=10))
                    del self.disappeared_tracks[best_match_id]
                else:
                    self.id_label_map[track_id] = self.assign_label(cls_name, vehicle_classes)
            self.track_history[track_id]['positions'].append(center)
            self.track_history[track_id]['features'] = features
            self.track_history[track_id]['depth_history'].append(depth_stats)
            self.track_history[track_id]['bbox_heights'].append(bbox_height)
            track_data = self.track_history[track_id]
            if len(track_data['positions']) >= 2:
                speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh = self.speed_estimator.calculate_speed_from_positions(track_data['positions'], track_data['depth_history'], class_name=track_data['class'])
                # speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh = self.speed_estimator.calculate_speed_from_positions(track_data['positions'], track_data['depth_history'], track_data['bbox_heights'])
                
                if len(track_data['speeds']) > 0:
                    prev_speed = track_data['speeds'][-1]['speed_kmh']
                    smoothed_speed = 0.7 * prev_speed + 0.3 * speed_kmh
                    prev_vector = track_data['speeds'][-1]['velocity_vector_kmh']
                    smoothed_vector_kmh = tuple((0.7 * prev_vector[i] + 0.3 * velocity_vector_kmh[i] for i in range(3)))
                    smoothed_vector_ms = tuple((v / 3.6 for v in smoothed_vector_kmh))
                else:
                    smoothed_speed = speed_kmh
                    smoothed_vector_kmh = velocity_vector_kmh
                    smoothed_vector_ms = velocity_vector_ms
                speed_data = {'speed_ms': speed_ms, 'speed_kmh': smoothed_speed, 'velocity_vector_ms': smoothed_vector_ms, 'velocity_vector_kmh': smoothed_vector_kmh, 'vector_dimension': '3D', 'coordinate_system': 'ego_vehicle_relative', 'coordinate_description': {'x': 'lateral (positive=right, negative=left)', 'y': 'longitudinal (positive=away/forward, negative=toward/backward)', 'z': 'vertical (positive=up, negative=down)'}, 'direction_examples': {'approaching_vehicle': 'negative Y component', 'departing_vehicle': 'positive Y component', 'overtaking_right': 'positive X component', 'overtaking_left': 'negative X component'}, 'relative_to_ego': True, 'timestamp': datetime.now().isoformat()}
                track_data['speeds'].append(speed_data)
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                if track_id not in self.disappeared_tracks:
                    self.disappeared_tracks[track_id] = {'count': 0, 'label': self.id_label_map[track_id], 'positions': self.track_history[track_id]['positions'], 'features': self.track_history[track_id]['features'], 'class': self.track_history[track_id]['class'], 'depth_history': self.track_history[track_id].get('depth_history', deque()), 'bbox_heights': self.track_history[track_id].get('bbox_heights', deque()), 'speeds': self.track_history[track_id].get('speeds', deque())}
                self.disappeared_tracks[track_id]['count'] += 1
                if self.disappeared_tracks[track_id]['count'] > self.max_disappeared:
                    del self.disappeared_tracks[track_id]
                    del self.track_history[track_id]
                    del self.id_label_map[track_id]

def _should_load_frame(run_dict, json_exists):
    """
    Load frame only if it's actually used:
      - overwrite == True  (to recreate/replace JSON; need dims)
      - OR detection == True (RF-DETR needs pixels)
      - OR depth == True (DepthPro needs pixels)
    If all three are False:
      - require an existing JSON; otherwise we cannot proceed without pixels.
    """
    overwrite = run_dict.get('overwrite', False)
    detection = run_dict.get('detection', True)
    depth = run_dict.get('depth', True)
    if overwrite or detection or depth:
        return True
    return False

def process_frame_with_depth_and_speed(frame_path, model, vehicle_classes, target_classes, camera_name, output_base_dir, DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device, run_dict=None):
    if run_dict is None:
        run_dict = {'detection': True, 'depth': True, 'speed': True, 'overwrite': False}
    output_dir = os.path.join(output_base_dir, f'{camera_name}_Segmented')
    json_dir = os.path.join(output_base_dir, f'{camera_name}_Annotations')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, os.path.splitext(os.path.basename(frame_path))[0] + '.json')
    overwrite = run_dict.get('overwrite', False)
    json_exists = not overwrite and os.path.exists(json_path)
    image = None
    if json_exists:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    else:
        if not _should_load_frame(run_dict, json_exists=False):
            return
        image = cv2.imread(frame_path)
        if image is None:
            return
        json_data = {'info': {'description': 'L2D Dataset Detection Results', 'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 'images': [{'id': 1, 'file_name': os.path.basename(frame_path), 'width': image.shape[1], 'height': image.shape[0], 'camera': camera_name, 'has_depth': False}], 'annotations': [], 'categories': [{'id': 1, 'name': 'vehicle', 'supercategory': 'traffic'}, {'id': 2, 'name': 'pedestrian', 'supercategory': 'traffic'}], 'meta': {'id_label_map': {}}}
    if 'meta' not in json_data:
        json_data['meta'] = {}
    if 'id_label_map' not in json_data['meta']:
        json_data['meta']['id_label_map'] = {}
    existing_annots_by_track = {a['track_id']: a for a in json_data.get('annotations', [])}
    if image is None and _should_load_frame(run_dict, json_exists=True):
        image = cv2.imread(frame_path)
        if image is None:
            return
    depth_map, depth_colormap = (None, None)
    if run_dict.get('depth', True) and DEPTH_PRO_AVAILABLE and (depth_model is not None):
        if image is None: pass
        else:
            depth_map, depth_colormap = estimate_depth(frame_path, depth_model, depth_transform, depth_device)
            depth_dir = os.path.join(output_base_dir, f'{camera_name}_DepthMaps')
            depth_colormap_dir = os.path.join(output_base_dir, f'{camera_name}_DepthColormaps')
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(depth_colormap_dir, exist_ok=True)
            json_data['images'][0]['has_depth'] = True
    current_detections = {}
    if run_dict.get('detection', True):
        if model is None: pass
        elif image is None: pass
        else:
            try:
                detection_results = rfdetr_detect_objects(model, image, confidence_threshold=0.4)
                mapped_detections = map_rfdetr_to_target_classes(detection_results, target_classes)
                for track_id, (bbox, cls_name) in mapped_detections.items():
                    current_detections[track_id] = (bbox, cls_name, None)
                last_detections_cache[camera_name] = current_detections.copy()
            except Exception as e: pass
    else:
        try:
            with open(json_path, 'r') as f:
                saved = json.load(f)
            for ann in saved.get('annotations', []):
                x, y, w, h = ann['bbox']
                bbox = [x, y, x + w, y + h]
                cls = ann['attributes'].get('class', 'unknown')
                depth_stats = ann['attributes'].get('depth_stats', None)
                current_detections[ann['track_id']] = (bbox, cls, depth_stats)
        except Exception as e:
            return
    if depth_map is not None:
        for tid in current_detections:
            bbox, cls = (current_detections[tid][0], current_detections[tid][1])
            current_detections[tid] = (bbox, cls, extract_depth_from_bbox(depth_map, bbox))
    if run_dict.get('speed', True):
        if camera_name not in camera_trackers:
            camera_trackers[camera_name] = EnhancedRobustTracker(camera_name)
        tracker = camera_trackers[camera_name]
        if not run_dict.get('detection', True):
            saved_map = json_data.get('meta', {}).get('id_label_map', {})
            if isinstance(saved_map, dict):
                tracker.id_label_map.update(saved_map)
            for tid in current_detections.keys():
                if tid not in tracker.id_label_map:
                    tracker.id_label_map[tid] = tid
        if current_detections:
            tracker.update(image, current_detections, vehicle_classes)
    else:
        tracker = None
    annotation_id = max([a['id'] for a in json_data.get('annotations', [])], default=0) + 1
    for track_id, (box, cls_name, depth_stats) in current_detections.items():
        x1, y1, x2, y2 = box
        if not run_dict.get('detection', True):
            label = track_id
        elif camera_name in camera_trackers and track_id in camera_trackers[camera_name].id_label_map:
            label = camera_trackers[camera_name].id_label_map[track_id]
        else:
            label = f'ID{track_id}'
        speed_info = None
        if run_dict.get('speed', True) and tracker:
            track_data = tracker.track_history.get(track_id, {})
            if 'speeds' in track_data and len(track_data['speeds']) > 0:
                speed_info = track_data['speeds'][-1]
        if image is not None:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if label in existing_annots_by_track:
            if run_dict.get('detection', True):
                existing_annots_by_track[label]['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                existing_annots_by_track[label]['attributes']['class'] = cls_name
            existing_annots_by_track[label].setdefault('attributes', {})
            if run_dict.get('depth', True):
                existing_annots_by_track[label]['attributes']['depth_stats'] = depth_stats
            if run_dict.get('speed', True):
                existing_annots_by_track[label]['attributes']['speed_info'] = speed_info
        elif run_dict.get('detection', True):
            json_data['annotations'].append({'id': annotation_id, 'image_id': 1, 'category_id': 1 if cls_name in vehicle_classes else 2, 'track_id': label, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'attributes': {'class': cls_name, 'depth_stats': depth_stats if run_dict.get('depth', True) else None, 'speed_info': speed_info if run_dict.get('speed', True) else None}})
            annotation_id += 1
        else:
            pass
    if run_dict.get('speed', True) and tracker:
        json_data['meta']['id_label_map'] = tracker.id_label_map
    if VERBOSE_MODE and image is not None:
        try:
            cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_path)), image)
        except Exception:
            pass
    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(json_data), f, indent=2)
    return {'frame': os.path.basename(frame_path), 'detections': len(json_data['annotations']), 'path': os.path.join(output_dir, os.path.basename(frame_path)) if image is not None else None, 'json': json_path, 'has_depth': depth_map is not None}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple((convert_numpy_types(item) for item in obj))
    else:
        return obj

def process_episode_frames_with_depth_and_speed(DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device, episode_num, camera_views, input_base_dir, output_base_dir, model, vehicle_classes, target_classes, run_dict=None, depth_available=False):
    if run_dict is None:
        run_dict = {'detection': True, 'depth': True, 'speed': True, 'overwrite': False}
    results = {}
    episode_dir = os.path.join(input_base_dir, f'Episode{episode_num:06d}')
    pixel_dependent = run_dict.get('overwrite', False) or run_dict.get('detection', True) or run_dict.get('depth', True)
    if pixel_dependent and (not os.path.exists(episode_dir)):
        return results
    global camera_trackers, last_detections_cache
    if not isinstance(camera_trackers, dict):
        camera_trackers = {}
    if not isinstance(last_detections_cache, dict):
        last_detections_cache = {}
    camera_trackers.clear()
    last_detections_cache.clear()
    pixel_dependent = run_dict.get('overwrite', False) or run_dict.get('detection', True) or run_dict.get('depth', True)
    for camera in camera_views:
        camera_frames_dir = os.path.join(episode_dir, camera)
        camera_name = camera.split('.')[-1]
        results[camera_name] = []
        if pixel_dependent:
            if not os.path.exists(camera_frames_dir):
                continue
            frame_files = sorted([f for f in os.listdir(camera_frames_dir) if f.lower().endswith(('.jpg', '.png'))])
            if not frame_files:
                continue
            for frame_file in frame_files:
                frame_path = os.path.join(camera_frames_dir, frame_file)
                frame_result = process_frame_with_depth_and_speed(frame_path, model, vehicle_classes, target_classes, camera_name, output_base_dir, DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device, run_dict)
                if frame_result:
                    results[camera_name].append(frame_result)
        else:
            ann_dir = os.path.join(output_base_dir, f'{camera_name}_Annotations')
            if not os.path.exists(ann_dir):
                continue
            json_files = sorted([f for f in os.listdir(ann_dir) if f.lower().endswith('.json')])
            if not json_files:
                continue
            for jf in json_files:
                base = os.path.splitext(jf)[0]
                fake_frame_path = os.path.join(ann_dir, base + '.jpg')
                frame_result = process_frame_with_depth_and_speed(fake_frame_path, model, vehicle_classes, target_classes, camera_name, output_base_dir, DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device, run_dict)
                if frame_result:
                    results[camera_name].append(frame_result)
    return results

def initialize_rfdetr_model():
    """Initialize RF-DETR model"""
    try:
        model = RFDETRBase()
        return model
    except Exception as e:
        return None

def rfdetr_detect_objects(model, image, confidence_threshold=0.4):
    """
    Run RF-DETR detection on image and return results in YOLO-compatible format
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        detections = model.predict(pil_image, threshold=confidence_threshold)
        detection_results = {}
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
                pseudo_track_id = f'det_{i}'
                x1, y1, x2, y2 = map(int, bbox)
                detection_results[pseudo_track_id] = {'bbox': [x1, y1, x2, y2], 'class_name': class_name, 'confidence': float(confidence), 'class_id': int(class_id)}
        return detection_results
    except Exception as e:
        return {}

def map_rfdetr_to_target_classes(detection_results, target_classes):
    """
    Filter RF-DETR detections to only include target classes and assign track IDs
    """
    filtered_detections = {}
    track_id_counter = 1
    for det_key, detection in detection_results.items():
        class_name = detection['class_name']
        class_mapping = {'bicycle': 'bicycle', 'car': 'car', 'motorcycle': 'motorcycle', 'bus': 'bus', 'train': 'train', 'truck': 'truck', 'boat': 'boat', 'person': 'person'}
        mapped_class = class_mapping.get(class_name, class_name)
        if mapped_class in target_classes:
            track_id = track_id_counter
            filtered_detections[track_id] = (detection['bbox'], mapped_class)
            track_id_counter += 1
    return filtered_detections

def process_frames(min_ep, max_ep=-1, input_base_dir='../data/raw/L2D/frames', output_base_dir='../data/processed_frames/L2D', cameras_on=None, run_dict=None, verbose=False):
    global VERBOSE_MODE
    VERBOSE_MODE = bool(verbose)

    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

    if run_dict is None:
        run_dict = {'detection': True, 'depth': True, 'speed': True, 'overwrite': False}
    if run_dict.get('depth', True):
        DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device = setting_up(verbose=verbose)
    else:
        DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device = (False, None, None, torch.device('cpu'))
    if run_dict.get('detection', True):
        model = initialize_rfdetr_model()
        if model is None:
            raise RuntimeError('Failed to initialize RF-DETR model.')
        if hasattr(depth_device, 'type') and depth_device.type == 'cuda':
            try:
                model.optimize_for_inference()
            except:
                pass
    else:
        model = None
    if cameras_on is None:
        cameras_on = ['observation.images.front_left', 'observation.images.left_forward', 'observation.images.right_forward', 'observation.images.right_backward', 'observation.images.rear', 'observation.images.left_backward']
    vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat']
    target_classes = vehicle_classes + ['person']
    all_results = {}
    for ep in tqdm(iterable):
        episode_results = process_episode_frames_with_depth_and_speed(DEPTH_PRO_AVAILABLE, depth_model, depth_transform, depth_device, episode_num=ep, camera_views=cameras_on, input_base_dir=input_base_dir, output_base_dir=output_base_dir + f'/Episode{ep:06d}', model=model, vehicle_classes=vehicle_classes, target_classes=target_classes, run_dict=run_dict)
        all_results[ep] = episode_results
