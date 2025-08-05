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
# RF-DETR imports
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

camera_trackers = {}
last_detections_cache = {}
CAMERA_CONFIG = {
    "fps": 10,  # Assume 10 FPS for L2D dataset (adjust based on actual FPS)
    "time_step": 2,  # From processing_images_view.py - frames are extracted with time_step=2
}

def quick_setup_depth_pro(verbose):
    """Quick setup function to download Depth Pro model"""
    if not os.path.exists("./ml-depth-pro"):
        if verbose:print("❌ ml-depth-pro directory not found!")
        if verbose:print("Please clone it first: git clone https://github.com/apple/ml-depth-pro.git")
        return False
    
    checkpoint_dir = "./ml-depth-pro/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if verbose:print("📥 Downloading Depth Pro model from Hugging Face...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download the model file
        model_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False
        )
        
        if verbose:print(f"✅ Model downloaded to: {model_path}")
        return True
        
    except Exception as e:
        if verbose:print(f"❌ Download failed: {e}")
        if verbose:print("\n🔧 Manual download instructions:")
        if verbose:print("1. Go to https://huggingface.co/apple/DepthPro")
        if verbose:print("2. Download 'depth_pro.pt' file")
        if verbose:print(f"3. Place it in: {checkpoint_dir}/")
        if verbose:print("4. Re-run this cell")
        return False
    
def initialize_depth_pro(DEPTH_PRO_AVAILABLE,depth_pro,verbose):
    """Initialize the Depth Pro model"""
    if not DEPTH_PRO_AVAILABLE:
        return None, None, None
    
    try:
        if verbose:print("🔍 Initializing Depth Pro...")
        
        # Check and download model if needed
        if not download_depth_pro_model(verbose):
            return None, None, None
        
        # Change to ml-depth-pro directory temporarily
        original_dir = os.getcwd()
        depth_pro_dir = "./ml-depth-pro"
        
        try:
            os.chdir(depth_pro_dir)
            model, transform = depth_pro.create_model_and_transforms()
            model.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            if verbose:print(f"✅ Depth Pro loaded on {device}")
            return model, transform, device
            
        finally:
            os.chdir(original_dir)
    
    except FileNotFoundError as e:
        if verbose:print(f"❌ Model file not found: {e}")
        if verbose:print("🔧 Trying alternative download method...")
        return try_alternative_download()
    
    except Exception as e:
        if verbose:print(f"❌ Error initializing Depth Pro: {e}")
        return None, None, None
    
def download_depth_pro_model(verbose):
    """Download Depth Pro model if not exists"""
    checkpoint_dir = "./ml-depth-pro/checkpoints"
    model_path = os.path.join(checkpoint_dir, "depth_pro.pt")
    
    if os.path.exists(model_path):
        if verbose:print("✅ Depth Pro model already exists")
        return True
    
    if verbose:print("📥 Downloading Depth Pro model...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download from Hugging Face
        downloaded_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False
        )
        
        if verbose:print("✅ Depth Pro model downloaded successfully")
        return True
        
    except Exception as e:
        if verbose:print(f"❌ Failed to download model: {e}")
        if verbose:print("🔧 Manual download instructions:")
        if verbose:print("1. Go to: https://huggingface.co/apple/DepthPro")
        if verbose:print("2. Download depth_pro.pt")
        if verbose:print(f"3. Place it in: {checkpoint_dir}/")
        return False
    
def try_alternative_download(verbose):
    """Try alternative method to get Depth Pro working"""
    try:
        import subprocess
        
        # Try to run the download script if it exists
        script_path = "./ml-depth-pro/get_pretrained_models.sh"
        if os.path.exists(script_path):
            if verbose:print("🔄 Running download script...")
            result = subprocess.run(["bash", script_path], 
                                  cwd="./ml-depth-pro", 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:print("✅ Download script completed")
                # Try initialization again
                original_dir = os.getcwd()
                try:
                    os.chdir("./ml-depth-pro")
                    model, transform = depth_pro.create_model_and_transforms()
                    model.eval()
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    if verbose:print(f"✅ Depth Pro loaded on {device}")
                    return model, transform, device
                finally:
                    os.chdir(original_dir)
            else:
                if verbose:print(f"❌ Download script failed: {result.stderr}")
        
        if verbose:print("🔧 Manual setup required:")
        if verbose:print("1. cd ml-depth-pro")
        if verbose:print("2. bash get_pretrained_models.sh")
        if verbose:print("3. Or download manually from Hugging Face")
        
        return None, None, None
        
    except Exception as e:
        if verbose:print(f"❌ Alternative download failed: {e}")
        return None, None, None
    
def estimate_depth(image, model, transform, device):
    """Estimate depth using Depth Pro"""
    if model is None:
        return None, None
    
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transforms and run inference
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model.infer(image_tensor)
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

def extract_depth_from_bbox(depth_map, bbox):
    """Extract depth statistics from bounding box region"""
    if depth_map is None:
        return None
        
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI and get valid depths
        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[depth_roi > 0]
        
        if len(valid_depths) == 0:
            return None
        
        # Calculate statistics
        depth_stats = {
            'mean_depth': float(np.mean(valid_depths)),
            'median_depth': float(np.median(valid_depths)),
            'min_depth': float(np.min(valid_depths)),
            'max_depth': float(np.max(valid_depths)),
            'std_depth': float(np.std(valid_depths)),
            'percentile_25': float(np.percentile(valid_depths, 25)),
            'percentile_75': float(np.percentile(valid_depths, 75)),
            'valid_pixel_ratio': float(len(valid_depths) / depth_roi.size),
            'center_depth': None
        }
        
        # Get center point depth
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= center_y < h and 0 <= center_x < w:
            center_depth = depth_map[center_y, center_x]
            if center_depth > 0:
                depth_stats['center_depth'] = float(center_depth)
        
        return depth_stats
        
    except Exception as e:
        print(f"Error extracting depth from bbox: {e}")
        return None

# -------------------------
# SETUP FUNCTION
# -------------------------
def setting_up(verbose=True):
    checkpoint_dir = "./ml-depth-pro/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if verbose: print("Downloading Depth Pro model manually...")
    try:
        model_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False
        )
        if verbose: print(f"✅ Model downloaded successfully to: {model_path}")
    except Exception as e:
        if verbose: print(f"❌ Download failed: {e}")
        if verbose: print("Please download manually from: https://huggingface.co/apple/DepthPro")

    if os.path.exists("../ml-depth-pro"):
        sys.path.insert(0, "../ml-depth-pro")

    try:
        import depth_pro
        DEPTH_PRO_AVAILABLE = True
        if verbose: print("✅ Depth Pro available")
    except ImportError as e:
        DEPTH_PRO_AVAILABLE = False
        if verbose: print(f"⚠️ Depth Pro not available: {e}")
        if verbose: print("Will run RF-DETR-only mode")

    if verbose: print(f"PyTorch: {torch.__version__}")
    if verbose: print(f"CUDA available: {torch.cuda.is_available()}")

    if DEPTH_PRO_AVAILABLE:
        model_path = "./ml-depth-pro/checkpoints/depth_pro.pt"
        if not os.path.exists(model_path):
            if verbose: print("⚠️ Depth Pro model not found. Attempting download...")
            quick_setup_depth_pro(verbose)
        else:
            if verbose: print("✅ Depth Pro model found")

    depth_model, depth_transform, depth_device = initialize_depth_pro(DEPTH_PRO_AVAILABLE,depth_pro,verbose)
    return DEPTH_PRO_AVAILABLE,depth_model,depth_transform,depth_device,

class SpeedEstimator:
    """
    Class to handle 3D vector speed estimation for vehicles
    """
    def __init__(self, fps=10, time_step=2):
        self.fps = fps
        self.time_step = time_step
        # Actual time between consecutive frames considering time_step
        self.frame_time = time_step / fps  # e.g., 2/10 = 0.2 seconds between frames
        
    def calculate_pixel_to_meter_ratio(self, depth_stats, bbox_height):
        """
        Estimate pixel to meter ratio using depth information
        """
        if depth_stats is None or depth_stats.get('median_depth') is None:
            # Fallback estimation based on typical vehicle heights
            # Assume average car height is 1.5m
            estimated_depth = 20.0  # meters (fallback)
            return 1.5 / bbox_height, estimated_depth
        
        median_depth = depth_stats['median_depth']
        # Assume average vehicle height is 1.5 meters
        vehicle_height_m = 1.5
        pixel_to_meter = vehicle_height_m / bbox_height
        
        return pixel_to_meter, median_depth
    
    def calculate_speed_from_positions(self, pos_history, depth_history, bbox_heights):
        """
        Calculate relative 3D velocity vector using position history and depth information.
        
        Note: This calculates velocity relative to the ego vehicle, not absolute velocity,
        since the camera is mounted on the moving ego vehicle.
        
        3D Coordinate System (Ego Vehicle Relative):
        - X: lateral (positive = right, negative = left)
        - Y: longitudinal (positive = forward/away, negative = backward/toward)  
        - Z: vertical (positive = up, negative = down)
        
        Direction Examples:
        - Vehicle moving away from ego: positive Y
        - Vehicle approaching ego: negative Y
        - Vehicle overtaking on right: positive X
        - Vehicle overtaking on left: negative X
        
        Returns both scalar and 3D vector speeds:
        - Scalar: magnitude of movement
        - Vector: (x,y,z) components relative to ego vehicle coordinate system
        """
        if len(pos_history) < 2:
            return 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)  # speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh
        
        # Get last two positions
        current_pos = pos_history[-1]
        prev_pos = pos_history[-2] if len(pos_history) >= 2 else pos_history[-1]
        
        # Calculate pixel displacement components
        dx_pixels = current_pos[0] - prev_pos[0]  # Horizontal displacement in image
        dy_pixels = current_pos[1] - prev_pos[1]  # Vertical displacement in image
        
        # Get depth information for scale and Z-component
        current_depth = depth_history[-1] if len(depth_history) > 0 else None
        prev_depth = depth_history[-2] if len(depth_history) > 1 else current_depth
        current_bbox_height = bbox_heights[-1] if bbox_heights else 50  # fallback
        
        # Convert pixel displacement to meters
        pixel_to_meter, depth = self.calculate_pixel_to_meter_ratio(current_depth, current_bbox_height)
        
        # Calculate 3D velocity components relative to ego vehicle
        # Map image coordinates to ego vehicle coordinate system with intuitive signs
        
        # X-component: Lateral movement (image horizontal → ego lateral)
        # Positive dx_pixels = moving right in image = positive X (right relative to ego)
        velocity_x_ms = (dx_pixels * pixel_to_meter) / self.frame_time
        
        # Y-component: Longitudinal movement (image vertical → ego longitudinal)
        # FIXED: Positive dy_pixels = moving down in image = approaching ego = negative Y
        #        Negative dy_pixels = moving up in image = moving away = positive Y
        # This makes intuitive sense: approaching vehicle = negative Y, departing vehicle = positive Y
        velocity_y_ms = (-dy_pixels * pixel_to_meter) / self.frame_time
        
        # Z-component: Vertical movement (estimated from depth changes)
        velocity_z_ms = 0.0
        if current_depth and prev_depth and current_depth.get('median_depth') and prev_depth.get('median_depth'):
            depth_change = current_depth['median_depth'] - prev_depth['median_depth']
            # Positive depth change = moving away = positive Z relative to ego
            # Negative depth change = moving closer = negative Z relative to ego
            velocity_z_ms = depth_change / self.frame_time
        
        # Calculate scalar speed (magnitude of 2D movement in image plane)
        pixel_displacement = math.sqrt(dx_pixels*dx_pixels + dy_pixels*dy_pixels)
        displacement_m = pixel_displacement * pixel_to_meter
        speed_ms = displacement_m / self.frame_time
        speed_kmh = speed_ms * 3.6
        
        # Create velocity vectors
        velocity_vector_ms = (velocity_x_ms, velocity_y_ms, velocity_z_ms)
        velocity_vector_kmh = (velocity_x_ms * 3.6, velocity_y_ms * 3.6, velocity_z_ms * 3.6)
        
        # Apply reasonable limits for scalar speed
        speed_kmh = max(0, min(speed_kmh, 300))  # 0-300 km/h relative speed range
        speed_ms = speed_kmh / 3.6
        
        # Apply reasonable limits for vector components (±150 km/h per component)
        max_component_kmh = 150
        velocity_vector_kmh = tuple(
            max(-max_component_kmh, min(max_component_kmh, v)) for v in velocity_vector_kmh
        )
        velocity_vector_ms = tuple(v / 3.6 for v in velocity_vector_kmh)
        
        return speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh

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
        self.speed_estimator = SpeedEstimator(fps=CAMERA_CONFIG["fps"], time_step=CAMERA_CONFIG["time_step"])
        
    def get_camera_prefix(self):
        """Map camera folder names to their corresponding prefix codes"""
        camera_prefixes = {
            'front_left': 'FL',
            'left_forward': 'LF', 
            'right_forward': 'RF',
            'right_backward': 'RB',
            'rear': 'RE',
            'left_backward': 'LB',
            'map': 'MP'
        }
        return camera_prefixes.get(self.camera_code, 'XX')
    
    def assign_label(self, cls_name, vehicle_classes):
        """Assign a new label for a track with camera-specific prefix"""
        prefix = self.get_camera_prefix()
        
        if cls_name in vehicle_classes:
            letter_id = chr(ord('A') + self.vehicle_counter % 26)
            label = f"Veh_{prefix}_{letter_id}"
            self.vehicle_counter += 1
        else:
            letter_id = chr(ord('A') + self.pedestrian_counter % 26)
            label = f"Ped_{prefix}_{letter_id}"
            self.pedestrian_counter += 1
        return label

    def calculate_features(self, image, box):
        """Enhanced feature extraction using HSV color space"""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

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

            upper_roi = hsv_roi[:height//2, :]
            lower_roi = hsv_roi[height//2:, :]

            hist_bins = [8, 8, 8]
            hist_ranges = [180, 256, 256]
            features = {}

            # Upper and lower region histograms
            for i, channel in enumerate(['h', 's', 'v']):
                hist = cv2.calcHist([upper_roi], [i], None, [hist_bins[i]], [0, hist_ranges[i]])
                hist = cv2.normalize(hist, hist).flatten()
                features[f'upper_{channel}'] = hist.tolist()

            for i, channel in enumerate(['h', 's', 'v']):
                hist = cv2.calcHist([lower_roi], [i], None, [hist_bins[i]], [0, hist_ranges[i]])
                hist = cv2.normalize(hist, hist).flatten()
                features[f'lower_{channel}'] = hist.tolist()

            # Dominant colors using K-means
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
                    dominant_colors.append({
                        'color': center.tolist(),
                        'percentage': float(percentage)
                    })
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

            # Add dominant color comparison
            if 'dominant_colors' in hist1 and 'dominant_colors' in hist2:
                color_sim = self.compare_dominant_colors(
                    hist1['dominant_colors'],
                    hist2['dominant_colors']
                )
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
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
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
            center = ((box[0] + box[2])/2, (box[1] + box[3])/2)
            features = self.calculate_features(image, box)
            bbox_height = box[3] - box[1]
            
            if track_id not in self.track_history:
                # Initialize new track
                self.track_history[track_id] = {
                    'positions': deque(maxlen=self.max_history),
                    'features': features,
                    'class': cls_name,
                    'depth_history': deque(maxlen=self.max_history),
                    'bbox_heights': deque(maxlen=self.max_history),
                    'speeds': deque(maxlen=10),  # Keep last 10 speed measurements
                }
                
                # Try to match with disappeared tracks
                best_match_id = None
                best_match_score = 0
                
                for lost_id, lost_data in list(self.disappeared_tracks.items()):
                    if lost_data['class'] != cls_name:
                        continue
                        
                    predicted_pos = self.predict_next_position(lost_data['positions'])
                    if predicted_pos is None:
                        continue
                    
                    dist = np.sqrt((center[0] - predicted_pos[0])**2 + 
                                 (center[1] - predicted_pos[1])**2)
                    pos_score = np.exp(-dist / 100)
                    
                    appear_score = self.calculate_similarity_score(features, 
                                                                lost_data['features'])
                    
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
            
            # Update position and depth history
            self.track_history[track_id]['positions'].append(center)
            self.track_history[track_id]['features'] = features
            self.track_history[track_id]['depth_history'].append(depth_stats)
            self.track_history[track_id]['bbox_heights'].append(bbox_height)
            
            # Calculate speed if we have enough history
            track_data = self.track_history[track_id]
            if len(track_data['positions']) >= 2:
                speed_ms, speed_kmh, velocity_vector_ms, velocity_vector_kmh = self.speed_estimator.calculate_speed_from_positions(
                    track_data['positions'],
                    track_data['depth_history'],
                    track_data['bbox_heights']
                )
                
                # Smooth speed estimates (both scalar and vector)
                if len(track_data['speeds']) > 0:
                    # Apply simple smoothing to scalar speed
                    prev_speed = track_data['speeds'][-1]['speed_kmh']
                    smoothed_speed = 0.7 * prev_speed + 0.3 * speed_kmh
                    
                    # Apply smoothing to vector components
                    prev_vector = track_data['speeds'][-1]['velocity_vector_kmh']
                    smoothed_vector_kmh = tuple(
                        0.7 * prev_vector[i] + 0.3 * velocity_vector_kmh[i] 
                        for i in range(3)
                    )
                    smoothed_vector_ms = tuple(v / 3.6 for v in smoothed_vector_kmh)
                else:
                    smoothed_speed = speed_kmh
                    smoothed_vector_kmh = velocity_vector_kmh
                    smoothed_vector_ms = velocity_vector_ms
                
                speed_data = {
                    'speed_ms': speed_ms,
                    'speed_kmh': smoothed_speed,
                    'velocity_vector_ms': smoothed_vector_ms,  # (x, y, z) in m/s
                    'velocity_vector_kmh': smoothed_vector_kmh,  # (x, y, z) in km/h
                    'vector_dimension': '3D',
                    'coordinate_system': 'ego_vehicle_relative',
                    'coordinate_description': {
                        'x': 'lateral (positive=right, negative=left)',
                        'y': 'longitudinal (positive=away/forward, negative=toward/backward)',
                        'z': 'vertical (positive=up, negative=down)'
                    },
                    'direction_examples': {
                        'approaching_vehicle': 'negative Y component',
                        'departing_vehicle': 'positive Y component',
                        'overtaking_right': 'positive X component',
                        'overtaking_left': 'negative X component'
                    },
                    'relative_to_ego': True,
                    'timestamp': datetime.now().isoformat()
                }
                track_data['speeds'].append(speed_data)
        
        # Handle disappeared tracks (same as before)
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                if track_id not in self.disappeared_tracks:
                    self.disappeared_tracks[track_id] = {
                        'count': 0,
                        'label': self.id_label_map[track_id],
                        'positions': self.track_history[track_id]['positions'],
                        'features': self.track_history[track_id]['features'],
                        'class': self.track_history[track_id]['class'],
                        'depth_history': self.track_history[track_id].get('depth_history', deque()),
                        'bbox_heights': self.track_history[track_id].get('bbox_heights', deque()),
                        'speeds': self.track_history[track_id].get('speeds', deque())
                    }
                self.disappeared_tracks[track_id]['count'] += 1
                
                if self.disappeared_tracks[track_id]['count'] > self.max_disappeared:
                    del self.disappeared_tracks[track_id]
                    del self.track_history[track_id]
                    del self.id_label_map[track_id]

# -------------------------
# FRAME PROCESSING
# -------------------------
def process_frame_with_depth_and_speed(
    frame_path,
    model,
    vehicle_classes,
    target_classes,
    camera_name,
    output_base_dir,
    DEPTH_PRO_AVAILABLE,
    depth_model,
    depth_transform,
    depth_device,
    run_dict=None,
):
    if run_dict is None:
        run_dict = {"detection": True, "depth": True, "speed": True}

    image = cv2.imread(frame_path)
    if image is None:
        print(f"Failed to read image: {frame_path}")
        return

    # Prepare output dirs
    output_dir = os.path.join(output_base_dir, f"{camera_name}_Segmented")
    json_dir = os.path.join(output_base_dir, f"{camera_name}_Annotations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    depth_map, depth_colormap = None, None
    if run_dict.get("depth", True) and DEPTH_PRO_AVAILABLE and depth_model is not None:
        depth_map, depth_colormap = estimate_depth(image, depth_model, depth_transform, depth_device)
        depth_dir = os.path.join(output_base_dir, f"{camera_name}_DepthMaps")
        depth_colormap_dir = os.path.join(output_base_dir, f"{camera_name}_DepthColormaps")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(depth_colormap_dir, exist_ok=True)
        np.save(os.path.join(depth_dir, os.path.splitext(os.path.basename(frame_path))[0] + '_depth.npy'), depth_map)
        cv2.imwrite(os.path.join(depth_colormap_dir, os.path.splitext(os.path.basename(frame_path))[0] + '_depth_colormap.jpg'), depth_colormap)

    json_data = {
        "info": {"description": "L2D Dataset Detection Results", "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "images": [{
            "id": 1,
            "file_name": os.path.basename(frame_path),
            "width": image.shape[1],
            "height": image.shape[0],
            "camera": camera_name,
            "has_depth": depth_map is not None
        }],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "vehicle", "supercategory": "traffic"},
            {"id": 2, "name": "pedestrian", "supercategory": "traffic"}
        ]
    }

    current_detections = {}

    # -----------------
    # Detection phase
    # -----------------
    if run_dict.get("detection", True):
        try:
            # Run RF-DETR detection
            detection_results = rfdetr_detect_objects(model, image, confidence_threshold=0.4)

            # Map detections to target classes
            mapped_detections = map_rfdetr_to_target_classes(detection_results, target_classes)

            # Store results with depth info if available
            for track_id, (bbox, cls_name) in mapped_detections.items():
                depth_stats = extract_depth_from_bbox(depth_map, bbox) if depth_map is not None else None
                current_detections[track_id] = (bbox, cls_name, depth_stats)

            last_detections_cache[camera_name] = current_detections.copy()

        except Exception as e:
            print(f"Error processing RF-DETR detection for {frame_path}: {e}")

    else:
        # Use last known detections if available
        current_detections = last_detections_cache.get(camera_name, {})

    # -----------------
    # Speed tracking phase
    # -----------------
    if run_dict.get("speed", True):
        if camera_name not in camera_trackers:
            camera_trackers[camera_name] = EnhancedRobustTracker(camera_name)
        tracker = camera_trackers[camera_name]
        if current_detections:
            tracker.update(image, current_detections, vehicle_classes)
    else:
        tracker = None

    # -----------------
    # Annotate image & JSON
    # -----------------
    annotation_id = 1
    for track_id, (box, cls_name, depth_stats) in current_detections.items():
        x1, y1, x2, y2 = box
        label = f"ID{track_id}"

        speed_info = None
        if run_dict.get("speed", True) and tracker:
            track_data = tracker.track_history.get(track_id, {})
            if 'speeds' in track_data and len(track_data['speeds']) > 0:
                speed_info = track_data['speeds'][-1]
                label += f" {speed_info['speed_kmh']:.1f}km/h"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        json_data["annotations"].append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": 1 if cls_name in vehicle_classes else 2,
            "track_id": label,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "attributes": {
                "class": cls_name,
                "depth_stats": depth_stats,
                "speed_info": speed_info
            }
        })
        annotation_id += 1

    # Save results
    cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_path)), image)
    json_path = os.path.join(json_dir, os.path.splitext(os.path.basename(frame_path))[0] + '.json')
    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(json_data), f, indent=2)

    return {
        "frame": os.path.basename(frame_path),
        "detections": len(json_data["annotations"]),
        "path": os.path.join(output_dir, os.path.basename(frame_path)),
        "json": json_path,
        "has_depth": depth_map is not None
    }

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
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# -------------------------
# EPISODE PROCESSING
# -------------------------
def process_episode_frames_with_depth_and_speed(
    DEPTH_PRO_AVAILABLE,depth_model,depth_transform,depth_device,
    episode_num,
    camera_views,
    input_base_dir,
    output_base_dir,
    model,
    vehicle_classes,
    target_classes,
    run_dict=None,
    depth_available=False,
    camera_trackers=None
):
    if run_dict is None:
        run_dict = {"detection": True, "depth": True, "speed": True}

    print(f"\n🔹 Processing episode {episode_num} with config: {run_dict}")
    results = {}

    episode_dir = os.path.join(input_base_dir, f"Episode{episode_num:06d}")
    if not os.path.exists(episode_dir):
        print(f"⚠️ Episode folder not found: {episode_dir}")
        return results

    for camera in camera_views:
        camera_frames_dir = os.path.join(episode_dir, camera)
        if not os.path.exists(camera_frames_dir):
            print(f"⚠️ Missing camera data: {camera_frames_dir}")
            continue

        camera_name = camera.split('.')[-1]
        results[camera_name] = []

        frame_files = sorted([f for f in os.listdir(camera_frames_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not frame_files:
            print(f"⚠️ No frames found for {camera_name}")
            continue

        print(f"  📷 Camera {camera_name}: {len(frame_files)} frames found")

        for frame_file in frame_files:
            frame_path = os.path.join(camera_frames_dir, frame_file)
            frame_result = process_frame_with_depth_and_speed(
                frame_path,
                model,
                vehicle_classes,
                target_classes,
                camera_name,
                output_base_dir,
                DEPTH_PRO_AVAILABLE,
                depth_model,
                depth_transform,
                depth_device,
                run_dict,
            )
            if frame_result:
                results[camera_name].append(frame_result)

        print(f"  ✅ Processed {len(results[camera_name])} frames for {camera_name}")

    return results

def initialize_rfdetr_model():
    """Initialize RF-DETR model"""
    try:
        print("🤖 Initializing RF-DETR model...")
        model = RFDETRBase()
        print("✅ RF-DETR model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading RF-DETR model: {e}")
        return None
    
def rfdetr_detect_objects(model, image, confidence_threshold=0.4):
    """
    Run RF-DETR detection on image and return results in YOLO-compatible format
    """
    try:
        # Convert BGR to RGB for RF-DETR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Run RF-DETR detection
        detections = model.predict(pil_image, threshold=confidence_threshold)
        
        # Convert RF-DETR results to format compatible with existing tracker
        detection_results = {}
        
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                # Get detection data
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                
                # Get class name from COCO classes
                class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                
                # Create a pseudo track_id for compatibility (will be handled by tracker)
                pseudo_track_id = f"det_{i}"
                
                # Convert bbox to integer coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                detection_results[pseudo_track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'class_id': int(class_id)
                }
        
        return detection_results
    
    except Exception as e:
        print(f"Error in RF-DETR detection: {e}")
        return {}
    
def map_rfdetr_to_target_classes(detection_results, target_classes):
    """
    Filter RF-DETR detections to only include target classes and assign track IDs
    """
    filtered_detections = {}
    track_id_counter = 1
    
    for det_key, detection in detection_results.items():
        class_name = detection['class_name']
        
        # Map some COCO classes to our target classes
        class_mapping = {
            'bicycle': 'bicycle',
            'car': 'car', 
            'motorcycle': 'motorcycle',
            'bus': 'bus',
            'train': 'train',
            'truck': 'truck',
            'boat': 'boat',
            'person': 'person'
        }
        
        mapped_class = class_mapping.get(class_name, class_name)
        
        if mapped_class in target_classes:
            # Assign a new track ID
            track_id = track_id_counter
            filtered_detections[track_id] = (detection['bbox'], mapped_class)
            track_id_counter += 1
    
    return filtered_detections

def process_frames(
    min_ep,
    max_ep=-1,
    input_base_dir='../data/raw/L2D/frames',
    output_base_dir='../data/processed_frames/L2D',
    cameras_on=None,
    run_dict=None,
    verbose=False
):

    if max_ep == -1: max_ep = min_ep + 1

    # 1️⃣ Initialize DepthPro and global vars
    if verbose: print("\n=== Setting up DepthPro ===")
    DEPTH_PRO_AVAILABLE,depth_model,depth_transform,depth_device = setting_up(verbose=verbose)

    # 2️⃣ Initialize RF-DETR model
    if verbose: print("\n=== Initializing RF-DETR Model ===")
    model = initialize_rfdetr_model()
    if model is None:
        raise RuntimeError("Failed to initialize RF-DETR model.")

    # 3️⃣ Default camera list
    if cameras_on is None:
        cameras_on = [
            "observation.images.front_left",
            "observation.images.left_forward", 
            "observation.images.right_forward",
            "observation.images.right_backward",
            "observation.images.rear",
            "observation.images.left_backward"
        ]

    # 4️⃣ Default run toggles
    if run_dict is None:
        run_dict = {
            "detection": True,
            "depth": True,
            "speed": True
        }

    # 5️⃣ Vehicle & target classes
    vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat']
    target_classes = vehicle_classes + ['person']

    # 6️⃣ Process episodes
    all_results = {}
    if depth_device.type == "cuda": 
        try: model.optimize_for_inference()
        except: pass

    for ep in range(min_ep, max_ep):
        #if verbose: print(f"\n🚗 Processing Episode {ep}")
        #try:
            episode_results = process_episode_frames_with_depth_and_speed(
                DEPTH_PRO_AVAILABLE,depth_model,depth_transform,depth_device,
                episode_num=ep,
                camera_views=cameras_on,
                input_base_dir=input_base_dir,
                output_base_dir=output_base_dir + f"/Episode{ep:06d}",
                model=model,
                vehicle_classes=vehicle_classes,
                target_classes=target_classes,
                run_dict=run_dict
            )
            all_results[ep] = episode_results
            if verbose: print(f"✅ Finished Episode {ep}")
        #except Exception as e:
        #    print(f"⚠️ Error processing Episode {ep}: {e}")

    #return all_results
