import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
from collections import defaultdict, deque
from tqdm import tqdm

# -----------------------------
# Shapely (optional dependency)
# -----------------------------
try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from shapely.validation import explain_validity
    try:
        # Shapely >= 2
        from shapely import make_valid, set_precision, intersection as s_intersection
    except Exception:
        make_valid = None
        set_precision = None
        s_intersection = None
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    make_valid = None
    set_precision = None
    s_intersection = None


class EnhancedLaneDetector:
    """
    Enhanced lane detection system with vehicle lane classification capabilities
    """

    def __init__(self, image_width=1920, image_height=1080):
        self.image_width = image_width
        self.image_height = image_height

        self.canny_low_threshold = 50
        self.canny_high_threshold = 150

        self.hough_rho = 2
        self.hough_theta = np.pi / 180
        self.hough_threshold = 20
        self.hough_min_line_length = 20
        self.hough_max_line_gap = 300

        self.min_slope_threshold = 0.5
        self.max_slope_threshold = 10.0

        self.lane_history = {
            'left_lanes': deque(maxlen=5),
            'right_lanes': deque(maxlen=5)
        }

        self.ego_lane_width_threshold = 3.0
        self.lane_overlap_threshold = 0.3

    # -----------------------------
    # Helpers for robust geometry
    # -----------------------------
    def _fix_polygon(self, g):
        """Return a cleaned, valid polygon. Accepts list-of-points or Polygon."""
        if g is None:
            return None
        if not SHAPELY_AVAILABLE:
            return g

        # Accept coords too
        if isinstance(g, (list, tuple)):
            try:
                g = Polygon(g)
            except Exception:
                return None

        if g.is_empty:
            return g

        # First pass: classic self-intersection fix
        try:
            g_fixed = g.buffer(0)
        except Exception:
            g_fixed = g

        # Shapely 2: make_valid can handle more cases
        if make_valid is not None and (not g_fixed.is_valid):
            try:
                g_fixed = make_valid(g_fixed)
            except Exception:
                pass

        # If MultiPolygon, use the largest face for the ego lane
        if getattr(g_fixed, "geom_type", "") == "MultiPolygon":
            try:
                g_fixed = max(g_fixed.geoms, key=lambda gg: gg.area)
            except Exception:
                pass

        # Optional quantization to reduce numeric robustness issues
        if set_precision is not None:
            try:
                g_fixed = set_precision(g_fixed, 1e-6)
            except Exception:
                pass

        return g_fixed

    def _polygon_coords(self, poly_or_coords):
        """Return a list of exterior coords whether input is coords or Polygon."""
        if hasattr(poly_or_coords, "exterior"):
            try:
                return list(poly_or_coords.exterior.coords)
            except Exception:
                return None
        return poly_or_coords

    # -----------------------------
    # ROI & pre-processing
    # -----------------------------
    def get_roi_vertices(self, img):
        """Define region of interest vertices for lane detection (balanced)."""
        height, width = img.shape[:2]
        top_ratio = 0.55
        left_ratio = 0.08
        right_ratio = 0.96
        top_left_ratio = 0.38
        top_right_ratio = 0.92
        vertices = np.array([[
            (int(width * left_ratio), height),
            (int(width * top_left_ratio), int(height * top_ratio)),
            (int(width * top_right_ratio), int(height * top_ratio)),
            (int(width * right_ratio), height)
        ]], dtype=np.int32)
        return vertices

    def get_roi_tuning_guide(self):
        """(Text guide omitted here to keep code compact—retain yours if needed.)"""
        return "See original tuning guide in your prior version."

    def region_of_interest(self, img, vertices):
        """Apply region of interest mask to focus on road area."""
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detect_edges(self, img):
        """Convert to grayscale and apply Canny edge detection."""
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low_threshold, self.canny_high_threshold)
        return edges

    def detect_lines(self, edges):
        """Use Hough Line Transform to detect lines in edge image."""
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        return lines

    def classify_lines(self, lines, img_shape):
        """Classify detected lines into left and right lanes based on slope."""
        if lines is None:
            return ([], [])
        left_lines = []
        right_lines = []
        height, width = img_shape[:2]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.min_slope_threshold or abs(slope) > self.max_slope_threshold:
                continue
            line_center_x = (x1 + x2) / 2
            if slope < 0 and line_center_x < width * 0.6:
                left_lines.append(line[0])
            elif slope > 0 and line_center_x > width * 0.4:
                right_lines.append(line[0])
        return (left_lines, right_lines)

    def fit_lane_line(self, lines):
        """Fit a single line through multiple line segments."""
        if not lines:
            return None
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.extend([(x1, y1), (x2, y2)])
        if len(points) < 2:
            return None
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        try:
            coeffs = np.polyfit(y_coords, x_coords, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            return (slope, intercept)
        except np.linalg.LinAlgError:
            return None

    def extrapolate_lane_line(self, slope, intercept, img_shape):
        """Extrapolate lane line to cover the full region of interest."""
        height, width = img_shape[:2]
        y1 = height
        y2 = int(height * 0.55)
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        return [(x1, y1), (x2, y2)]

    def update_roi_parameters(self, top_ratio=0.55, left_ratio=0.08, right_ratio=0.96, top_left_ratio=0.38, top_right_ratio=0.92):
        """Update ROI parameters for dynamic tuning."""
        self.roi_params = {
            'top_ratio': top_ratio,
            'left_ratio': left_ratio,
            'right_ratio': right_ratio,
            'top_left_ratio': top_left_ratio,
            'top_right_ratio': top_right_ratio
        }

    def get_roi_vertices_custom(self, img, roi_params=None):
        """Generate ROI vertices with custom parameters for testing."""
        height, width = img.shape[:2]
        if roi_params is None:
            roi_params = getattr(self, 'roi_params', {
                'top_ratio': 0.55,
                'left_ratio': 0.08,
                'right_ratio': 0.96,
                'top_left_ratio': 0.38,
                'top_right_ratio': 0.92
            })
        vertices = np.array([[
            (int(width * roi_params['left_ratio']), height),
            (int(width * roi_params['top_left_ratio']), int(height * roi_params['top_ratio'])),
            (int(width * roi_params['top_right_ratio']), int(height * roi_params['top_ratio'])),
            (int(width * roi_params['right_ratio']), height)
        ]], dtype=np.int32)
        return vertices

    def smooth_lanes(self, left_line, right_line):
        """Smooth lane detection using historical data."""
        if left_line:
            self.lane_history['left_lanes'].append(left_line)
        if right_line:
            self.lane_history['right_lanes'].append(right_line)

        smoothed_left = None
        smoothed_right = None

        if len(self.lane_history['left_lanes']) > 0:
            left_lines = list(self.lane_history['left_lanes'])
            avg_left = np.mean(left_lines, axis=0).astype(int)
            smoothed_left = [tuple(avg_left[0]), tuple(avg_left[1])]

        if len(self.lane_history['right_lanes']) > 0:
            right_lines = list(self.lane_history['right_lanes'])
            avg_right = np.mean(right_lines, axis=0).astype(int)
            smoothed_right = [tuple(avg_right[0]), tuple(avg_right[1])]

        return (smoothed_left, smoothed_right)

    def define_ego_lane_polygon(self, left_lane, right_lane, img_shape):
        """
        Define the ego vehicle's lane as a polygon between left and right lane lines.
        Returns a cleaned Shapely Polygon if available, else list of points.
        """
        if not left_lane or not right_lane:
            return None

        left_pt1, left_pt2 = left_lane
        right_pt1, right_pt2 = right_lane

        ego_lane_points = [left_pt1, left_pt2, right_pt2, right_pt1]
        # Explicitly close ring
        if ego_lane_points[0] != ego_lane_points[-1]:
            ego_lane_points.append(ego_lane_points[0])

        if SHAPELY_AVAILABLE:
            try:
                poly = Polygon(ego_lane_points)
                poly = self._fix_polygon(poly)
                if poly is None or poly.is_empty or (hasattr(poly, "area") and poly.area < 1e-9):
                    return None
                return poly
            except Exception:
                return None
        else:
            return ego_lane_points

    # -----------------------------
    # Fallback (ray casting)
    # -----------------------------
    def point_in_polygon_fallback(self, point, polygon_points):
        """
        Fallback method to check if point is in polygon
        Uses ray casting algorithm. Accepts coords or Shapely polygon.
        """
        polygon_points = self._polygon_coords(polygon_points)
        if not polygon_points or len(polygon_points) < 3:
            return False

        x, y = point
        n = len(polygon_points)
        inside = False
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = (p2x, p2y)
        return inside

    def calculate_overlap_fallback(self, bbox, ego_lane_points):
        """
        Fallback method to estimate overlap (coords or Shapely polygon accepted).
        """
        coords = self._polygon_coords(ego_lane_points)
        if not coords or len(coords) < 3:
            return 0.0

        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            return 0.0

        step = 5  # adjust for speed/accuracy trade-off
        xs = range(int(x1), int(x2), step)
        ys = range(int(y1), int(y2), step)
        if not xs or not ys:
            return 0.0

        inside = 0
        total = 0
        for xx in xs:
            for yy in ys:
                total += 1
                if self.point_in_polygon_fallback((xx, yy), coords):
                    inside += 1
        return inside / total if total else 0.0

    # -----------------------------
    # Main classification
    # -----------------------------
    def classify_object_lane_position(self, bbox, center, left_lane, right_lane, img_shape):
        """
        Classify whether an object is in the ego vehicle's lane.

        Returns:
        - ('in_lane' | 'out_of_lane_left' | 'out_of_lane_right' | 'unknown', overlap_ratio)
        """
        ego_lane_data = self.define_ego_lane_polygon(left_lane, right_lane, img_shape)
        if ego_lane_data is None:
            return ('unknown', 0.0)

        x1, y1, x2, y2 = bbox

        if SHAPELY_AVAILABLE and hasattr(ego_lane_data, 'intersection'):
            try:
                object_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                ego = self._fix_polygon(ego_lane_data)
                obj = self._fix_polygon(object_polygon)
                if ego is None or obj is None or ego.is_empty or obj.is_empty:
                    overlap_ratio = 0.0
                else:
                    if s_intersection is not None:
                        inter = s_intersection(ego, obj, grid_size=1e-6)
                    else:
                        inter = ego.intersection(obj)
                    overlap_ratio = (inter.area / obj.area) if (hasattr(inter, "area") and obj.area > 0) else 0.0
            except Exception:
                coords = self._polygon_coords(ego_lane_data)
                overlap_ratio = self.calculate_overlap_fallback(bbox, coords)
        else:
            coords = self._polygon_coords(ego_lane_data)
            overlap_ratio = self.calculate_overlap_fallback(bbox, coords)

        if overlap_ratio >= self.lane_overlap_threshold:
            return ('in_lane', overlap_ratio)
        else:
            center_x = center[0]
            if left_lane and right_lane:
                left_x_at_center_y = self.get_lane_x_at_y(left_lane, center[1])
                right_x_at_center_y = self.get_lane_x_at_y(right_lane, center[1])
                if left_x_at_center_y is not None and right_x_at_center_y is not None:
                    lane_center_x = (left_x_at_center_y + right_x_at_center_y) / 2
                    if center_x < lane_center_x:
                        return ('out_of_lane_left', overlap_ratio)
                    else:
                        return ('out_of_lane_right', overlap_ratio)
            return ('unknown', overlap_ratio)

    def get_lane_x_at_y(self, lane_line, y):
        """Get X coordinate of lane line at given Y coordinate."""
        if not lane_line:
            return None
        (x1, y1), (x2, y2) = lane_line
        if y2 == y1:
            return x1
        slope = (x2 - x1) / (y2 - y1)
        x = x1 + slope * (y - y1)
        return x

    # -----------------------------
    # End-to-end detection per frame
    # -----------------------------
    def detect_lanes(self, image):
        """Main lane detection function with enhanced capabilities."""
        vertices = self.get_roi_vertices(image)
        roi_image = self.region_of_interest(image, vertices)
        edges = self.detect_edges(roi_image)
        lines = self.detect_lines(edges)

        left_lines, right_lines = self.classify_lines(lines, image.shape)

        left_fit = self.fit_lane_line(left_lines)
        right_fit = self.fit_lane_line(right_lines)

        left_lane = None
        right_lane = None
        if left_fit:
            slope, intercept = left_fit
            left_lane = self.extrapolate_lane_line(slope, intercept, image.shape)
        if right_fit:
            slope, intercept = right_fit
            right_lane = self.extrapolate_lane_line(slope, intercept, image.shape)

        smoothed_left, smoothed_right = self.smooth_lanes(left_lane, right_lane)
        ego_lane_data = self.define_ego_lane_polygon(smoothed_left, smoothed_right, image.shape)

        if SHAPELY_AVAILABLE and hasattr(ego_lane_data, 'exterior'):
            try:
                ego_lane_coords = list(ego_lane_data.exterior.coords)
            except Exception:
                ego_lane_coords = ego_lane_data if isinstance(ego_lane_data, list) else None
        else:
            ego_lane_coords = ego_lane_data if isinstance(ego_lane_data, list) else None

        lane_info = {
            'left_lane': smoothed_left,
            'right_lane': smoothed_right,
            'ego_lane_polygon': ego_lane_data,
            'ego_lane_coords': ego_lane_coords,
            'roi_vertices': vertices.tolist(),
            'detected_lines_count': {
                'left': len(left_lines),
                'right': len(right_lines),
                'total': len(lines) if lines is not None else 0
            },
            'balanced_roi': True,
            'roi_coverage': '55% from top (balanced)'
        }
        debug_images = {
            'roi_mask': roi_image,
            'edges': edges,
            'original': image.copy()
        }
        return (lane_info, debug_images)


enhanced_lane_detector = EnhancedLaneDetector()


# -----------------------------
# Visualization utilities
# -----------------------------
def extract_vehicles_from_yolo_json(yolo_data):
    """
    Extract vehicle and pedestrian information from YOLO JSON annotations.
    """
    if not yolo_data or 'annotations' not in yolo_data:
        return []

    vehicles = []
    for annotation in yolo_data['annotations']:
        if 'attributes' in annotation:
            attrs = annotation['attributes']
            vehicle_info = {
                'track_id': annotation.get('track_id', 'Unknown'),
                'category_id': annotation.get('category_id', 0),
                'class': attrs.get('class', 'unknown'),
                'confidence': attrs.get('confidence', 0.0),
                'bbox': annotation.get('bbox', [0, 0, 0, 0]),
                'center': attrs.get('center', [0, 0]),
                'area': annotation.get('area', 0),
                'depth_stats': attrs.get('depth_stats'),
                'speed_info': attrs.get('speed_info'),
                'features': attrs.get('features'),
                'track_length': attrs.get('track_length', 0),
                'lane_classification': None,
                'lane_overlap_ratio': 0.0
            }
            if len(vehicle_info['bbox']) == 4:
                x, y, w, h = vehicle_info['bbox']
                vehicle_info['bbox_xyxy'] = [x, y, x + w, y + h]
            else:
                vehicle_info['bbox_xyxy'] = vehicle_info['bbox']
            vehicles.append(vehicle_info)
    return vehicles


def draw_enhanced_lanes_with_vehicles(image, lane_info, vehicles, ego_lane_alpha=0.2):
    """
    Draw lanes and classify vehicles with enhanced visualizations.
    """
    result_image = image.copy()
    colors = {
        'in_lane': (0, 255, 0),
        'out_of_lane_left': (255, 0, 0),
        'out_of_lane_right': (0, 0, 255),
        'unknown': (128, 128, 128),
        'left_lane': (255, 255, 0),
        'right_lane': (255, 0, 255),
        'ego_lane': (0, 255, 255)
    }
    classification_labels = {
        'in_lane': 'IN-LANE',
        'out_of_lane_left': 'OUT-LEFT',
        'out_of_lane_right': 'OUT-RIGHT',
        'unknown': 'UNKNOWN'
    }

    if lane_info['left_lane'] and lane_info['right_lane']:
        left_pt1, left_pt2 = lane_info['left_lane']
        right_pt1, right_pt2 = lane_info['right_lane']
        ego_lane_area = np.array([left_pt1, left_pt2, right_pt2, right_pt1], dtype=np.int32)
        overlay = result_image.copy()
        cv2.fillPoly(overlay, [ego_lane_area], colors['ego_lane'])
        result_image = cv2.addWeighted(result_image, 1 - ego_lane_alpha, overlay, ego_lane_alpha, 0)

    if lane_info['left_lane']:
        pt1, pt2 = lane_info['left_lane']
        cv2.line(result_image, pt1, pt2, colors['left_lane'], 6)
    if lane_info['right_lane']:
        pt1, pt2 = lane_info['right_lane']
        cv2.line(result_image, pt1, pt2, colors['right_lane'], 6)

    if lane_info['roi_vertices']:
        vertices = np.array(lane_info['roi_vertices'], dtype=np.int32)
        cv2.polylines(result_image, [vertices], True, (255, 255, 255), 2)

    for vehicle in vehicles:
        bbox = vehicle['bbox_xyxy']
        x1, y1, x2, y2 = map(int, bbox)
        classification = vehicle.get('lane_classification', 'unknown')
        color = colors.get(classification, colors['unknown'])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

        label_parts = [vehicle['track_id']]
        label_parts.append(classification_labels.get(classification, 'UNK'))

        overlap = vehicle.get('lane_overlap_ratio', 0.0)
        if overlap > 0:
            label_parts.append(f'{overlap:.1%}')

        speed_info = vehicle.get('speed_info')
        if speed_info and speed_info.get('speed_kmh', 0) > 2:
            label_parts.append(f"{speed_info['speed_kmh']:.1f}km/h")

        depth_stats = vehicle.get('depth_stats')
        if depth_stats and depth_stats.get('median_depth'):
            label_parts.append(f"{depth_stats['median_depth']:.1f}m")

        display_label = ' | '.join(label_parts)
        label_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(result_image, display_label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tiny legend
    legend_y = 30
    for classification, color in colors.items():
        if classification in ['in_lane', 'out_of_lane_left', 'out_of_lane_right', 'unknown']:
            label = classification_labels.get(classification, classification)
            cv2.rectangle(result_image, (result_image.shape[1] - 200, legend_y),
                          (result_image.shape[1] - 180, legend_y + 20), color, -1)
            cv2.putText(result_image, label, (result_image.shape[1] - 170, legend_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 30

    return result_image


def create_comprehensive_debug_visualization(original_image, debug_images, lane_info, vehicles):
    """
    Create comprehensive debug visualization with vehicle classifications.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    result_image = draw_enhanced_lanes_with_vehicles(original_image, lane_info, vehicles)

    axes[0, 0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Lane Detection + Vehicle Classification')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(debug_images['roi_mask'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Balanced Region of Interest')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(debug_images['edges'], cmap='gray')
    axes[0, 2].set_title('Edge Detection (Canny)')
    axes[0, 2].axis('off')

    axes[1, 0].axis('off')
    classification_counts = {}
    for vehicle in vehicles:
        classification = vehicle.get('lane_classification', 'unknown')
        classification_counts[classification] = classification_counts.get(classification, 0) + 1

    stats_text = (
        f"Vehicle Lane Classification:\n\n"
        f"Total Vehicles: {len(vehicles)}\n\n"
        f"In Ego Lane: {classification_counts.get('in_lane', 0)}\n"
        f"Out of Lane (Left): {classification_counts.get('out_of_lane_left', 0)}\n"
        f"Out of Lane (Right): {classification_counts.get('out_of_lane_right', 0)}\n"
        f"Unknown Position: {classification_counts.get('unknown', 0)}\n\n"
        f"Lane Detection:\n"
        f"Left Lane: {('✓' if lane_info['left_lane'] else '✗')}\n"
        f"Right Lane: {('✓' if lane_info['right_lane'] else '✗')}\n"
        f"Balanced ROI: {('✓' if lane_info.get('balanced_roi') else '✗')}\n\n"
        f"Detected Lines:\n"
        f"- Left: {lane_info['detected_lines_count']['left']}\n"
        f"- Right: {lane_info['detected_lines_count']['right']}\n"
        f"- Total: {lane_info['detected_lines_count']['total']}\n"
    )
    axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

    axes[1, 1].axis('off')
    in_lane_speeds, out_lane_speeds = [], []
    for vehicle in vehicles:
        speed_info = vehicle.get('speed_info')
        if speed_info and speed_info.get('speed_kmh'):
            speed = speed_info['speed_kmh']
            if vehicle.get('lane_classification') == 'in_lane':
                in_lane_speeds.append(speed)
            elif 'out_of_lane' in vehicle.get('lane_classification', ''):
                out_lane_speeds.append(speed)
    in_lane_avg = np.mean(in_lane_speeds) if in_lane_speeds else 0
    in_lane_max = np.max(in_lane_speeds) if in_lane_speeds else 0
    out_lane_avg = np.mean(out_lane_speeds) if out_lane_speeds else 0
    out_lane_max = np.max(out_lane_speeds) if out_lane_speeds else 0
    speed_text = (
        f'Speed Analysis:\n\n'
        f'In-Lane Vehicles:\n'
        f'Count: {len(in_lane_speeds)}\nAvg Speed: {in_lane_avg:.1f} km/h\nMax Speed: {in_lane_max:.1f} km/h\n\n'
        f'Out-of-Lane Vehicles:\n'
        f'Count: {len(out_lane_speeds)}\nAvg Speed: {out_lane_avg:.1f} km/h\nMax Speed: {out_lane_max:.1f} km/h\n\n'
        f'Note: Speeds are relative to ego vehicle\n'
    )
    axes[1, 1].text(0.1, 0.9, speed_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

    axes[1, 2].axis('off')
    in_lane_depths, out_lane_depths = [], []
    for vehicle in vehicles:
        depth_stats = vehicle.get('depth_stats')
        if depth_stats and depth_stats.get('median_depth'):
            depth = depth_stats['median_depth']
            if vehicle.get('lane_classification') == 'in_lane':
                in_lane_depths.append(depth)
            elif 'out_of_lane' in vehicle.get('lane_classification', ''):
                out_lane_depths.append(depth)
    in_lane_avg_depth = np.mean(in_lane_depths) if in_lane_depths else 0
    in_lane_min_depth = np.min(in_lane_depths) if in_lane_depths else 0
    out_lane_avg_depth = np.mean(out_lane_depths) if out_lane_depths else 0
    out_lane_min_depth = np.min(out_lane_depths) if out_lane_depths else 0
    depth_text = (
        f'Depth Analysis:\n\n'
        f'In-Lane Vehicles:\n'
        f'Count: {len(in_lane_depths)}\nAvg Distance: {in_lane_avg_depth:.1f}m\nMin Distance: {in_lane_min_depth:.1f}m\n\n'
        f'Out-of-Lane Vehicles:\n'
        f'Count: {len(out_lane_depths)}\nAvg Distance: {out_lane_avg_depth:.1f}m\nMin Distance: {out_lane_min_depth:.1f}m\n\n'
        f'Balanced ROI provides optimized\n'
        f'road area detection with tunable\n'
        f'parameters for different conditions\n'
    )
    axes[1, 2].text(0.1, 0.9, depth_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()


def load_yolo_annotations(yolo_annotations_dir, frame_filename):
    """
    Load YOLO+Depth+Speed annotations from existing JSON files.
    """
    base_name = os.path.splitext(frame_filename)[0]
    possible_paths = [
        os.path.join(yolo_annotations_dir, f'{base_name}.json')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def create_roi_test_scenarios():
    """Create predefined ROI test scenarios for common driving conditions."""
    scenarios = {
        'highway':  {'top_ratio': 0.5,  'left_ratio': 0.1,  'right_ratio': 0.95, 'top_left_ratio': 0.4,  'top_right_ratio': 0.9},
        'city':     {'top_ratio': 0.6,  'left_ratio': 0.05, 'right_ratio': 0.98, 'top_left_ratio': 0.35, 'top_right_ratio': 0.95},
        'balanced': {'top_ratio': 0.55, 'left_ratio': 0.08, 'right_ratio': 0.96, 'top_left_ratio': 0.38, 'top_right_ratio': 0.92},
        'focused':  {'top_ratio': 0.65, 'left_ratio': 0.12, 'right_ratio': 0.93, 'top_left_ratio': 0.42, 'top_right_ratio': 0.88},
        'wide':     {'top_ratio': 0.45, 'left_ratio': 0.03, 'right_ratio': 0.99, 'top_left_ratio': 0.32, 'top_right_ratio': 0.97}
    }
    return scenarios


def test_roi_settings(image_path, roi_params_list, detector=None):
    """
    Test multiple ROI parameter sets on a single image to find optimal settings.
    """
    if detector is None:
        detector = enhanced_lane_detector

    image = cv2.imread(image_path)
    if image is None:
        return

    num_tests = len(roi_params_list)
    cols = min(3, num_tests)
    rows = (num_tests + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_tests == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    results = []
    for i, roi_params in enumerate(roi_params_list):
        row = i // cols
        col = i % cols

        vertices = detector.get_roi_vertices_custom(image, roi_params)
        roi_image = detector.region_of_interest(image, vertices)
        edges = detector.detect_edges(roi_image)
        lines = detector.detect_lines(edges)
        line_count = len(lines) if lines is not None else 0

        test_image = image.copy()
        cv2.polylines(test_image, [vertices], True, (0, 255, 255), 3)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ax = axes[col] if rows == 1 else axes[row, col]
        ax.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Test {i + 1}: {line_count} lines\nTop:{roi_params['top_ratio']:.2f}")
        ax.axis('off')

        results.append({
            'params': roi_params,
            'line_count': line_count,
            'coverage': f"{int((1 - roi_params['top_ratio']) * 100)}% from top"
        })

    for i in range(num_tests, rows * cols):
        row = i // cols
        col = i % cols
        (axes[col] if rows == 1 else axes[row, col]).axis('off')

    plt.tight_layout()
    plt.show()

    return results


def quick_roi_optimization(test_image_path):
    """Quick ROI optimization using predefined scenarios."""
    scenarios = create_roi_test_scenarios()
    params_list = list(scenarios.values())
    results = test_roi_settings(test_image_path, params_list)
    best_result = max(results, key=lambda x: x['line_count'])
    best_index = results.index(best_result)
    scenario_names = list(scenarios.keys())
    return (scenario_names[best_index], best_result['params'])


def fine_tune_roi(base_params, test_image_path, param_name='top_ratio', test_values=None):
    """Fine-tune a specific ROI parameter."""
    if test_values is None:
        if param_name == 'top_ratio':
            test_values = [0.45, 0.5, 0.55, 0.6, 0.65]
        elif param_name in ['left_ratio', 'right_ratio']:
            test_values = [base_params[param_name] - 0.02, base_params[param_name], base_params[param_name] + 0.02]

    params_list = []
    for value in test_values:
        params = base_params.copy()
        params[param_name] = value
        params_list.append(params)
    return test_roi_settings(test_image_path, params_list)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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


# -----------------------------
# Per-frame processing
# -----------------------------
def process_frame_with_vehicle_lane_classification(
    frame_path,
    yolo_annotations_dir,
    camera_name='front_left',
    output_base_dir='../data/processed_frames/L2D_lanes',
    verbose=False
):
    """
    Process a single frame with lane detection and vehicle lane classification.
    Saves images only when verbose=True. JSON annotations are always saved.
    """
    image = cv2.imread(frame_path)
    if image is None:
        return None

    filename = os.path.basename(frame_path)
    base_name = os.path.splitext(filename)[0]
    episode_num = int(os.path.basename(os.path.dirname(os.path.dirname(frame_path))).replace('Episode', ''))
    episode_output_dir = os.path.join(output_base_dir, f'Episode{episode_num:06d}')

    lane_info, debug_images = enhanced_lane_detector.detect_lanes(image)

    yolo_data = load_yolo_annotations(yolo_annotations_dir, filename)
    vehicles = extract_vehicles_from_yolo_json(yolo_data) if yolo_data else []

    for vehicle in vehicles:
        classification, overlap_ratio = enhanced_lane_detector.classify_object_lane_position(
            vehicle['bbox_xyxy'], vehicle['center'], lane_info['left_lane'], lane_info['right_lane'], image.shape
        )
        vehicle['lane_classification'] = classification
        vehicle['lane_overlap_ratio'] = overlap_ratio

    segmented_dir = os.path.join(episode_output_dir, f'{camera_name}_Enhanced_LaneSegmented')
    debug_dir = os.path.join(episode_output_dir, f'{camera_name}_Enhanced_LaneDebug')
    annotations_dir = os.path.join(episode_output_dir, f'{camera_name}_Enhanced_LaneAnnotations')
    for directory in [segmented_dir, debug_dir, annotations_dir]:
        os.makedirs(directory, exist_ok=True)

    result_image = draw_enhanced_lanes_with_vehicles(image, lane_info, vehicles)
    output_path = os.path.join(segmented_dir, filename)

    # Save images only if verbose
    if verbose:
        cv2.imwrite(output_path, result_image)
        # Optional: debug layers
        cv2.imwrite(os.path.join(debug_dir, f'{base_name}_roi.jpg'), debug_images['roi_mask'])
        cv2.imwrite(os.path.join(debug_dir, f'{base_name}_edges.jpg'), debug_images['edges'])

    json_data = {
        'info': {
            'description': 'L2D Dataset with Enhanced Lane Detection and Vehicle Lane Classification',
            'version': '5.0',
            'contributor': 'YOLO + Depth Pro + Speed + Enhanced Lane Detection + Vehicle Classification',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_view': camera_name,
            'enhanced_features': [
                'Original ROI for focused road area detection',
                'Vehicle lane position classification',
                'Integration with YOLO+Depth+Speed annotations',
                'Ego vehicle lane identification'
            ]
        },
        'images': [{
            'id': 1,
            'file_name': filename,
            'width': image.shape[1],
            'height': image.shape[0],
            'camera': camera_name,
            'balanced_roi': True,
            'roi_coverage': '55% from top (balanced)'
        }],
        'enhanced_lane_detection': {
            'left_lane_detected': lane_info['left_lane'] is not None,
            'right_lane_detected': lane_info['right_lane'] is not None,
            'ego_lane_defined': lane_info['left_lane'] is not None and lane_info['right_lane'] is not None,
            'left_lane_points': lane_info['left_lane'],
            'right_lane_points': lane_info['right_lane'],
            'roi_vertices': lane_info['roi_vertices'],
            'detection_stats': lane_info['detected_lines_count'],
            'balanced_roi_enabled': lane_info.get('balanced_roi', False),
            'roi_coverage': lane_info.get('roi_coverage', 'Unknown')
        },
        'vehicle_lane_classification': {
            'total_vehicles': len(vehicles),
            'classification_summary': {},
            'vehicles': []
        },
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'vehicle', 'supercategory': 'traffic'},
            {'id': 2, 'name': 'pedestrian', 'supercategory': 'traffic'},
            {'id': 3, 'name': 'lane', 'supercategory': 'road_infrastructure'}
        ]
    }

    classification_counts = {}
    annotation_id = 1
    for vehicle in vehicles:
        classification = vehicle.get('lane_classification', 'unknown')
        classification_counts[classification] = classification_counts.get(classification, 0) + 1

        annotation = {
            'id': annotation_id,
            'image_id': 1,
            'category_id': vehicle['category_id'],
            'track_id': vehicle['track_id'],
            'bbox': vehicle['bbox'],
            'bbox_xyxy': vehicle['bbox_xyxy'],
            'area': vehicle['area'],
            'attributes': {
                'class': vehicle['class'],
                'confidence': vehicle['confidence'],
                'center': vehicle['center'],
                'depth_stats': vehicle['depth_stats'],
                'speed_info': vehicle['speed_info'],
                'features': vehicle['features'],
                'track_length': vehicle['track_length'],
                'lane_classification': vehicle['lane_classification'],
                'lane_overlap_ratio': vehicle['lane_overlap_ratio'],
                'in_ego_lane': vehicle['lane_classification'] == 'in_lane',
                'ego_lane_available': lane_info['left_lane'] is not None and lane_info['right_lane'] is not None
            }
        }
        json_data['annotations'].append(annotation)

        json_data['vehicle_lane_classification']['vehicles'].append({
            'track_id': vehicle['track_id'],
            'class': vehicle['class'],
            'lane_classification': vehicle['lane_classification'],
            'overlap_ratio': vehicle['lane_overlap_ratio'],
            'speed_kmh': vehicle['speed_info']['speed_kmh'] if vehicle['speed_info'] else None,
            'distance_m': vehicle['depth_stats']['median_depth'] if vehicle['depth_stats'] else None
        })
        annotation_id += 1

    json_data['vehicle_lane_classification']['classification_summary'] = classification_counts

    json_path = os.path.join(annotations_dir, f'{base_name}.json')
    json_data_clean = convert_numpy_types(json_data)
    with open(json_path, 'w') as f:
        json.dump(json_data_clean, f, indent=2)

    return {
        'frame': filename,
        'total_vehicles': len(vehicles),
        'lane_classifications': classification_counts,
        'lanes_detected': {
            'left': lane_info['left_lane'] is not None,
            'right': lane_info['right_lane'] is not None,
            'ego_lane': lane_info['left_lane'] is not None and lane_info['right_lane'] is not None
        },
        'output_path': output_path,
        'json_path': json_path,
        'debug_path': os.path.join(output_base_dir, f'Episode{episode_num:06d}', f'{camera_name}_Enhanced_LaneDebug')
    }


# -----------------------------
# Episode-level processing
# -----------------------------
def process_episode_with_vehicle_lane_classification(
    episode_num,
    raw_frames_dir,
    yolo_annotations_dir,
    output_base_dir,
    camera_name='front_left',
    verbose=False
):
    """
    Process entire episode with enhanced lane detection and vehicle classification.
    Saves images only when verbose=True. JSON summary is always saved.
    """
    camera_frames_dir = os.path.join(raw_frames_dir, f'Episode{episode_num:06d}', f'observation.images.{camera_name}')
    episode_output_dir = os.path.join(output_base_dir, f'Episode{episode_num:06d}')
    episode_yolo_annotations_dir = os.path.join(yolo_annotations_dir, f'Episode{episode_num:06d}', f'{camera_name}_Annotations')

    if not os.path.exists(camera_frames_dir):
        return None

    frame_files = sorted([f for f in os.listdir(camera_frames_dir) if f.lower().endswith(('.jpg', '.png'))])
    if not frame_files:
        return None

    results = []
    total_classification_counts = {'in_lane': 0, 'out_of_lane_left': 0, 'out_of_lane_right': 0, 'unknown': 0}
    lanes_detected_count = {'left': 0, 'right': 0, 'both': 0}
    total_vehicles = 0

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(camera_frames_dir, frame_file)
        result = process_frame_with_vehicle_lane_classification(
            frame_path, episode_yolo_annotations_dir, camera_name, output_base_dir, verbose=verbose
        )
        if result:
            results.append(result)
            total_vehicles += result['total_vehicles']
            for classification, count in result['lane_classifications'].items():
                if classification in total_classification_counts:
                    total_classification_counts[classification] += count
            if result['lanes_detected']['left']:
                lanes_detected_count['left'] += 1
            if result['lanes_detected']['right']:
                lanes_detected_count['right'] += 1
            if result['lanes_detected']['ego_lane']:
                lanes_detected_count['both'] += 1

    summary = {
        'episode': episode_num,
        'camera': camera_name,
        'processing_info': {
            'total_frames': len(frame_files),
            'successfully_processed': len(results),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_features': [
                'Balanced ROI for optimized road area detection',
                'ROI parameter tuning and optimization tools',
                'Vehicle lane position classification',
                'Integration with YOLO+Depth+Speed data',
                'Ego vehicle lane identification'
            ]
        },
        'lane_detection_statistics': {
            'frames_with_left_lane': lanes_detected_count['left'],
            'frames_with_right_lane': lanes_detected_count['right'],
            'frames_with_ego_lane': lanes_detected_count['both'],
            'left_lane_detection_rate': lanes_detected_count['left'] / len(results) if results else 0,
            'right_lane_detection_rate': lanes_detected_count['right'] / len(results) if results else 0,
            'ego_lane_detection_rate': lanes_detected_count['both'] / len(results) if results else 0
        },
        'vehicle_classification_statistics': {
            'total_vehicles_detected': total_vehicles,
            'classification_breakdown': total_classification_counts,
            'in_lane_percentage': (total_classification_counts['in_lane'] / total_vehicles * 100) if total_vehicles > 0 else 0,
            'out_of_lane_percentage': ((total_classification_counts['out_of_lane_left'] + total_classification_counts['out_of_lane_right']) / total_vehicles * 100) if total_vehicles > 0 else 0,
            'unknown_percentage': (total_classification_counts['unknown'] / total_vehicles * 100) if total_vehicles > 0 else 0
        },
        'output_structure': {
            'base_directory': output_base_dir,
            'enhanced_segmented_images': f'{camera_name}_Enhanced_LaneSegmented/',
            'enhanced_debug_images': f'{camera_name}_Enhanced_LaneDebug/',
            'enhanced_annotations': f'{camera_name}_Enhanced_LaneAnnotations/',
            'summary_file': 'enhanced_lane_vehicle_summary.json'
        }
    }

    summary_path = os.path.join(episode_output_dir, 'enhanced_lane_vehicle_summary.json')
    os.makedirs(episode_output_dir, exist_ok=True)
    summary_clean = convert_numpy_types(summary)
    with open(summary_path, 'w') as f:
        json.dump(summary_clean, f, indent=2)

    return summary


# -----------------------------
# Batch entry point
# -----------------------------
def process_lanes_directory(min_ep, max_ep=-1, raw_frames_dir='./data/raw/L2D/frames', 
                            yolo_annotations_dir='./data/processed_frames/L2D', 
                            output_dir='./data/processed_frames/L2D_lanes', verbose=False):
    """
    Run lane processing across one or more episodes.
    - If min_ep is an int: processes range(min_ep, max_ep or min_ep+1)
    - If min_ep is a list: processes each episode number in the list
    Saves images only when verbose=True.
    """
    if not isinstance(min_ep, list):
        if max_ep == -1:
            max_ep = min_ep + 1
        iterable = range(min_ep, max_ep)
    else:
        iterable = min_ep

    summary = None
    for ep_num in tqdm(iterable, desc="Processing Lanes"):
        summary = process_episode_with_vehicle_lane_classification(
            episode_num=ep_num,
            raw_frames_dir=raw_frames_dir,
            yolo_annotations_dir=yolo_annotations_dir,
            output_base_dir=output_dir,
            camera_name='front_left',
            verbose=verbose
        )
    return summary
