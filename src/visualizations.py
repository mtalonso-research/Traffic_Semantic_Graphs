import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import webbrowser
import os
from src.utils import extract_frames

def visualize_frames(frames_data, use_lat_lon=False):
    """
    Visualizes traffic scene data, supporting both static and animated plots.

    - Static mode (default): Displays a single frame with velocity vectors.
    - Animated mode: Displays a sequence of frames with a time slider.

    Args:
        frames_data (dict or list): A single frame dictionary or a list of frames.
        use_lat_lon (bool): If True, plot using longitude/latitude. Defaults to False (x/y).
        animated (bool): If True, generate an animated plot with a slider. 
                         Defaults to False (static plot with velocity lines).

    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # --- 1. Input Handling and Data Extraction ---
    # Ensure frames_data is a list for consistent processing
    if not isinstance(frames_data, list):
        frames_data = [frames_data]
        animated = False
    else: animated = True

    if not frames_data:
        print("Warning: The provided data is empty. Returning an empty figure.")
        return go.Figure()

    # In static mode, only visualize the first frame
    if not animated and len(frames_data) > 1:
        print("Warning: Static mode selected. Only the first frame will be visualized.")
        frames_data = [frames_data[0]]

    all_entities = []
    x_key = 'longitude' if use_lat_lon else 'x'
    y_key = 'latitude' if use_lat_lon else 'y'

    for frame in frames_data:
        time_identifier = frame.get('t', 0)
        
        # Extract ego vehicle data
        ego_features = frame['ego']['features']
        all_entities.append({
            'id': frame['ego']['id'], 'type': 'ego', 't': time_identifier,
            'plot_x': ego_features[x_key], 'plot_y': ego_features[y_key],
            'vx': ego_features['vx'], 'vy': ego_features['vy']
        })

        # Extract data for other entities
        for category in ['vehicles', 'pedestrians', 'objects']:
            if category in frame:
                for entity in frame[category]:
                    features = entity['features']
                    all_entities.append({
                        'id': entity['id'], 'type': entity.get('type', 'unknown'), 't': time_identifier,
                        'plot_x': features[x_key], 'plot_y': features[y_key],
                        'vx': features.get('vx', 0), 'vy': features.get('vy', 0)
                    })

    df = pd.DataFrame(all_entities)

    # --- 2. Plotting Logic ---
    if animated:
        # --- Animated Plot with Slider ---
        title_prefix = 'Animated Traffic Scene'
        labels = {
            'plot_x': 'Longitude' if use_lat_lon else 'X Coordinate',
            'plot_y': 'Latitude' if use_lat_lon else 'Y Coordinate',
            'vx': 'Velocity X (m/s)', 'vy': 'Velocity Y (m/s)'
        }
        
        # Fix axis ranges for smooth animation
        x_range = [df['plot_x'].min() - 5, df['plot_x'].max() + 5]
        y_range = [df['plot_y'].min() - 5, df['plot_y'].max() + 5]

        fig = px.scatter(
            df, x='plot_x', y='plot_y', animation_frame='t', animation_group='id',
            symbol='type', color='type', hover_name='id', hover_data={'vx': ':.2f', 'vy': ':.2f'},
            title=f"{title_prefix} ({'Geographic' if use_lat_lon else 'Cartesian'})",
            labels=labels, width=1200, height=800, template='plotly_dark',
            range_x=x_range, range_y=y_range,
            category_orders={'type': ['ego', 'vehicle', 'pedestrian', 'object', 'unknown']}
        )
    else:
        # --- Static Plot with Velocity Lines ---
        title_prefix = 'Traffic Scene Visualization'
        labels = {
            'plot_x': 'Longitude' if use_lat_lon else 'X Coordinate',
            'plot_y': 'Latitude' if use_lat_lon else 'Y Coordinate',
            'vx': 'Velocity X (m/s)', 'vy': 'Velocity Y (m/s)'
        }

        fig = px.scatter(
            df, x='plot_x', y='plot_y', symbol='type', color='type',
            hover_name='id', hover_data={'vx': ':.2f', 'vy': ':.2f'},
            title=f"{title_prefix} ({'Geographic' if use_lat_lon else 'Cartesian'})",
            labels=labels, width=1200, height=800, template='plotly_dark'
        )

        # Add velocity vectors as line shapes
        velocity_vectors = []
        R_EARTH = 6378137
        mean_lat_rad = np.radians(df['plot_y'].mean()) if use_lat_lon else 0

        for _, row in df.iterrows():
            if row['vx'] != 0 or row['vy'] != 0:
                x0, y0 = row['plot_x'], row['plot_y']
                scale_factor = 2.0

                if use_lat_lon:
                    delta_lat = (row['vy'] / R_EARTH) * (180 / np.pi)
                    delta_lon = (row['vx'] / (R_EARTH * np.cos(mean_lat_rad))) * (180 / np.pi)
                    x1, y1 = x0 + delta_lon * scale_factor, y0 + delta_lat * scale_factor
                else:
                    x1, y1 = x0 + scale_factor * row['vx'], y0 + scale_factor * row['vy']
                
                velocity_vectors.append(go.layout.Shape(
                    type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="red", width=2)
                ))
        fig.update_layout(shapes=velocity_vectors)

    # --- 3. Final Layout Updates ---
    fig.update_traces(marker=dict(size=7))
    if not use_lat_lon:
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        
    return fig

def scene_visualizer(dataset_path,episode,frame=None):

    with open(dataset_path + f'/{episode}_graph.json','r') as file:
        scene = json.load(file)
    frames = extract_frames(scene)

    if frame:
        fig = visualize_frames(frames[frame])
    else:
        fig = visualize_frames(frames)

    dataset = dataset_path.split('/')[-1]
    html_file_path = f'./figures/scene_visual_{dataset}_{episode}.html'
    fig.write_html(html_file_path)
    webbrowser.open('file://' + os.path.realpath(html_file_path))