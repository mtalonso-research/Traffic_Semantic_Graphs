import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import webbrowser
import os
from src.utils import extract_frames
from risk_analysis.risk_analysis import RiskAnalysis
import os
import webbrowser
try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None

def is_in_jupyter():
    """Checks if the code is running in a Jupyter Notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False  
    except (NameError, AttributeError):
        return False    

def visualize_frames(frames_data, use_lat_lon=False, diagnostics=None, scene_risk=None):
    """
    Visualizes traffic scene data, supporting both static and animated plots.

    - Static mode (default): Displays a single frame with velocity vectors.
    - Animated mode: Displays a sequence of frames with a time slider.
    - Risk Mode: If 'diagnostics' are provided (static mode only),
                   colors agents by risk and adds metrics to hover.

    Args:
        frames_data (dict or list): A single frame dictionary or a list of frames.
        use_lat_lon (bool): If True, plot using longitude/latitude. Defaults to False (x/y).
        diagnostics (list, optional): A list of agent diagnostics from RiskAnalysis.
                                      If provided, enables risk visualization.
        scene_risk (float, optional): The overall scene risk score, to be
                                      displayed in the title.

    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # --- 1. Input Handling and Data Extraction ---
    if not isinstance(frames_data, list):
        frames_data = [frames_data]
        animated = False
    else: 
        animated = True
        if diagnostics:
            print("Warning: 'diagnostics' provided in animated mode. Risk visualization is "
                  "only supported in static (single-frame) mode and will be ignored.")
            diagnostics = None # Ignore diagnostics in animated mode

    if not frames_data:
        print("Warning: The provided data is empty. Returning an empty figure.")
        return go.Figure()

    if not animated and len(frames_data) > 1:
        print("Warning: Static mode selected. Only the first frame will be visualized.")
        frames_data = [frames_data[0]]

    all_entities = []
    x_key = 'longitude' if use_lat_lon else 'x'
    y_key = 'latitude' if use_lat_lon else 'y'

    for frame in frames_data:
        time_identifier = frame.get('t', 0)
        
        ego_features = frame['ego']['features']
        all_entities.append({
            'id': frame['ego']['id'], 'type': 'ego', 't': time_identifier,
            'plot_x': ego_features[x_key], 'plot_y': ego_features[y_key],
            'vx': ego_features['vx'], 'vy': ego_features['vy'], 'speed': ego_features['speed']
        })

        for category in ['vehicles', 'pedestrians', 'objects']:
            if category in frame:
                for entity in frame[category]:
                    features = entity['features']
                    all_entities.append({
                        'id': entity['id'], 
                        'type': entity.get('type', category.rstrip('s')), 
                        't': time_identifier,
                        'plot_x': features[x_key], 'plot_y': features[y_key],
                        'vx': features.get('vx', 0), 'vy': features.get('vy', 0), 
                        'speed': features.get('speed',0), 'dist_to_ego': features.get('dist_to_ego',0)
                    })

    df = pd.DataFrame(all_entities)

    # --- 1b. Merge Diagnostics Data (if provided) ---
    if diagnostics and not animated:
        diag_df = pd.DataFrame(diagnostics)
        # Merge diagnostics into the main frame DataFrame
        df = pd.merge(df, diag_df, on='id', how='left')
        
        # Fill default values for entities not in diagnostics (like ego)
        df['risk_i'] = df['risk_i'].fillna(0.0)
        df['ttc'] = df['ttc'].fillna(np.inf)
        df['drac'] = df['drac'].fillna(0.0)
        df['pet'] = df['pet'].fillna(np.inf)

    # --- 2. Plotting Logic ---
    if animated:
        # --- Animated Plot with Slider (No Risk) ---
        title_prefix = 'Animated Traffic Scene'
        labels = {
            'plot_x': 'Longitude' if use_lat_lon else 'X Coordinate',
            'plot_y': 'Latitude' if use_lat_lon else 'Y Coordinate',
            'vx': 'Velocity X (m/s)', 'vy': 'Velocity Y (m/s)'
        }
        
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
        fig.update_traces(marker=dict(size=7))
        
    else:
        # --- Static Plot ---
        title_prefix = 'Traffic Scene Visualization'
        
        # Extract weather for title
        weather_desc = "Weather: N/A"
        if frames_data:
            first_frame = frames_data[0]
            env_features = first_frame.get("environment", {}).get("features", {})
            code = env_features.get('conditions', 0)
            weather_desc = {0:'clear',1:'overcast',2:'raining',3:'snow',4:'fog'}[code]

        labels = {
            'plot_x': 'Longitude' if use_lat_lon else 'X Coordinate',
            'plot_y': 'Latitude' if use_lat_lon else 'Y Coordinate',
            'vx': 'Velocity X (m/s)', 'vy': 'Velocity Y (m/s)',
            'risk_i': 'Risk Score',
            'ttc': 'TTC (s)',
            'drac': 'DRAC (m/s^2)',
            'pet': 'PET (s)'
        }
        
        subtitle = f"<sup>Weather: {weather_desc} | ({'Geographic' if use_lat_lon else 'Cartesian'})</sup>"

        if diagnostics:
            # --- Static Plot WITH RISK ---
            
            # Build title with optional risk score
            title_base = f"{title_prefix} with Risk Analysis"
            if scene_risk is not None:
                title_base += f" (Scene Risk: {scene_risk:.3f})"
            
            title = f"{title_base}<br>{subtitle}"

            # Separate ego from other entities for plotting
            ego_df = df[df['type'] == 'ego']
            agents_df = df[df['type'] != 'ego']
            
            # 1. Plot all agents (non-ego) colored by risk
            fig = px.scatter(
                agents_df, x='plot_x', y='plot_y', symbol='type',
                color='risk_i',  # Color by risk
                color_continuous_scale='YlGn_r', # Add color bar
                range_color=[0.0, 1.0],         # Set stable 0-1 range
                hover_name='id', 
                hover_data={ # Add risk metrics to hover
                    'vx': ':.2f', 'vy': ':.2f', 
                    'speed': ':.2f', 'dist_to_ego': ':.2f',
                    'risk_i': ':.3f',
                    'ttc': ':.2f',
                    'drac': ':.2f',
                    'pet': ':.2f'
                },
                title=title,
                labels=labels, width=1200, height=800, template='plotly_dark',
            )
            fig.update_traces(marker_size=8)
            
            # 2. Add the ego vehicle as a separate trace
            fig.add_trace(go.Scatter(
                x=ego_df['plot_x'],
                y=ego_df['plot_y'],
                mode='markers',
                marker=dict(
                    color=px.colors.qualitative.Plotly[0], # Default 'ego' blue
                    size=10,
                    symbol='cross' # Make ego distinct
                ),
                name='ego',
                customdata=np.stack((
                    ego_df['vx'], ego_df['vy'],
                    ego_df['risk_i'], ego_df['ttc'],
                    ego_df['drac'], ego_df['pet'],
                    ego_df['speed']
                ), axis=-1),
                hovertemplate=(
                    "<b>id: %{hovertext}</b><br><br>" +
                    "type: ego<br>" +
                    "vx: %{customdata[0]:.2f}<br>" +
                    "vy: %{customdata[1]:.2f}<br>" +
                    "speed: %{customdata[6]:.2f}<br>" +
                    "risk_i: %{customdata[2]:.3f}<br>" +
                    "ttc: %{customdata[3]:.2f}<br>" +
                    "drac: %{customdata[4]:.2f}<br>" +
                    "pet: %{customdata[5]:.2f}<br>" +
                    "<extra></extra>"
                ),
                hovertext=ego_df['id']
            ))
            
            # 3. Fix legend/colorbar overlap
            fig.update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            # Ensure other agent markers are the standard size
            fig.update_traces(marker=dict(size=15), selector=dict(type='scatter', name_not='ego'))

        else:
            # --- Static Plot WITHOUT RISK (Original Behavior) ---
            title = f"{title_prefix}<br>{subtitle}"
            
            fig = px.scatter(
                df, x='plot_x', y='plot_y', symbol='type', color='type',
                hover_name='id', hover_data={'vx': ':.2f', 'vy': ':.2f'},
                title=title,
                labels=labels, width=1200, height=800, template='plotly_dark'
            )
            fig.update_traces(marker=dict(size=15))

        # Add velocity vectors for static plot
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
    if not use_lat_lon:
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        
    return fig

def scene_visualizer(dataset_path, episode, frame=None, analyzer=None):
    """
    Loads and visualizes a scene, optionally running risk analysis.

    Args:
        dataset_path (str): Path to the dataset directory.
        episode (int or str): Episode number.
        frame (int, optional): Specific frame to visualize (static). 
                               If None, animates all frames.
        analyzer (RiskAnalysis, optional): An instance of the RiskAnalysis class.
                                           If provided, risk will be computed
                                           and visualized for a static frame.
    """
    if not analyzer:
        analyzer = RiskAnalysis()
    try:
        with open(os.path.join(dataset_path, f'{episode}_graph.json'), 'r') as file:
            scene = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {os.path.join(dataset_path, f'{episode}_graph.json')}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {episode}_graph.json")
        return

    frames = extract_frames(scene)
    if not frames:
        print(f"Warning: No frames extracted from episode {episode}")
        return

    diagnostics = None
    fig = None
    scene_risk = None # Variable to hold the risk score

    if frame is not None:
        # --- Static Frame Visualization ---
        if frame < 0 or frame >= len(frames):
            print(f"Error: Frame index {frame} is out of range (0 to {len(frames)-1}).")
            return
            
        current_frame = frames[frame]
        
        # If an analyzer is passed, run the analysis
        if analyzer:
            try:
                # Capture the scene_risk here
                risk, diags, _, _ = analyzer.analyze_frame(current_frame)
                diagnostics = diags
                scene_risk = risk # Store the risk
                print(f"Analyzed frame {frame}. Max risk: {scene_risk:.3f}")
            except Exception as e:
                print(f"Error during frame analysis: {e}")
                
        # Pass the diagnostics AND scene_risk (or None) to the visualizer
        fig = visualize_frames(
            current_frame, 
            diagnostics=diagnostics, 
            scene_risk=scene_risk
        )
        
    else:
        # --- Animated Scene Visualization ---
        # (Risk analysis is not supported for animation)
        fig = visualize_frames(frames)

    # --- Save and Show Figure ---
    if fig:
        if is_in_jupyter():
            # If in Jupyter, just display the figure inline
            print("Displaying figure in notebook...")
            fig.show()
        else:
            # If not in Jupyter (e.g., running as a script), save and open in browser
            print("Not in Jupyter. Saving to HTML and opening browser...")
            figures_dir = './figures'
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            
            # Assuming dataset_path, episode, and frame are defined earlier
            dataset_name = os.path.basename(dataset_path) 
            frame_suffix = f'_frame_{frame}' if frame is not None else '_animated'
            html_file_path = os.path.join(figures_dir, f'scene_visual_{dataset_name}_{episode}{frame_suffix}.html')
            
            fig.write_html(html_file_path)
            print(f"Figure saved to: {html_file_path}")
            webbrowser.open('file://' + os.path.realpath(html_file_path))
    else:
        print("Error: Figure was not generated.")

