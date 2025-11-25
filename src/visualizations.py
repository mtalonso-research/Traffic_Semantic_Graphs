import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import webbrowser
import os
from src.utils import extract_frames
from src.risk_analysis.risk_analysis import RiskAnalysis
import os
import webbrowser
import networkx as nx
import ipycytoscape
from pyvis.network import Network
from collections import defaultdict
from tqdm import tqdm

try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None

def is_in_jupyter():
    """
    Checks if the code is running in a Jupyter Notebook.
    
    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.
    """
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
    # Step 1: Input Handling and Data Extraction
    if not isinstance(frames_data, list):
        frames_data = [frames_data]
        animated = False
    else: 
        animated = True
        if diagnostics:
            print("Warning: 'diagnostics' provided in animated mode. Risk visualization is "
                  "only supported in static (single-frame) mode and will be ignored.")
            diagnostics = None

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

    # Step 2: Merge Diagnostics Data (if provided)
    if diagnostics and not animated:
        diag_df = pd.DataFrame(diagnostics)
        df = pd.merge(df, diag_df, on='id', how='left')
        
        df['risk_i'] = df['risk_i'].fillna(0.0)
        df['ttc'] = df['ttc'].fillna(np.inf)
        df['drac'] = df['drac'].fillna(0.0)
        df['pet'] = df['pet'].fillna(np.inf)

    # Step 3: Plotting Logic
    if animated:
        # Step 3a: Animated Plot with Slider (No Risk)
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
        # Step 3b: Static Plot
        title_prefix = 'Traffic Scene Visualization'
        
        weather_desc = "Weather: N/A"
        if frames_data:
            first_frame = frames_data[0]
            env_features = first_frame.get("environment", {}).get("features", {})
            code = env_features.get('conditions', 0)
            try: weather_desc = {0:'clear',1:'overcast',2:'raining',3:'snow',4:'fog'}[code]
            except: weather_desc = 'not processed yet'

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
            # Step 3b-1: Static Plot WITH RISK
            title_base = f"{title_prefix} with Risk Analysis"
            if scene_risk is not None:
                title_base += f" (Scene Risk: {scene_risk:.3f})"
            
            title = f"{title_base}<br>{subtitle}"

            ego_df = df[df['type'] == 'ego']
            agents_df = df[df['type'] != 'ego']
            
            fig = px.scatter(
                agents_df, x='plot_x', y='plot_y', symbol='type',
                color='risk_i',
                color_continuous_scale='YlGn_r',
                range_color=[0.0, 1.0],
                hover_name='id', 
                hover_data={
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
            
            fig.add_trace(go.Scatter(
                x=ego_df['plot_x'],
                y=ego_df['plot_y'],
                mode='markers',
                marker=dict(
                    color=px.colors.qualitative.Plotly[0],
                    size=10,
                    symbol='cross'
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
            
            fig.update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            fig.update_traces(marker=dict(size=15), selector=dict(type='scatter', name_not='ego'))

        else:
            # Step 3b-2: Static Plot WITHOUT RISK
            title = f"{title_prefix}<br>{subtitle}"
            
            fig = px.scatter(
                df, x='plot_x', y='plot_y', symbol='type', color='type',
                hover_name='id', hover_data={'vx': ':.2f', 'vy': ':.2f'},
                title=title,
                labels=labels, width=1200, height=800, template='plotly_dark'
            )
            fig.update_traces(marker=dict(size=15))

        # Step 3c: Add velocity vectors for static plot
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

    # Step 4: Final Layout Updates
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
    # Step 1: Initialize analyzer and load scene data
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

    # Step 2: Extract frames from the scene
    frames = extract_frames(scene)
    if not frames:
        print(f"Warning: No frames extracted from episode {episode}")
        return

    diagnostics = None
    fig = None
    scene_risk = None

    # Step 3: Generate visualization based on frame selection
    if frame is not None:
        # Step 3a: Static Frame Visualization
        if frame < 0 or frame >= len(frames):
            print(f"Error: Frame index {frame} is out of range (0 to {len(frames)-1}).")
            return
            
        current_frame = frames[frame]
        
        if analyzer:
            #try:
                risk, diags, _, _ = analyzer.analyze_frame(current_frame)
                diagnostics = diags
                scene_risk = risk
                print(f"Analyzed frame {frame}. Max risk: {scene_risk:.3f}")
            #except Exception as e:
            #    print(f"Error during frame analysis: {e}")
                
        fig = visualize_frames(
            current_frame, 
            diagnostics=diagnostics, 
            scene_risk=scene_risk
        )
        
    else:
        # Step 3b: Animated Scene Visualization
        fig = visualize_frames(frames)

    # Step 4: Save and Show Figure
    if fig:
        if is_in_jupyter():
            print("Displaying figure in notebook...")
            fig.show()
        else:
            print("Not in Jupyter. Saving to HTML and opening browser...")
            figures_dir = 'figures/scene/'
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            
            dataset_name = os.path.basename(dataset_path) 
            frame_suffix = f'_frame_{frame}' if frame is not None else '_animated'
            html_file_path = os.path.join(figures_dir, f'scene_visual_{dataset_name}_{episode}{frame_suffix}.html')
            
            fig.write_html(html_file_path)
            print(f"Figure saved to: {html_file_path}")
            webbrowser.open('file://' + os.path.realpath(html_file_path))
    else:
        print("Error: Figure was not generated.")

def json_graph_to_networkx(graph_data):
    """
    Converts a JSON graph to a NetworkX graph.
    
    Args:
        graph_data (dict): The JSON graph data.
        
    Returns:
        tuple: A tuple containing the NetworkX graph and a mapping of original node IDs to new node IDs.
    """
    # Step 1: Initialize graph and mappings
    G = nx.DiGraph()
    node_mapping = {}
    existing_ids = set()

    # Step 2: Add nodes safely
    for node_type, node_list in graph_data.get("nodes", {}).items():
        for i, node in enumerate(node_list):
            original_id = node.get("id", f"unknown_{len(node_mapping)}")

            unique_id = original_id
            while unique_id in existing_ids:
                unique_id = f"{original_id}_{i}"
            existing_ids.add(unique_id)

            node_id = f"{unique_id}"
            node_mapping[original_id] = node_id
            
            features = dict(node.get("features", {}))
            features.update({"id": node_id, "type": node_type})
            
            G.add_node(node_id, **features)

    # Step 3: Add edges
    for edge_type, edge_list in graph_data.get("edges", {}).items():
        for edge in edge_list:
            src = edge.get("source")
            tgt = edge.get("target")
            src_id = node_mapping.get(src)
            tgt_id = node_mapping.get(tgt)

            if src_id is None or tgt_id is None:
                continue

            attributes = dict(edge.get("features", {}))
            attributes["interaction"] = edge_type
            G.add_edge(src_id, tgt_id, **attributes)

    return G, node_mapping

def combined_graph_viewer(graphs_by_frame: dict, episode_num: int, dataset_path: str):
    """
    Visualizes a combined graph of all frames in an episode.
    
    Args:
        graphs_by_frame (dict): A dictionary of graphs, where keys are frame numbers and values are graph data.
        episode_num (int): The episode number.
        dataset_path (str): The path to the dataset.
        
    Returns:
        ipycytoscape.CytoscapeWidget or None: A Cytoscape widget if in a Jupyter Notebook, otherwise None.
    """
    # Step 1: Initialize combined graph and color map
    type_color_map = {
        'vehicle': '#1f77b4',
        'pedestrian': '#ff7f0e',
        'environment': '#2ca02c',
        'ego': '#FF0000'
    }

    G_combined = nx.DiGraph()
    framewise_nodes = []

    # Step 2: Combine all frames into a single graph
    for frame_key, graph_data in graphs_by_frame.items():
        G_frame, node_mapping = json_graph_to_networkx(graph_data)
        G_combined.update(G_frame)
        framewise_nodes.append(node_mapping)

    # Step 3: Add temporal edges between frames
    for i in range(len(framewise_nodes) - 1):
        for orig_id in framewise_nodes[i]:
            if orig_id in framewise_nodes[i + 1]:
                G_combined.add_edge(
                    framewise_nodes[i][orig_id],
                    framewise_nodes[i + 1][orig_id],
                    interaction="temporal"
                )

    # Step 4: Add hover tooltips to nodes
    for node, attrs in G_combined.nodes(data=True):
        title = f"ID: {attrs.get('id', 'N/A')}\nType: {attrs.get('type', 'N/A')}"
        features = attrs.copy()
        features.pop('id', None)
        features.pop('type', None)
        for key, value in features.items():
            title += f"\n{key}: {value}"
        G_combined.nodes[node]['title'] = title

    # Step 5: Add semantic tags node
    tag_dir_base = 'data/semantic_tags'
    if 'L2D' in dataset_path:
        tag_dir = os.path.join(tag_dir_base, 'L2D')
    elif 'nuplan_boston' in dataset_path:
        tag_dir = os.path.join(tag_dir_base, 'nuplan_boston')
    elif 'nuplan_pittsburgh' in dataset_path:
        tag_dir = os.path.join(tag_dir_base, 'nuplan_pittsburgh')
    else:
        tag_dir = None

    if tag_dir:
        tag_file_path = os.path.join(tag_dir, f"episode_{episode_num:06d}.json")
        if os.path.exists(tag_file_path):
            with open(tag_file_path, 'r') as f:
                tags_content = f.read()
            
            G_combined.add_node(
                "semantic_tags",
                title=tags_content,
                label="Semantic Tags",
                color="#9467bd",
                shape="star"
            )

    # Step 6: Render graph
    if is_in_jupyter():
        # Step 6a: Render in Jupyter
        cyto = ipycytoscape.CytoscapeWidget()
        cyto.graph.add_graph_from_networkx(G_combined)

        for node in cyto.graph.nodes:
            node.data['label'] = node.data.get('id', '?')
            node.data['tooltip'] = '\n'.join(f"{k}: {v}" for k, v in node.data.items() if k != 'id')
            node_type = node.data.get('type', 'unknown')
            node.data['color'] = type_color_map.get(node_type, '#d3d3d3')

        for edge in cyto.graph.edges:
            edge.data['tooltip'] = '\n'.join(f"{k}: {v}" for k, v in edge.data.items() if k not in ['source', 'target'])

        cyto.set_style([
            {'selector': 'node',
             'style': {
                 'label': 'data(label)',
                 'background-color': 'data(color)',
                 'width': '25',
                 'height': '25',
                 'font-size': '8px'
             }},
            {'selector': 'edge',
             'style': {
                 'label': 'data(interaction)',
                 'width': 1,
                 'line-color': '#ccc',
                 'target-arrow-color': '#ccc',
                 'target-arrow-shape': 'triangle',
                 'font-size': '6px'
             }}
        ])
        return cyto
    else:
        # Step 6b: Generate and save an HTML file
        net = Network(notebook=False, directed=True)
        net.from_nx(G_combined)

        output_dir = "figures/graphs"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"episode_{episode_num}_graph.html")

        net.write_html(file_path)
        print(f"Graph visualization saved to {file_path}")
        
        webbrowser.open(f"file://{os.path.realpath(file_path)}")
        return None

def plot_feature_histogram(data_directory, node_type, feature_name):
    """
    Generates and displays a histogram for a given feature and node type
    from a directory of graph JSON files.
    
    Args:
        data_directory (str): The directory containing the graph JSON files.
        node_type (str): The type of node to plot (e.g., 'vehicle', 'pedestrian').
        feature_name (str): The name of the feature to plot.
    """
    # Step 1: Aggregate feature values
    feature_values = defaultdict(list)
    files_to_process = [f for f in os.listdir(data_directory) if f.endswith('_graph.json')]

    for filename in tqdm(files_to_process, desc=f"Aggregating data for {node_type}-{feature_name}"):
        file_path = os.path.join(data_directory, filename)
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        
        for node in graph_data.get('nodes', {}).get(node_type, []):
            if feature_name == 'all':
                for key, value in node.get('features', {}).items():
                    if isinstance(value, (int, float)):
                        feature_values[key].append(value)
            else:
                if feature_name in node.get('features', {}):
                    value = node['features'][feature_name]
                    if isinstance(value, (int, float)):
                        feature_values[feature_name].append(value)

    if not feature_values:
        print(f"No data found for node type '{node_type}' and feature '{feature_name}' in {data_directory}")
        return

    # Step 2: Create DataFrame and plot histogram
    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in feature_values.items()]))

    if feature_name == 'all':
        fig = px.histogram(df, x=df.columns, title=f'Histograms for all features of node type: {node_type}')
    else:
        fig = px.histogram(df, x=feature_name, title=f'Histogram for {node_type}-{feature_name}')

    # Step 3: Show or save the plot
    if is_in_jupyter():
        fig.show()
    else:
        output_dir = 'figures/histograms'
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{node_type}_{feature_name}_histogram.html')
        fig.write_html(file_path)
        print(f"Histogram saved to {file_path}")
        webbrowser.open(f'file://{os.path.realpath(file_path)}')