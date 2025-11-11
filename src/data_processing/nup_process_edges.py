import os
import json
from tqdm import tqdm

def process_edges(json_dir):
    """
    Rewrites the edges in the graph JSON files in-place.

    Args:
        json_dir (str): The directory containing the graph JSON files.
    """
    # Step 1: Get all graph JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and "_graph" in f]

    # Step 2: Process each graph file
    for fname in tqdm(json_files, desc="Processing edges"):
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r") as f:
            graph = json.load(f)

        # Step 3: Clear existing edges
        graph["edges"] = {
            "ego_to_ego": [],
            "env_to_env": [],
            "ego_to_vehicle": [],
            "ego_to_pedestrian": [],
            "ego_to_object": [],
            "ego_to_environment": [],
        }

        num_frames = len(graph["nodes"]["ego"])

        # Step 4: Create ego-to-ego and env-to-env edges
        for i in range(num_frames - 1):
            graph["edges"]["ego_to_ego"].append({"source": f"ego_{i}", "target": f"ego_{i+1}", "features": {}})
            graph["edges"]["env_to_env"].append({"source": f"env_{i}", "target": f"env_{i+1}", "features": {}})

        # Step 5: Create ego-to-frame nodes edges
        for i in range(num_frames):
            ego_node_id = f"ego_{i}"
            
            # Step 5a: Ego to env
            graph["edges"]["ego_to_environment"].append({"source": ego_node_id, "target": f"env_{i}", "features": {}})

            # Step 5b: Ego to vehicle
            for vehicle_node in graph["nodes"]["vehicle"]:
                if vehicle_node["id"].endswith(f"_{i}"):
                    graph["edges"]["ego_to_vehicle"].append({"source": ego_node_id, "target": vehicle_node["id"], "features": {}})
            
            # Step 5c: Ego to pedestrian
            for pedestrian_node in graph["nodes"]["pedestrian"]:
                if pedestrian_node["id"].endswith(f"_{i}"):
                    graph["edges"]["ego_to_pedestrian"].append({"source": ego_node_id, "target": pedestrian_node["id"], "features": {}})

            # Step 5d: Ego to object
            for object_node in graph["nodes"]["object"]:
                if object_node["id"].endswith(f"_{i}"):
                    graph["edges"]["ego_to_object"].append({"source": ego_node_id, "target": object_node["id"], "features": {}})

        # Step 6: Write the updated graph back to the file
        with open(fpath, "w") as f:
            json.dump(graph, f, indent=2)