import json
import os
from typing import Any, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import TimeToCollisionStatistics
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from src.risk_analysis.mock_map_api import MockMapFactory


def from_scene_tracked_object(scene, object_type):
    """
    Convert scene to a TrackedObject.
    
    Args:
        scene (Dict[str, Any]): scene of an agent.
        object_type (TrackedObjectType): type of the resulting object.
        
    Returns:
        TrackedObject: Agent extracted from a scene.
    """
    # Step 1: Extract the token, box, pose, and size from the scene
    token = scene['id']
    box = scene['box']
    pose = box['pose']
    size = box['size'] if 'size' in box.keys() else [0.5, 0.5]
    default_height = 1.5
    box = OrientedBox(StateSE2(*pose), width=size[0], length=size[1], height=default_height)
    
    # Step 2: Create the TrackedObject
    if object_type in AGENT_TYPES:
        return Agent(
            metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0),
            tracked_object_type=object_type,
            oriented_box=box,
            velocity=StateVector2D(scene['speed'], 0),
        )
    else:
        return StaticObject(
            metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0),
            tracked_object_type=object_type,
            oriented_box=box,
        )


def from_scene_to_tracked_objects(scene):
    """
    Convert scene["world"] into boxes
    
    Args:
        scene (Dict[str, Any]): scene["world"] coming from json
        
    Returns:
        TrackedObjects: List of boxes representing all agents
    """
    # Step 1: Check if the scene contains the "world" key
    if "world" in scene.keys():
        raise ValueError("You need to pass only the 'world' field of scene, not the whole dict!")
    
    # Step 2: Convert the scene to a list of TrackedObjects
    tracked_objects: List[TrackedObject] = []
    scene_labels_map = {
        'vehicles': TrackedObjectType.VEHICLE,
        'bicycles': TrackedObjectType.BICYCLE,
        'pedestrians': TrackedObjectType.PEDESTRIAN,
    }
    for label, object_type in scene_labels_map.items():
        if label in scene:
            tracked_objects.extend(
                [from_scene_tracked_object(scene_object, object_type) for scene_object in scene[label]]
            )

    return TrackedObjects(tracked_objects)


def setup_history(scene, scenario):
    """
    Mock the history with a mock scenario.
    
    Args:
        scene (Dict[str, Any]): The json scene.
        scenario (MockAbstractScenario): Scenario object.
        
    Returns:
        SimulationHistory: The mock history.
    """
    # Step 1: Update expert driving if it exists in the .json file
    if 'expert_ego_states' in scene:
        expert_ego_states = scene['expert_ego_states']
        expert_egos = []

        for expert_ego_state in expert_ego_states:
            ego_state = EgoState.build_from_rear_axle(
                time_point=TimePoint(expert_ego_state['time_us']),
                rear_axle_pose=StateSE2(
                    x=expert_ego_state['pose'][0], y=expert_ego_state['pose'][1], heading=expert_ego_state['pose'][2]
                ),
                rear_axle_velocity_2d=StateVector2D(
                    x=expert_ego_state['velocity'][0], y=expert_ego_state['velocity'][1]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    x=expert_ego_state['acceleration'][0], y=expert_ego_state['acceleration'][1]
                ),
                tire_steering_angle=0,
                vehicle_parameters=scenario.ego_vehicle_parameters,
            )
            expert_egos.append(ego_state)

        if len(expert_egos):
            scenario.get_expert_ego_trajectory = lambda: expert_egos
            scenario.get_ego_future_trajectory = lambda iteration, time_horizon, num_samples: expert_egos[
                iteration : iteration + time_horizon + 1 : time_horizon // num_samples
            ][1 : num_samples + 1]

    # Step 2: Load the map
    map_api = MockMapFactory().build_map_from_name('')

    # Step 3: Extract Agent Box
    tracked_objects = from_scene_to_tracked_objects(scene['world'])
    for tracked_object in tracked_objects:
        tracked_object._track_token = tracked_object.token

    ego_pose = scene['ego']['pose']
    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]

    # Step 4: Add both ego states and agents in the current timestamps
    ego_states = []
    observations = []
    ego_state = EgoState.build_from_rear_axle(
        time_point=TimePoint(scene['ego']['time_us']),
        rear_axle_pose=StateSE2(x=ego_x, y=ego_y, heading=ego_heading),
        rear_axle_velocity_2d=StateVector2D(x=scene['ego']['velocity'][0], y=scene['ego']['velocity'][1]),
        rear_axle_acceleration_2d=StateVector2D(x=scene['ego']['acceleration'][0], y=scene['ego']['acceleration'][1]),
        tire_steering_angle=0,
        vehicle_parameters=scenario.ego_vehicle_parameters,
    )
    ego_states.append(ego_state)
    observations.append(DetectionsTracks(tracked_objects))

    # Step 5: Add both ego states and agents in the future timestamp
    ego_future_states: List[Dict[str, Any]] = scene['ego_future_states'] if 'ego_future_states' in scene else []
    world_future_states: List[Dict[str, Any]] = scene['world_future_states'] if 'world_future_states' in scene else []
    assert len(ego_future_states) == len(world_future_states), (
        f'Length of world world_future_states: '
        f'{len(world_future_states)} and '
        f'length of ego_future_states: '
        f'{len(ego_future_states)} not same'
    )
    for index, (ego_future_state, future_world_state) in enumerate(zip(ego_future_states, world_future_states)):
        pose = ego_future_state['pose']
        time_us = ego_future_state['time_us']
        ego_state = EgoState.build_from_rear_axle(
            time_point=TimePoint(time_us),
            rear_axle_pose=StateSE2(x=pose[0], y=pose[1], heading=pose[2]),
            rear_axle_velocity_2d=StateVector2D(x=ego_future_state['velocity'][0], y=ego_future_state['velocity'][1]),
            rear_axle_acceleration_2d=StateVector2D(
                x=ego_future_state['acceleration'][0], y=ego_future_state['acceleration'][1]
            ),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            tire_steering_angle=0,
        )
        future_tracked_objects = from_scene_to_tracked_objects(future_world_state)
        for future_tracked_object in future_tracked_objects:
            future_tracked_object._track_token = future_tracked_object.token

        ego_states.append(ego_state)
        observations.append(DetectionsTracks(future_tracked_objects))

    if ego_states:
        scenario.get_number_of_iterations = lambda: len(ego_states)

    # Step 6: Add simulation iterations and trajectory for each iteration
    simulation_iterations = []
    trajectories = []
    for index, ego_state in enumerate(ego_states):
        simulation_iterations.append(SimulationIteration(ego_state.time_point, index))
        history_buffer = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=10,
            ego_states=[ego_states[index]],
            observations=[observations[index]],
            sample_interval=1,
        )
        planner_input = PlannerInput(
            iteration=SimulationIteration(ego_states[index].time_point, 0), history=history_buffer
        )
        planner = SimplePlanner(horizon_seconds=10.0, sampling_time=1, acceleration=[0.0, 0.0])
        trajectories.append(planner.compute_planner_trajectory(planner_input))

    # Step 7: Create simulation histories
    history = SimulationHistory(map_api, scenario.get_mission_goal())
    for ego_state, simulation_iteration, trajectory, observation in zip(
        ego_states, simulation_iterations, trajectories, observations
    ):
        history.add_sample(
            SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=observation,
                traffic_light_status=scenario.get_traffic_light_status_at_iteration(simulation_iteration.index),
            )
        )

    return history


def build_mock_history_scenario_test(scene):
    """
    A common template to create a test history and scenario.
    
    Args:
        scene (Dict[str, Any]): A json format to represent a scene.
        
    Returns:
        Tuple[SimulationHistory, MockAbstractScenario]: The mock history and scenario.
    """
    # Step 1: Get the goal pose
    goal_pose = None
    if 'goal' in scene and 'pose' in scene['goal'] and scene['goal']['pose']:
        goal_pose = StateSE2(x=scene['goal']['pose'][0], y=scene['goal']['pose'][1], heading=scene['goal']['pose'][2])
    
    # Step 2: Set the initial timepoint and time_step from the scene
    if (
        'ego' in scene
        and 'time_us' in scene['ego']
        and 'ego_future_states' in scene
        and scene['ego_future_states']
        and 'time_us' in scene['ego_future_states'][0]
    ):
        initial_time_us = TimePoint(time_us=scene['ego']['time_us'])
        time_step = (scene['ego_future_states'][0]['time_us'] - scene['ego']['time_us']) * 1e-6
        mock_abstract_scenario = MockAbstractScenario(initial_time_us=initial_time_us, time_step=time_step)
    else:
        mock_abstract_scenario = MockAbstractScenario()
    if goal_pose is not None:
        mock_abstract_scenario.get_mission_goal = lambda: goal_pose
    
    # Step 3: Setup the history
    history = setup_history(scene, mock_abstract_scenario)

    return history, mock_abstract_scenario


def _create_scene_from_graph(graph_data):
    """
    Converts a graph JSON dictionary to a scene dictionary compatible with the nuPlan devkit.
    
    Args:
        graph_data (Dict[str, Any]): The graph JSON dictionary.
        
    Returns:
        Dict[str, Any]: The scene dictionary.
    """
    # Step 1: Initialize the scene dictionary
    scene = {
        "map": {"area": "us-ma-boston"},
        "ego": {},
        "world": {'vehicles':[], 'bicycles':[], 'pedestrians':[]},
        "ego_future_states": [],
        "world_future_states": [],
        "goal": {},
        "expected": []
    }

    ego_nodes = graph_data.get("nodes", {}).get("ego", [])
    if not ego_nodes:
        return scene

    # Step 2: Set the initial ego state
    initial_ego_node = ego_nodes[0]["features"]
    scene["ego"] = {
        "time_us": 0,
        "pose": [initial_ego_node["x"], initial_ego_node["y"], initial_ego_node["heading"]],
        "velocity": [initial_ego_node["vx"], initial_ego_node["vy"]],
        "acceleration": [initial_ego_node["ax"], initial_ego_node["ay"]]
    }

    # Step 3: Set the expert ego states
    expert_ego_states = []
    for i, ego_node in enumerate(ego_nodes):
        features = ego_node["features"]
        expert_ego_states.append({
            "time_us": i * 100000,
            "pose": [features["x"], features["y"], features["heading"]],
            "velocity": [features["vx"], features["vy"]],
            "acceleration": [features["ax"], features["ay"]]
        })
    scene["expert_ego_states"] = expert_ego_states

    scene["ego_future_states"] = expert_ego_states[1:]

    # Step 4: Set the world states
    actor_nodes = graph_data.get("nodes", {}).get("actor", [])
    if actor_nodes:
        for actor_node in actor_nodes:
            features = actor_node['features']
            scene['world']['vehicles'].append({
                'id': actor_node['id'],
                'box': {
                    'pose': [features['x'], features['y'], features['heading']],
                    'size': [features['width'], features['length']]
                },
                'speed': np.linalg.norm([features['vx'], features['vy']])
            })

    for i in range(len(expert_ego_states) - 1):
        scene['world_future_states'].append({'vehicles':[], 'bicycles':[], 'pedestrians':[]})

    return scene


def extract_ground_truth(graph_file_path):
    """
    Extracts ground truth risk labels for a single NuPlan graph JSON file.

    Args:
        graph_file_path (str): Path to the graph JSON file.
        
    Returns:
        Dict[str, Any]: A dictionary containing the extracted ground truth labels.
    """
    # Step 1: Load the graph data
    with open(graph_file_path, "r") as f:
        graph_data = json.load(f)

    scene = _create_scene_from_graph(graph_data)

    history, scenario = build_mock_history_scenario_test(scene)

    # Step 2: Compute the metrics
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    _ = ego_lane_change_metric.compute(history, scenario)

    no_ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric
    )
    collision_results = no_ego_at_fault_collisions_metric.compute(history, scenario)

    time_to_collision_metric = TimeToCollisionStatistics(
        'time_to_collision_statistics',
        'Planning',
        ego_lane_change_metric,
        no_ego_at_fault_collisions_metric,
        time_step_size=0.1,
        time_horizon=5.0,
        least_min_ttc=1.0,
    )
    ttc_results = time_to_collision_metric.compute(history, scenario)

    # Step 3: Return the ground truth
    ground_truth = {
        "collisions": {
            "statistics": [stat.serialize() for stat in collision_results[0].statistics],
            "time_series": collision_results[0].time_series.serialize() if collision_results[0].time_series else None,
        },
        "time_to_collision": {
            "statistics": [stat.serialize() for stat in ttc_results[0].statistics],
            "time_series": ttc_results[0].time_series.serialize() if ttc_results[0].time_series else None,
        },
    }

    return ground_truth

def analyze_directory(directory_path):
    """
    Runs the ground truth risk analysis for all JSON files in a directory and returns a DataFrame.

    Args:
        directory_path (str): Path to the directory containing the graph JSON files.
        
    Returns:
        pd.DataFrame: A pandas DataFrame with the analysis results.
    """
    # Step 1: Get all the graph files
    graph_files = glob.glob(os.path.join(directory_path, '*.json'))
    results = []

    # Step 2: Analyze each file
    for graph_file in tqdm(graph_files, desc="Analyzing files"):
        try:
            ground_truth = extract_ground_truth(graph_file)
            
            collision_stat = ground_truth['collisions']['statistics'][0]['value']
            min_ttc = ground_truth['time_to_collision']['statistics'][0]['value']

            risk_level = 'low'
            if not collision_stat:
                risk_level = 'high'
            elif min_ttc < 2:
                risk_level = 'high'
            elif min_ttc < 5:
                risk_level = 'medium'
            
            results.append({
                'file': os.path.basename(graph_file),
                'collision': not collision_stat,
                'min_ttc': min_ttc,
                'risk_level': risk_level
            })
        except Exception as e:
            print(f"Error processing {graph_file}: {e}")

    return pd.DataFrame(results)
