import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import extract_frames

class RiskAnalysis:
    """
    A class for performing risk analysis on traffic scenes.
    """
    def __init__(self, config=None):
        """
        Initializes the risk analysis class.
        
        Args:
            config (dict, optional): A dictionary of configuration parameters. Defaults to None.
        """
        if config is None:
            config = {}

        # Step 1: Default Environment Values
        self.default_temp_c = config.get('default_temp_c', 20.0)
        self.default_precip_mm = config.get('default_precip_mm', 0.0)
        self.default_weather_code = config.get('default_weather_code', 0)
        self.default_is_daylight = config.get('default_is_daylight', True)
        self.default_cloud_cover = config.get('default_cloud_cover', 0.0)
        
        # Step 2: General Physics & Driver Model
        self.g = config.get('g', 9.81)
        self.mu0 = config.get('mu0', 0.9)
        self.T_react = config.get('T_react', 1.0)
        
        # Step 3: Scene Geometry
        self.lane_width = config.get('lane_width', 3.7)

        # Step 4: Weather Code Mapping
        self.weather_code_map = config.get('weather_code_map', {
            0: ("Clear", 1.0, 1.0, 1.0),
            1: ("Overcast", 1.0, 0.9, 1.0),
            2: ("Raining", 1.2, 0.6, 0.7),
            3: ("Snow", 1.4, 0.5, 0.5),
            4: ("Fog", 1.3, 0.4, 0.9),
        })
        self.default_weather_props = ("Unknown", 1.1, 0.8, 0.8)

        # Step 5: Env Hazard Multiplier Config
        self.illum_risk_night = config.get('illum_risk_night', 1.2)
        self.precip_risk_light = config.get('precip_risk_light', 1.15)
        self.precip_risk_heavy = config.get('precip_risk_heavy', 1.35)
        self.precip_thresh_light = config.get('precip_thresh_light', 0.05)
        self.precip_thresh_heavy = config.get('precip_thresh_heavy', 2.0)
        self.temp_risk_cold = config.get('temp_risk_cold', 1.4)
        self.temp_risk_freezing = config.get('temp_risk_freezing', 1.2)
        self.temp_thresh_cold = config.get('temp_thresh_cold', -5.0)
        self.temp_thresh_freezing = config.get('temp_thresh_freezing', 0.0)
        self.w_illum = config.get('w_illum', 0.2)
        self.w_precip = config.get('w_precip', 0.3)
        self.w_temp = config.get('w_temp', 0.2)
        self.w_weather = config.get('w_weather', 0.3)
        self.max_env_hazard = config.get('max_env_hazard', 2.0)

        # Step 6: Risk Model Config
        self.ego_min_speed_for_pet = config.get('ego_min_speed_for_pet', 2.0)
        self.pet_half_life = config.get('pet_half_life', 1.0)
        self.pet_k_steepness = config.get('pet_k_steepness', 2.0)
        self.max_pet_risk_contribution = config.get('max_pet_risk_contribution', 0.7)
        self.visibility_react_scalar = config.get('visibility_react_scalar', 1.5)

        # Step 7: Visibility/Friction Config
        self.precip_vis_dampening = config.get('precip_vis_dampening', 0.1)
        self.light_factor_night = config.get('light_factor_night', 0.8)
        self.min_visibility_factor = config.get('min_visibility_factor', 0.1)
        
        self.precip_fric_dampening = config.get('precip_fric_dampening', 0.05)
        self.fric_ice_keyword = config.get('fric_ice_keyword', 0.4)
        self.fric_snow_keyword = config.get('fric_snow_keyword', 0.5)
        self.min_friction_factor = config.get('min_friction_factor', 0.1)


    def env_hazard_multiplier(self, env_features):
        """
        Calculates the environmental hazard multiplier.
        """
        # Step 1: Get the environmental features
        temp_C = float(env_features.get("temperature_C", self.default_temp_c))
        precip = float(env_features.get("precipitation_mm", self.default_precip_mm))
        code = int(env_features.get("weather_code", self.default_weather_code))
        daylight = bool(env_features.get("is_daylight", self.default_is_daylight))

        # Step 2: Calculate the risk from each environmental factor
        illum_risk = 1.0 if daylight else self.illum_risk_night

        if precip <= self.precip_thresh_light:
            precip_risk = 1.0
        elif precip <= self.precip_thresh_heavy:
            precip_risk = self.precip_risk_light
        else:
            precip_risk = self.precip_risk_heavy

        if temp_C <= self.temp_thresh_cold:
            temp_risk = self.temp_risk_cold
        elif temp_C <= self.temp_thresh_freezing:
            temp_risk = self.temp_risk_freezing
        else:
            temp_risk = 1.0

        weather_risk = self.weather_code_map.get(code, self.default_weather_props)[1]

        # Step 3: Calculate the environmental hazard multiplier
        E_hazard = (
            1.0
            + self.w_illum   * (illum_risk   - 1.0)
            + self.w_precip  * (precip_risk  - 1.0)
            + self.w_temp    * (temp_risk    - 1.0)
            + self.w_weather * (weather_risk - 1.0)
        )

        E_hazard = float(np.clip(E_hazard, 1.0, self.max_env_hazard))
        return E_hazard

    def compute_risk(self, frame, friction_factor, visibility_factor):
        """
        Computes the risk for a given frame.
        """
        # Step 1: Get the ego vehicle's state
        ego = frame['ego']['features']
        ego_pos_2d = np.array([ego['x'], ego['y']])
        ego_vel_2d = np.array([ego['vx'], ego['vy']])
        ego_speed = np.linalg.norm(ego_vel_2d)

        diagnostics = []

        if 'heading' in ego and np.isfinite(ego['heading']):
            heading_vec = np.array([np.cos(ego['heading']), np.sin(ego['heading'])])
        else:
            vnorm = np.linalg.norm(ego_vel_2d)
            heading_vec = ego_vel_2d / (vnorm + 1e-6)

        all_entities = (
            frame.get('vehicles', []) + 
            frame.get('pedestrians', [])
        )

        T_react_scale = (1.0 + (1.0 - visibility_factor) * self.visibility_react_scalar)
        T_react_eff = self.T_react * T_react_scale

        # Step 2: Compute the risk for each agent
        for agent in all_entities:
            features = agent['features']
            pos = np.array([features['x'], features['y']])
            vel = np.array([features['vx'], features['vy']])

            rel_pos = pos - ego_pos_2d
            rel_vel = vel - ego_vel_2d

            dist_long = np.dot(rel_pos, heading_vec)
            rel_speed_long = np.dot(rel_vel, heading_vec)

            lateral_vec = np.array([-heading_vec[1], heading_vec[0]])
            dist_lat = abs(np.dot(rel_pos, lateral_vec))
            
            ttc = np.inf
            if dist_lat < (self.lane_width / 2.0):
                if dist_long > 0 and rel_speed_long < 0:
                    ttc = dist_long / (-rel_speed_long + 1e-6)
            
            drac = 0.0
            if dist_lat < (self.lane_width / 2.0):
                v_e_long = float(np.dot(ego_vel_2d, heading_vec))
                v_a_long = float(np.dot(vel, heading_vec))
                u = v_e_long - v_a_long
                if dist_long > 0 and u > 0:
                    drac = (u * u) / (2.0 * (dist_long + 1e-6))

            pet = np.inf
            try:
                A = np.column_stack((ego_vel_2d, -vel))
                b = pos - ego_pos_2d
                if np.linalg.matrix_rank(A) < 2: raise np.linalg.LinAlgError
                t_e, t_o = np.linalg.solve(A, b)
                if t_e > 0 and t_o > 0:
                    pet = abs(t_o - t_e)
            except np.linalg.LinAlgError:
                pass 

            R_ttc = 0.0
            max_decel = self.g * self.mu0 * friction_factor
            if max_decel > 1e-6 and ego_speed > 1e-6:
                t_critical = T_react_eff + ego_speed / max_decel
                alpha = 1.0 / t_critical
                if np.isfinite(ttc): R_ttc = np.exp(-alpha * ttc)

            beta = 1.0 / (self.g * self.mu0 * friction_factor + 1e-6)
            R_drac = 1 - np.exp(-beta * drac) if drac > 0 else 0.0

            R_pet = 0.0
            if ego_speed > self.ego_min_speed_for_pet and np.isfinite(pet):
                R_pet_raw = 1.0 / (1.0 + (pet / self.pet_half_life)**self.pet_k_steepness)
                R_pet = min(R_pet_raw, self.max_pet_risk_contribution)

            risk_i = 1 - (1 - R_ttc) * (1 - R_drac) * (1 - R_pet)
            
            diagnostics.append({
                "id": agent["id"],
                "distance": dist_long,
                "rel_speed": rel_speed_long,
                "ttc": ttc,
                "drac": drac,
                "pet": pet,
                "risk_i": risk_i,
                "r_ttc": R_ttc,
                "r_drac": R_drac,
                "r_pet": R_pet
            })

        if not diagnostics:
            return 0.0, []
        
        # Step 3: Compute the scene risk
        R_base = max(d["risk_i"] for d in diagnostics)
        
        env_features = frame.get("environment", {}).get("features", {})
        E_hazard = self.env_hazard_multiplier(env_features)

        R_scene = R_base * E_hazard

        return R_scene, diagnostics

    def compute_visibility_factor(self, env):
        """
        Computes the visibility factor.
        """
        # Step 1: Get the environmental features
        f = env.get("features", {})
        code = int(f.get("conditions", self.default_weather_code))
        precip = float(f.get("precipitation", self.default_precip_mm))
        daylight = bool(f.get("daylight", self.default_is_daylight))

        # Step 2: Compute the visibility factor
        props = self.weather_code_map.get(code, self.default_weather_props)
        base = props[2]

        precip_factor = 1.0 / (1.0 + self.precip_vis_dampening * precip)

        light_factor = 1.0 if daylight else self.light_factor_night

        visibility_factor = base * precip_factor * light_factor

        return float(np.clip(visibility_factor, self.min_visibility_factor, 1.0))

    def compute_friction_factor(self, env):
        """
        Calculates the friction factor.
        """
        # Step 1: Get the environmental features
        f = env.get("features", {})
        code = int(f.get("conditions", self.default_weather_code))
        precip = float(f.get("precipitation", self.default_precip_mm))
        desc = {0:'clear',1:'overcast',2:'raining',3:'snow',4:'fog'}[code]

        # Step 2: Compute the friction factor
        props = self.weather_code_map.get(code, self.default_weather_props)
        base = props[3]

        precip_factor = 1.0 / (1.0 + self.precip_fric_dampening * precip)

        if "ice" in desc or "freez" in desc:
            base = min(base, self.fric_ice_keyword)
        elif "snow" in desc:
            base = min(base, self.fric_snow_keyword)

        friction_factor = base * precip_factor

        return float(np.clip(friction_factor, self.min_friction_factor, 1.0))

    def analyze_frame(self, frame):
        """
        Analyzes a single frame.
        
        Args:
            frame (dict): The frame to analyze.
            
        Returns:
            tuple: A tuple containing the scene risk, diagnostics, visibility factor, and friction factor.
        """
        # Step 1: Compute the visibility and friction factors
        env = frame.get("environment", {})
        visibility_factor = self.compute_visibility_factor(env)
        friction_factor = self.compute_friction_factor(env)

        # Step 2: Compute the risk
        R_scene, diagnostics = self.compute_risk(
            frame, friction_factor, visibility_factor
        )

        return R_scene, diagnostics, visibility_factor, friction_factor

    def collect_risk_data(self, directory_path, num_episodes):
        """
        Collects risk data from a directory of graph files.
        
        Args:
            directory_path (str): The path to the directory containing the graph files.
            num_episodes (int): The number of episodes to process.
            
        Returns:
            pd.DataFrame: A DataFrame containing the risk data.
        """
        # Step 1: Initialize the results list
        results = []

        # Step 2: Process each episode
        for ep in tqdm(range(num_episodes)):
            file_path = os.path.join(directory_path, f"{ep}_graph.json")
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                graph = json.load(f)

            frames = extract_frames(graph)
            if not frames:
                continue

            risks, vis_list, fric_list, diagnostics_list = [], [], [], []
            ego_speeds, precips, num_agents_list = [], [], []

            # Step 3: Process each frame
            for i, frame in enumerate(frames):
                try:
                    R, diag, vis_f, fric_f = self.analyze_frame(frame)

                    risks.append(R)
                    vis_list.append(vis_f)
                    fric_list.append(fric_f)
                    diagnostics_list.append(diag) 

                    ego_speed = np.linalg.norm([
                        frame["ego"]["features"]["vx"],
                        frame["ego"]["features"]["vy"]
                    ])
                    ego_speeds.append(ego_speed)

                    env = frame.get("environment", {})
                    precip = env.get("features", {}).get("precipitation_mm", 0.0)
                    precips.append(precip)

                    num_agents = (
                        len(frame.get("vehicles", [])) +
                        len(frame.get("pedestrians", []))
                    )
                    num_agents_list.append(num_agents)

                except Exception as e:
                    continue

            if not risks:
                continue

            risks = np.array(risks)
            vis_list = np.array(vis_list)
            fric_list = np.array(fric_list)
            ego_speeds = np.array(ego_speeds)
            precips = np.array(precips)
            num_agents_list = np.array(num_agents_list)

            max_idx = int(np.argmax(risks))
            min_idx = int(np.argmin(risks))

            diag_at_max_risk = diagnostics_list[max_idx]
            
            agent_id_max_risk = None
            ttc_at_max_risk = np.inf
            drac_at_max_risk = 0.0
            pet_at_max_risk = np.inf
            r_ttc_at_max_risk = 0.0
            r_drac_at_max_risk = 0.0
            r_pet_at_max_risk = 0.0
            
            if diag_at_max_risk: 
                agent_diag = max(diag_at_max_risk, key=lambda d: d['risk_i'])
                
                agent_id_max_risk = agent_diag["id"]
                ttc_at_max_risk = agent_diag["ttc"]
                drac_at_max_risk = agent_diag["drac"]
                pet_at_max_risk = agent_diag["pet"]
                r_ttc_at_max_risk = agent_diag["r_ttc"]
                r_drac_at_max_risk = agent_diag["r_drac"]
                r_pet_at_max_risk = agent_diag["r_pet"]

            # Step 4: Append the results
            result = {
                "episode_num": ep,
                "mean_risk": float(np.mean(risks)),
                "max_risk": float(risks[max_idx]),
                "min_risk": float(risks[min_idx]),
                "frame_num_max_risk": max_idx,
                "frame_num_min_risk": min_idx,
                
                "agent_id_max_risk": agent_id_max_risk,
                "ttc_at_max_risk": float(ttc_at_max_risk),
                "drac_at_max_risk": float(drac_at_max_risk),
                "pet_at_max_risk": float(pet_at_max_risk),
                "r_ttc_at_max_risk": float(r_ttc_at_max_risk),
                "r_drac_at_max_risk": float(r_drac_at_max_risk),
                "r_pet_at_max_risk": float(r_pet_at_max_risk),
                
                "visibility_factor_max_risk": float(vis_list[max_idx]),
                "visibility_factor_min_risk": float(vis_list[min_idx]),
                "friction_factor_max_risk": float(fric_list[max_idx]),
                "friction_factor_min_risk": float(fric_list[min_idx]),
                "ego_speed_max_risk": float(ego_speeds[max_idx]),
                "ego_speed_min_risk": float(ego_speeds[min_idx]),
                "precipitation_max_risk": float(precips[max_idx]),
                "precipitation_min_risk": float(precips[min_idx]),
                "num_agents_max_risk": int(num_agents_list[max_idx]),
                "num_agents_min_risk": int(num_agents_list[min_idx]),
            }
            results.append(result)

        return pd.DataFrame(results)
