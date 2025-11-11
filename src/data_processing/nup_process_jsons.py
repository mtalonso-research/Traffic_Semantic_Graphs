from pyproj import CRS, Transformer
import json, os
from tqdm import tqdm
from datetime import datetime, timezone
import time, re, requests
from dateutil.parser import isoparse
from zoneinfo import ZoneInfo

MAP_ORIGINS = {
    "pittsburgh": (40.440624, -79.995888, 0.0),
    "singapore": (1.3521, 103.8198, 0.0),
    "las_vegas": (36.1699, -115.1398, 0.0),
    "boston": (42.3601, -71.0589, 0.0),
}

def build_transformer(lat0, lon0, alt0=0.0):
    """
    Builds a transformer to convert from local ENU coordinates to geodetic coordinates.
    
    Args:
        lat0 (float): The latitude of the origin of the local ENU frame.
        lon0 (float): The longitude of the origin of the local ENU frame.
        alt0 (float, optional): The altitude of the origin of the local ENU frame. Defaults to 0.0.
        
    Returns:
        pyproj.Transformer: A transformer to convert from local ENU coordinates to geodetic coordinates.
    """
    geodetic = CRS.from_epsg(4979)
    local_enu = CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
    )
    return Transformer.from_crs(local_enu, geodetic, always_xy=True)

def add_latlon_to_graphs(json_dir, out_dir=None, map_region="pittsburgh", episodes=None):
    """
    Adds latitude, longitude, and altitude to the nodes in the graph JSON files.
    
    Args:
        json_dir (str): The directory containing the graph JSON files.
        out_dir (str, optional): The directory where the updated JSON files will be saved. Defaults to None.
        map_region (str, optional): The map region to use for the coordinate transformation. Defaults to "pittsburgh".
        episodes (list, optional): A list of episode numbers to process. Defaults to None.
    """
    # Step 1: Initialize variables
    if map_region not in MAP_ORIGINS:
        raise ValueError(f"Unknown map_region: {map_region}")

    lat0, lon0, alt0 = MAP_ORIGINS[map_region]
    transformer = build_transformer(lat0, lon0, alt0)

    out_dir = out_dir or json_dir
    os.makedirs(out_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if episodes:
            episodes_to_process = set(episodes)
            json_files = [f for f in json_files if int(os.path.splitext(f)[0].split('_')[0]) in episodes_to_process]

    # Step 2: Process each JSON file
    for fname in tqdm(json_files, desc=f"Adding lat/lon/alt for {map_region}"):
        in_path = os.path.join(json_dir, fname)
        with open(in_path, "r") as f:
            graph = json.load(f)

        # Step 3: Process each node
        for ntype, nodes in graph.get("nodes", {}).items():
            for node in nodes:
                feat = node.get("features", {})
                x, y, z = feat.get("x"), feat.get("y"), feat.get("z")
                if x is None or y is None:
                    continue
                lon, lat, alt = transformer.transform(x, y, z or 0.0)
                feat["latitude"] = float(lat)
                feat["longitude"] = float(lon)
                feat["altitude"] = float(alt)
                node["features"] = feat

        # Step 4: Write the updated graph to a new file
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

_IDNUM_RE = re.compile(r".*_(\d+)$")

def _idx_from_id(node_id):
    """
    Extracts the index from a node ID.
    """
    m = _IDNUM_RE.match(node_id or "")
    return int(m.group(1)) if m else None

def fetch_weather_features(lat, lon, timestamp_raw_us):
    """
    Fetches weather features from the Open-Meteo API.
    
    Args:
        lat (float): The latitude.
        lon (float): The longitude.
        timestamp_raw_us (int): The timestamp in microseconds.
        
    Returns:
        dict: A dictionary of weather features.
    """
    # Step 1: Initialize variables
    ts = float(timestamp_raw_us) / 1e6
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,cloudcover,precipitation,weathercode,is_day"
        f"&timezone=UTC"
    )
    # Step 2: Fetch data from the API
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as http_err:
        if r.status_code == 429:
            print(f"Failed due to rate limiting (429 Too Many Requests).")
        else:
            print(f"HTTP error occurred: {http_err} - Status Code: {r.status_code}")
        return {}
    except requests.exceptions.RequestException as req_err:
        print(f"Request failed (e.g., timeout, network issue): {req_err}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        return {}

    times = data.get("hourly", {}).get("time", [])
    if not times:
        print('failed due to no times')
        return {}

    # Step 3: Find the nearest weather data to the given timestamp
    dt_list = [isoparse(t).replace(tzinfo=timezone.utc) for t in times]
    diffs = [abs((dti - dt).total_seconds()) for dti in dt_list]
    idx = int(min(range(len(diffs)), key=lambda i: diffs[i]))

    def g(key, cast=float):
        vals = data.get("hourly", {}).get(key, [])
        return cast(vals[idx]) if idx < len(vals) else None

    # Step 4: Return the weather features
    return {
        "precipitation_mm": g("precipitation", float),
        "weather_code": g("weathercode", int),
        "is_daylight": bool(data.get("hourly", {}).get("is_day", [0])[idx]),
    }

def enrich_weather_features(json_dir, out_dir=None, sleep_s=0.15, episodes=None):
    """
    Enriches the environment nodes in the graph JSON files with weather features.
    
    Args:
        json_dir (str): The directory containing the graph JSON files.
        out_dir (str, optional): The directory where the updated JSON files will be saved. Defaults to None.
        sleep_s (float, optional): The number of seconds to sleep between API requests. Defaults to 0.15.
        episodes (list, optional): A list of episode numbers to process. Defaults to None.
    """
    # Step 1: Initialize variables
    out_dir = out_dir or json_dir
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    if episodes:
            episodes_to_process = set(episodes)
            files = [f for f in files if int(os.path.splitext(f)[0].split('_')[0]) in episodes_to_process]

    # Step 2: Process each JSON file
    for fname in tqdm(files, desc="Enriching env nodes with weather"):
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            g = json.load(f)

        env_nodes = g.get("nodes", {}).get("environment", [])
        if env_nodes and 'weather_code' in env_nodes[0].get("features", {}):
            continue

        ego_nodes = g.get("nodes", {}).get("ego", [])
        env_nodes = g.get("nodes", {}).get("environment", [])

        ego_latlon = {}
        for n in ego_nodes:
            i = _idx_from_id(n.get("id", ""))
            if i is None:
                continue
            feat = n.get("features", {})
            lat, lon = feat.get("latitude"), feat.get("longitude")
            if lat is not None and lon is not None:
                ego_latlon[i] = (float(lat), float(lon))

        # Step 3: Process each environment node
        for env in env_nodes:
            feat = env.get("features", {})
            ts_raw = feat.get("timestamp_raw")
            lat, lon = feat.get("latitude"), feat.get("longitude")

            if lat is None or lon is None:
                i = _idx_from_id(env.get("id", ""))
                if i is not None and i in ego_latlon:
                    lat, lon = ego_latlon[i]
                    feat["latitude"], feat["longitude"] = lat, lon

            if ts_raw is None or lat is None or lon is None:
                continue

            weather = fetch_weather_features(lat, lon, ts_raw)
            if weather:
                feat.update(weather)
                env["features"] = feat

            time.sleep(sleep_s)

        # Step 4: Write the updated graph to a new file
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(g, f, indent=2)

WMO_WEATHER_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm (slight or moderate)",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

def replace_weather_code_with_description(json_dir, out_dir=None, node_type="environment", remove_numeric=False):
    """
    Replaces the weather code in the graph JSON files with a human-readable description.
    
    Args:
        json_dir (str): The directory containing the graph JSON files.
        out_dir (str, optional): The directory where the updated JSON files will be saved. Defaults to None.
        node_type (str, optional): The type of node to process. Defaults to "environment".
        remove_numeric (bool, optional): Whether to remove the numeric weather code after replacement. Defaults to False.
    """
    # Step 1: Initialize variables
    out_dir = out_dir or json_dir
    os.makedirs(out_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    # Step 2: Process each JSON file
    for fname in tqdm(json_files, desc="Replacing weather codes"):
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r") as f:
            graph = json.load(f)

        # Step 3: Process each node
        nodes = graph.get("nodes", {}).get(node_type, [])
        for n in nodes:
            feat = n.get("features", {})
            code = feat.get("weather_code")

            if isinstance(code, (int, float)):
                desc = WMO_WEATHER_MAP.get(int(code), f"Unknown ({code})")
                feat["weather_description"] = desc
                if remove_numeric:
                    feat.pop("weather_code", None)
                n["features"] = feat

        # Step 4: Write the updated graph to a new file
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

    print(f"✅ Weather code replacement complete for {len(json_files)} files.")

def add_temporal_features(json_dir, out_dir=None, node_type="environment", city="boston"):
    """
    Adds temporal features to the nodes in the graph JSON files.
    
    Args:
        json_dir (str): The directory containing the graph JSON files.
        out_dir (str, optional): The directory where the updated JSON files will be saved. Defaults to None.
        node_type (str, optional): The type of node to process. Defaults to "environment".
        city (str, optional): The city to use for the timezone conversion. Defaults to "boston".
    """
    # Step 1: Initialize variables
    city_to_timezone = {
        "boston": "America/New_York",
        "pittsburgh": "America/New_York",
        "singapore": "Asia/Singapore",
        "las_vegas": "America/Los_Angeles"
    }
    
    local_timezone = ZoneInfo(city_to_timezone.get(city, "UTC"))

    out_dir = out_dir or json_dir
    os.makedirs(out_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    # Step 2: Process each JSON file
    for fname in tqdm(json_files, desc=f"Adding temporal features to {node_type} nodes"):
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r") as f:
            graph = json.load(f)

        # Step 3: Process each node
        nodes = graph.get("nodes", {}).get(node_type, [])
        for n in nodes:
            feat = n.get("features", {})
            ts_raw = feat.get("timestamp_raw")
            if ts_raw is None:
                continue

            dt_utc = datetime.fromtimestamp(ts_raw / 1e6, tz=timezone.utc)
            
            dt_local = dt_utc.astimezone(local_timezone)

            feat.update({
                "month": dt_local.month,
                "day_of_week": dt_local.strftime("%A"),
                "time_of_day": dt_local.strftime("%H:%M:%S")
            })
            n["features"] = feat

        # Step 4: Write the updated graph to a new file
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

    print(f"✅ Temporal enrichment complete for {len(json_files)} scene files.")
    