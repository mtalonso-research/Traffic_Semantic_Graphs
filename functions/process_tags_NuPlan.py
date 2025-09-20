import os
import json
import re
from pathlib import Path
import datetime as dt

# ----------------------- Tokenization helpers -----------------------

def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9_]+', '_', s.lower()).strip('_')

def _tokenize_with_ngrams(label_str: str, max_n: int = 3):
    s = _normalize(label_str)
    parts = [p for p in s.split('_') if p]
    toks = set(parts)
    for n in range(2, max_n + 1):
        for i in range(len(parts) - n + 1):
            toks.add('_'.join(parts[i:i+n]))
    toks.add(s)
    return parts, toks

# ----------------------- Inference maps -----------------------

_ACTION_PATTERNS = [
    (("turn", "left"),         "left_turn"),
    (("turn", "right"),        "right_turn"),
    (("u", "turn"),            "u_turn"),
    (("uturn",),               "u_turn"),
    (("merge", "left"),        "merge_left"),
    (("merge", "right"),       "merge_right"),
    (("lane", "change", "left"),  "lane_change_left"),
    (("lane", "change", "right"), "lane_change_right"),
    (("go", "straight"),       "go_straight"),
    (("proceed", "straight"),  "go_straight"),
    (("straight",),            "go_straight"),
]

_STRAIGHT_HINTS = {"traversing", "on", "through", "across", "crossing", "continue"}
_TURN_FAMILIES = {"turn", "lane_change", "merge", "exit", "u_turn"}

_TRAFFIC_LIGHT_HINTS = {
    "traffic_light", "traffic_signal", "signal", "lights",
    "red_light", "yellow_light", "amber_light", "green_light",
    "red", "yellow", "amber", "green"
}
_STOP_SIGN_HINTS = {"stop_sign"}
_YIELD_HINTS     = {"yield_sign", "yield"}
_ROUNDABOUT_HINTS= {"roundabout"}
_STOP_LINE_HINTS = {"stopline", "stop_line", "stop_bar", "stopbar"}

_TRAFFIC_PRIORITY = [
    ("traffic_light", _TRAFFIC_LIGHT_HINTS),
    ("stop_sign",     _STOP_SIGN_HINTS),
    ("yield_sign",    _YIELD_HINTS),
    ("roundabout",    _ROUNDABOUT_HINTS),
    ("stop_line",     _STOP_LINE_HINTS),
    ("unmarked",      {"unmarked"}),
]

_ROAD_FEATURE_ALIASES = {
    "school_zone": {"school_zone"},
    "construction": {"construction", "work_zone"},
    "rail_crossing": {"rail_crossing", "railroad", "level_crossing"},
    "speed_bump": {"speed_bump", "speed_hump", "speed_table"},
    "tunnel": {"tunnel"},
    "bridge": {"bridge", "overpass", "viaduct"},
    "ped_crossing": {"crosswalk", "ped_crossing"},
    "bike_lane": {"bike_lane", "cycle_lane"},
}

# ----------------------- Inference functions -----------------------

def infer_action(label_str: str) -> str:
    parts, partset = _tokenize_with_ngrams(label_str)

    for patt, action in _ACTION_PATTERNS:
        if len(patt) == 1:
            if patt[0] in partset:
                return action
        else:
            for i in range(len(parts) - len(patt) + 1):
                if tuple(parts[i:i+len(patt)]) == patt:
                    return action

    has_left  = "left"  in partset
    has_right = "right" in partset
    family_present = any(fam in partset for fam in _TURN_FAMILIES)
    if family_present and (has_left or has_right):
        if has_left:
            return "left_turn"
        if has_right:
            return "right_turn"

    if _STRAIGHT_HINTS & partset:
        return "go_straight"

    if "stationary" in partset:
        return "stationary"

    return "unknown"

def infer_traffic_control(label_str: str) -> str:
    _, partset = _tokenize_with_ngrams(label_str)
    found = set()
    for canonical, hints in _TRAFFIC_PRIORITY:
        if hints & partset:
            found.add(canonical)
    if not found:
        return "unmarked"
    for canonical, _ in _TRAFFIC_PRIORITY:
        if canonical in found:
            return canonical
    return "unmarked"

def infer_road_and_env(label_str: str):
    _, partset = _tokenize_with_ngrams(label_str)
    road_feats = []
    for canonical, aliases in _ROAD_FEATURE_ALIASES.items():
        if aliases & partset:
            road_feats.append(canonical)
    return road_feats

def infer_environment_from_time(timestamps):
    tags = set()
    if not timestamps:
        return tags
    datetimes = [dt.datetime.utcfromtimestamp(ts/1e6) for ts in timestamps]
    months = {d.strftime("%B").lower() for d in datetimes}
    days   = {d.strftime("%A").lower() for d in datetimes}
    hours  = [d.hour for d in datetimes]
    winter_months = {'november', 'december', 'january', 'february', 'march'}
    if months & winter_months:
        tags.add('winter_conditions_possible')
    if any(h < 6 or h >= 20 for h in hours):
        tags.add('night_time')
    if days - {'saturday','sunday'}:
        if any(7 <= h <= 9 for h in hours) or any(16 <= h <= 18 for h in hours):
            tags.add('rush_hour')
    if days & {'saturday','sunday'}:
        tags.add('weekend')
    if 'night_time' in tags or 'winter_conditions_possible' in tags:
        tags.add('low_visibility_possible')
    if any(10 <= h <= 15 for h in hours):
        tags.add('off_peak_hours')
    return tags

# ----------------------- Episode-level processing -----------------------

def process_tags(src_dir: str, dst_dir: str, overwrite=False):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    processed, skipped, errors = 0, 0, 0

    for p in sorted(src.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            labels = []
            timestamps = []
            for env_node in data.get("nodes", {}).get("environment", []):
                feats = env_node.get("features", {})
                if "frame_labels_all" in feats and feats["frame_labels_all"]:
                    labels.extend(feats["frame_labels_all"])
                elif "frame_label" in feats and feats["frame_label"]:
                    labels.append(feats["frame_label"])
                if "timestamp_raw" in feats:
                    timestamps.append(int(feats["timestamp_raw"]))

            dynamic_actions, stationary_actions = set(), set()
            traffic_controls, road_features = set(), set()

            for lab in labels:
                a = infer_action(lab)
                if a == "stationary":
                    stationary_actions.add(a)
                elif a != "unknown":
                    dynamic_actions.add(a)

                tc = infer_traffic_control(lab)
                if tc != "unmarked":
                    traffic_controls.add(tc)

                rf = infer_road_and_env(lab)
                road_features.update(rf)

            # Priority: dynamic > stationary > unknown
            if dynamic_actions:
                action_tags = sorted(dynamic_actions)
            elif stationary_actions:
                action_tags = ["stationary"]
            else:
                action_tags = ["unknown"]

            if traffic_controls:
                for canonical, _ in _TRAFFIC_PRIORITY:
                    if canonical in traffic_controls:
                        traffic_controls = {canonical}
                        break

            env_tags = infer_environment_from_time(timestamps)

            out_obj = {
                "action_tags": action_tags,
                "traffic_control_tags": sorted(traffic_controls),
                "road_feature_tags": sorted(road_features),
                "environment_tags": sorted(env_tags),
            }

            out_path = dst / p.name
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=4)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {p.name}: {e}")

    print(f"Done. Wrote: {processed}, skipped: {skipped}, errors: {errors}")
