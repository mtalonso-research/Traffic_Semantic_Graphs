"""Build a separate, database-grouped 70/10/20 city view and frozen manifests.

This does not alter the legacy city views.  A database group is used because the
canonical corpus has one graph per scene token; keeping an entire log database
together is therefore the stricter practical leakage boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")


def risk_class(value: float) -> int:
    return 3 if value > 0.3442 else 2 if value > 0.1008 else 1 if value > 0.0043 else 0


def digest(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def load_city(graph_root: Path, risks: dict[str, float], city: str, inventory: dict | None = None) -> list[dict]:
    if inventory is not None:
        key = f"nuplan_{city}"
        records = inventory["cities"][key]["scene_records"]
        rows = []
        for record in records:
            sample_ids = record["sample_ids"]
            if len(sample_ids) != 1:
                raise ValueError(f"Inventory no longer supports one-window scene assumption: {record['scene_id']}")
            sample_id = sample_ids[0]
            path = graph_root / f"{sample_id}_graph.json"
            if not path.is_file() or sample_id not in risks:
                raise ValueError(f"Canonical graph/risk missing for inventory sample {sample_id}")
            rows.append({"sample_id": sample_id, "scene_id": record["scene_id"],
                         "database_id": record["database_id"], "label": int(next(iter(record["label_counts"]))),
                         "path": path})
        return rows
    rows = []
    for path in sorted(graph_root.glob("*_graph.json")):
        sample_id = path.name.split("_")[0]
        metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
        graph_city = metadata.get("city", "").split("_")[-1].lower()
        if graph_city != city:
            continue
        if sample_id not in risks:
            raise ValueError(f"Missing risk score for {sample_id}")
        scene = metadata.get("scene_token") or metadata.get("scene_name")
        database = metadata.get("db_file")
        if not scene or not database:
            raise ValueError(f"Missing scene/database metadata for {path}")
        rows.append({"sample_id": sample_id, "scene_id": scene, "database_id": database,
                     "label": risk_class(risks[sample_id]), "path": path})
    if not rows:
        raise ValueError(f"No graphs found for city {city}")
    return rows


def assign_groups(rows: list[dict], seed: int, city: str, ratios: dict[str, float]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["database_id"]].append(row)
    # A predeclared deterministic hash order avoids inspecting outcomes to pick a split.
    ordered = sorted(groups.items(), key=lambda pair: hashlib.sha256(
        f"scene-aware-nested-anchor-v1|{seed}|{city}|{pair[0]}".encode()).hexdigest())
    desired = {name: len(rows) * ratio for name, ratio in ratios.items()}
    current = {name: 0 for name in SPLITS}
    assigned = {name: [] for name in SPLITS}
    for _database, group in ordered:
        # Put each whole database into the most under-filled split.  No label-based
        # reassignment is performed.
        choice = min(SPLITS, key=lambda name: (current[name] / desired[name], SPLITS.index(name)))
        assigned[choice].extend(group)
        current[choice] += len(group)
    return assigned


def summarize(parts: dict[str, list[dict]]) -> dict:
    result = {}
    for name, rows in parts.items():
        scenes = {row["scene_id"] for row in rows}
        databases = {row["database_id"] for row in rows}
        result[name] = {"sample_count": len(rows), "scene_count": len(scenes),
                        "database_count": len(databases),
                        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
                        "sample_ids_sha256": digest(sorted(row["sample_id"] for row in rows)),
                        "database_ids_sha256": digest(sorted(databases))}
    return result


def assert_leakage_free(parts: dict[str, list[dict]]) -> None:
    for key in ("sample_id", "scene_id", "database_id"):
        sets = {name: {row[key] for row in rows} for name, rows in parts.items()}
        for left in SPLITS:
            for right in SPLITS:
                if left < right and sets[left] & sets[right]:
                    raise AssertionError(f"{key} leakage between {left} and {right}")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        os.symlink(os.path.relpath(src, dst.parent), dst)


def materialize(view: Path, source_parts: dict, target_parts: dict, source_risks: dict, target_risks: dict) -> None:
    for phase, names in (("training_data", ("train", "validation")), ("evaluation_data", ("test",))):
        for alias, parts, risks in (("clean", source_parts, source_risks), ("noisy_0", target_parts, target_risks), ("noisy_true", target_parts, target_risks)):
            rows = [row for name in names for row in parts[name]]
            root = view / phase / alias
            graph_root = root / "graphs"
            graph_root.mkdir(parents=True, exist_ok=True)
            selected = {}
            for row in rows:
                link(row["path"], graph_root / row["path"].name)
                selected[row["sample_id"]] = risks[row["sample_id"]]
            (root / "risk_scores.json").write_text(json.dumps(dict(sorted(selected.items())), indent=2) + "\n", encoding="utf-8")
            if phase == "evaluation_data" and alias in ("noisy_0", "noisy_true"):
                (root / "risk_scores_true.json").write_text(json.dumps(dict(sorted(selected.items())), indent=2) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    graph_root = Path(args.graph_root)
    risks = json.loads(Path(args.risks).read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8")) if args.inventory else None
    source = load_city(graph_root, risks, args.source_city, inventory)
    target = load_city(graph_root, risks, args.target_city, inventory)
    ratio_values = [float(value) / 100 for value in args.split_proportions.split(",")]
    if len(ratio_values) != 3 or any(value <= 0 for value in ratio_values) or abs(sum(ratio_values) - 1.0) > 1e-9:
        raise ValueError("--split-proportions must be three positive percentages summing to 100, e.g. 70,10,20")
    ratios = dict(zip(SPLITS, ratio_values))
    # Use the canonical inventory city identifier in the hash, so candidate
    # comparisons and frozen manifests share exactly the same assignment.
    source_parts = assign_groups(source, args.split_seed, f"nuplan_{args.source_city}", ratios)
    target_parts = assign_groups(target, args.split_seed, f"nuplan_{args.target_city}", ratios)
    assert_leakage_free(source_parts)
    assert_leakage_free(target_parts)
    summary = {"protocol": "scene_aware_nested_anchor_v1", "split_seed": args.split_seed,
               "split_proportions": ratios, "source": summarize(source_parts), "target": summarize(target_parts),
               "leakage_checks": {"sample_scene_database_overlap": False, "status": "passed"}}
    if args.dry_run:
        print(json.dumps({"status": "candidate-only; no manifests or view written", "summary": summary}, indent=2))
        return
    output = Path(args.output_root)
    view = output / "city_views" / f"{args.source_city}_to_{args.target_city}"
    manifest_root = output / "manifests" / f"{args.source_city}_to_{args.target_city}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing protocol output: {output}")
    manifest_root.mkdir(parents=True)
    manifest = {"protocol": "scene_aware_nested_anchor_v1", "split_seed": args.split_seed,
                "split_proportions": ratios, "grouping_key": "metadata.db_file",
                "scene_key": "metadata.scene_token", "grouping_rationale": "Each scene token occurs once in this corpus; database/log grouping is stricter and prevents cross-split log leakage.",
                "source_city": args.source_city, "target_city": args.target_city,
                "source_train": sorted(row["sample_id"] for row in source_parts["train"]),
                "source_validation": sorted(row["sample_id"] for row in source_parts["validation"]),
                "source_evaluation": sorted(row["sample_id"] for row in source_parts["test"]),
                "target_train": sorted(row["sample_id"] for row in target_parts["train"]),
                "target_validation": sorted(row["sample_id"] for row in target_parts["validation"]),
                "target_evaluation": sorted(row["sample_id"] for row in target_parts["test"]),
                "anchor_selection_seed": args.anchor_selection_seed,
                "anchor_selection": "strict nested prefixes of one deterministic uniform-without-replacement target-train permutation"}
    populations = {name: manifest[name] for name in ("source_train", "source_validation", "source_evaluation", "target_train", "target_validation", "target_evaluation")}
    manifest["digests"] = {name: digest(ids) for name, ids in populations.items()}
    (manifest_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (manifest_root / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    materialize(view, source_parts, target_parts, risks, risks)
    (view / "city_view_manifest.json").write_text(json.dumps({"protocol": manifest["protocol"], "manifest": str((manifest_root / "manifest.json").resolve()), "aliases": {"clean": args.source_city, "noisy_0": args.target_city, "noisy_true": args.target_city}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "view": str(view), "manifest": str(manifest_root / "manifest.json"), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--risks", required=True)
    parser.add_argument("--inventory", help="Completed corpus inventory; avoids rescanning graph JSON metadata")
    parser.add_argument("--output-root", default="experiment_results/nested_scene_split/protocol_data")
    parser.add_argument("--source-city", default="singapore")
    parser.add_argument("--target-city", default="boston")
    parser.add_argument("--split-seed", type=int, default=228)
    parser.add_argument("--anchor-selection-seed", type=int, default=228)
    parser.add_argument("--split-proportions", default="70,10,20", help="train,validation,test percentages")
    parser.add_argument("--dry-run", action="store_true", help="Compare a candidate without writing manifests or data views")
    main(parser.parse_args())
