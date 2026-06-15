import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple


SPLITS = ("training_data", "evaluation_data")
VIEW_TARGET_ALIAS = "noisy_0"


def city_from_graph(graph_path: Path) -> str:
    with graph_path.open("r") as f:
        scene = json.load(f)
    city = scene.get("metadata", {}).get("city", "")
    return city.split("_")[-1].lower()


def episode_id(graph_path: Path) -> str:
    return graph_path.name.split("_")[0]


def numeric_graph_sort_key(path: Path):
    token = path.name.split("_")[0]
    return (0, int(token)) if token.isdigit() else (1, path.name)


def sorted_json_dict(values: Dict[str, float]) -> Dict[str, float]:
    def key(item: Tuple[str, float]):
        ep = item[0]
        return (0, int(ep)) if ep.isdigit() else (1, ep)

    return dict(sorted(values.items(), key=key))


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_empty_or_create_dir(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            path.mkdir(parents=True, exist_ok=True)
            return
        safe_remove(path)
    path.mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.relpath(src, dst.parent), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_risk_scores(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Risk scores not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def split_one_city(
    data_root: Path,
    split_name: str,
    city: str,
    link_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, int]:
    source_graph_dir = data_root / split_name / "clean" / "graphs"
    source_risk_path = data_root / split_name / "clean" / "risk_scores.json"
    if not source_graph_dir.is_dir():
        raise FileNotFoundError(f"Clean graph directory not found: {source_graph_dir}")

    risks = load_risk_scores(source_risk_path)
    graph_files = sorted(source_graph_dir.glob("*.json"), key=numeric_graph_sort_key)
    selected = [path for path in graph_files if city_from_graph(path) == city]

    out_root = data_root / split_name / f"clean_{city}"
    out_graph_dir = out_root / "graphs"

    if not dry_run:
        ensure_empty_or_create_dir(out_graph_dir, overwrite=overwrite)

    city_risks = {}
    missing_risks = []
    for graph_path in selected:
        ep = episode_id(graph_path)
        if ep in risks:
            city_risks[ep] = risks[ep]
        else:
            missing_risks.append(ep)

        if not dry_run:
            transfer_file(graph_path, out_graph_dir / graph_path.name, mode=link_mode)

    if not dry_run:
        write_json(out_root / "risk_scores.json", sorted_json_dict(city_risks))
        write_json(
            out_root / "city_split_manifest.json",
            {
                "city": city,
                "split": split_name,
                "source_graph_dir": str(source_graph_dir),
                "source_risk_path": str(source_risk_path),
                "link_mode": link_mode,
                "graph_count": len(selected),
                "risk_count": len(city_risks),
                "missing_risk_count": len(missing_risks),
                "missing_risk_episode_ids": missing_risks[:50],
            },
        )

    return {
        "graphs": len(selected),
        "risks": len(city_risks),
        "missing_risks": len(missing_risks),
    }


def link_graph_dir(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == Path(os.path.relpath(src, dst.parent)):
            return
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Re-run with --overwrite to rebuild the city view."
            )
        safe_remove(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)


def copy_risk_aliases(src_risk: Path, alias_root: Path, split_name: str, alias: str) -> None:
    shutil.copy2(src_risk, alias_root / "risk_scores.json")
    if split_name == "evaluation_data" and alias in (VIEW_TARGET_ALIAS, "noisy_true"):
        shutil.copy2(src_risk, alias_root / "risk_scores_true.json")


def build_city_view(
    data_root: Path,
    source_city: str,
    target_city: str,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    view_root = data_root / "city_views" / f"{source_city}_to_{target_city}"
    aliases = {
        "clean": source_city,
        VIEW_TARGET_ALIAS: target_city,
        "noisy_true": target_city,
    }

    if dry_run:
        return view_root

    if overwrite and (view_root.exists() or view_root.is_symlink()):
        safe_remove(view_root)

    for split_name in SPLITS:
        for alias, city in aliases.items():
            city_root = data_root / split_name / f"clean_{city}"
            src_graph_dir = city_root / "graphs"
            src_risk = city_root / "risk_scores.json"
            if not src_graph_dir.is_dir():
                raise FileNotFoundError(f"City graph directory not found: {src_graph_dir}")
            if not src_risk.is_file():
                raise FileNotFoundError(f"City risk scores not found: {src_risk}")

            alias_root = view_root / split_name / alias
            alias_root.mkdir(parents=True, exist_ok=True)
            link_graph_dir(src_graph_dir, alias_root / "graphs", overwrite=overwrite)
            copy_risk_aliases(src_risk, alias_root, split_name=split_name, alias=alias)

    write_json(
        view_root / "city_view_manifest.json",
        {
            "source_city": source_city,
            "target_city": target_city,
            "note": (
                "This view maps source clean city data to the legacy 'clean' alias "
                "and target clean city data to 'noisy_0'/'noisy_true' aliases for "
                "compatibility with the existing training code. No noise is generated."
            ),
            "aliases": aliases,
        },
    )
    return view_root


def normalize_city(city: str) -> str:
    return city.strip().split("_")[-1].lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split NuPlan clean graph datasets by city and create a source-to-target "
            "city view for existing training code."
        )
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/NuPlan"))
    parser.add_argument("--source_city", type=str, default="singapore")
    parser.add_argument("--target_city", type=str, default="boston")
    parser.add_argument(
        "--link_mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize clean_<city>/graphs files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    cities = list(dict.fromkeys([source_city, target_city]))

    print(f"[city-split] data_root={data_root.resolve()}")
    print(f"[city-split] source_city={source_city} target_city={target_city}")
    print(f"[city-split] link_mode={args.link_mode} overwrite={args.overwrite} dry_run={args.dry_run}")

    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split_name in SPLITS:
        summary[split_name] = {}
        for city in cities:
            stats = split_one_city(
                data_root=data_root,
                split_name=split_name,
                city=city,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            summary[split_name][city] = stats
            print(
                f"[city-split] {split_name}/clean_{city}: "
                f"graphs={stats['graphs']} risks={stats['risks']} missing_risks={stats['missing_risks']}"
            )

    view_root = build_city_view(
        data_root=data_root,
        source_city=source_city,
        target_city=target_city,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"[city-split] city view: {view_root}")
    if args.dry_run:
        print("[city-split] dry run complete; no files were written.")
    else:
        print("[city-split] done.")


if __name__ == "__main__":
    main()
