import argparse
import json
import os
import random
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


def episode_sort_key(ep: str):
    return (0, int(ep)) if ep.isdigit() else (1, ep)


def sorted_json_dict(values: Dict[str, float]) -> Dict[str, float]:
    return dict(sorted(values.items(), key=lambda item: episode_sort_key(item[0])))


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_empty_or_create_dir(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if overwrite:
            safe_remove(path)
        elif path.is_dir() and not any(path.iterdir()):
            return
        else:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to rebuild it.")
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


def clean_split_inputs(data_root: Path, split_name: str) -> Tuple[Path, Dict[str, float]]:
    graph_dir = data_root / split_name / "clean" / "graphs"
    risk_path = data_root / split_name / "clean" / "risk_scores.json"
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Clean graph directory not found: {graph_dir}")
    return graph_dir, load_risk_scores(risk_path)


def collect_city_pool(data_root: Path, city: str) -> Dict[str, Tuple[Path, float]]:
    pool: Dict[str, Tuple[Path, float]] = {}
    missing_risks = []

    for split_name in SPLITS:
        graph_dir, risks = clean_split_inputs(data_root, split_name)
        for graph_path in sorted(graph_dir.glob("*.json"), key=numeric_graph_sort_key):
            if city_from_graph(graph_path) != city:
                continue

            ep = episode_id(graph_path)
            if ep not in risks:
                missing_risks.append(ep)
                continue

            pool[ep] = (graph_path, risks[ep])

    if missing_risks:
        preview = ", ".join(missing_risks[:10])
        raise ValueError(
            f"Missing risk scores for {len(missing_risks)} {city} graphs. "
            f"First missing episode ids: {preview}"
        )

    return pool


def fixed_test_split(
    pool: Dict[str, Tuple[Path, float]],
    test_size: int,
    seed: int,
) -> Tuple[Dict[str, Tuple[Path, float]], Dict[str, Tuple[Path, float]]]:
    if test_size <= 0:
        raise ValueError("--test_size must be positive.")
    if len(pool) < test_size:
        raise ValueError(f"Cannot make a test set of {test_size} from only {len(pool)} graphs.")

    episode_ids = sorted(pool, key=episode_sort_key)
    rng = random.Random(seed)
    test_ids = set(rng.sample(episode_ids, test_size))

    train_pool = {ep: pool[ep] for ep in episode_ids if ep not in test_ids}
    test_pool = {ep: pool[ep] for ep in episode_ids if ep in test_ids}
    return train_pool, test_pool


def write_city_split(
    data_root: Path,
    split_name: str,
    city: str,
    selected: Dict[str, Tuple[Path, float]],
    link_mode: str,
    overwrite: bool,
    dry_run: bool,
    test_size: int,
    split_seed: int,
    pool_total: int,
) -> Dict[str, int]:
    out_root = data_root / split_name / f"clean_{city}"
    out_graph_dir = out_root / "graphs"

    if not dry_run:
        ensure_empty_or_create_dir(out_graph_dir, overwrite=overwrite)

    city_risks = {}
    for ep, (graph_path, risk_score) in selected.items():
        city_risks[ep] = risk_score
        if not dry_run:
            transfer_file(graph_path, out_graph_dir / graph_path.name, mode=link_mode)

    if not dry_run:
        write_json(out_root / "risk_scores.json", sorted_json_dict(city_risks))
        write_json(
            out_root / "city_split_manifest.json",
            {
                "city": city,
                "split": split_name,
                "source": "fixed_test_size_city_split",
                "link_mode": link_mode,
                "test_size": test_size,
                "split_seed": split_seed,
                "pool_total": pool_total,
                "graph_count": len(city_risks),
                "risk_count": len(city_risks),
                "missing_risk_count": 0,
            },
        )

    return {
        "graphs": len(city_risks),
        "risks": len(city_risks),
        "missing_risks": 0,
    }


def split_one_city_fixed_test(
    data_root: Path,
    city: str,
    test_size: int,
    seed: int,
    link_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Dict[str, int]]:
    pool = collect_city_pool(data_root, city)
    train_pool, test_pool = fixed_test_split(pool, test_size=test_size, seed=seed)
    return {
        "training_data": write_city_split(
            data_root=data_root,
            split_name="training_data",
            city=city,
            selected=train_pool,
            link_mode=link_mode,
            overwrite=overwrite,
            dry_run=dry_run,
            test_size=test_size,
            split_seed=seed,
            pool_total=len(pool),
        ),
        "evaluation_data": write_city_split(
            data_root=data_root,
            split_name="evaluation_data",
            city=city,
            selected=test_pool,
            link_mode=link_mode,
            overwrite=overwrite,
            dry_run=dry_run,
            test_size=test_size,
            split_seed=seed,
            pool_total=len(pool),
        ),
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
    test_size: int,
    split_seed: int,
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
            "test_size": test_size,
            "split_seed": split_seed,
            "note": (
                "This view maps source clean city data to the legacy clean alias "
                "and target clean city data to noisy_0/noisy_true aliases for "
                "compatibility with the existing training code. No noise is generated. "
                "Each city evaluation set is fixed to test_size samples."
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
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--split_seed", type=int, default=228)
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
    print(
        f"[city-split] test_size={args.test_size} split_seed={args.split_seed} "
        f"link_mode={args.link_mode} overwrite={args.overwrite} dry_run={args.dry_run}"
    )

    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for city_index, city in enumerate(cities):
        city_summary = split_one_city_fixed_test(
            data_root=data_root,
            city=city,
            test_size=args.test_size,
            seed=args.split_seed + city_index,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for split_name, stats in city_summary.items():
            summary.setdefault(split_name, {})
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
        test_size=args.test_size,
        split_seed=args.split_seed,
    )
    print(f"[city-split] city view: {view_root}")
    if args.dry_run:
        print("[city-split] dry run complete; no files were written.")
    else:
        print("[city-split] done.")


if __name__ == "__main__":
    main()
