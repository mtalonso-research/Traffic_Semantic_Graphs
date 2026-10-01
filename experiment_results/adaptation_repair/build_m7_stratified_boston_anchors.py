"""Create audited class-stratified, strictly nested M7 Boston train-anchor manifests.

Only target_train IDs and their target-train risk labels are read.  Validation
and evaluation IDs are used solely for disjointness assertions, never labels.
"""
from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "experiment_results/nested_scene_split/frozen_boston"
MANIFEST = FROZEN / "manifests/singapore_to_boston/manifest.json"
RISK = FROZEN / "city_views/singapore_to_boston/training_data/noisy_0/risk_scores.json"
OUT = ROOT / "experiment_results/adaptation_repair/m7_stratified_boston_anchors"
COUNTS = {3: 109, 5: 181, 10: 362, 20: 723, 50: 1808}
SEEDS = (25, 42)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def cls(risk: float) -> int:
    return 3 if risk > .3442 else 2 if risk > .1008 else 1 if risk > .0043 else 0


def build_order(buckets: dict[int, list[str]], seed: int) -> list[str]:
    """Deterministic proportional interleaving, with every class in positions 0..3."""
    streams = {label: values[:] for label, values in buckets.items()}
    for label in range(4):
        random.Random((seed << 8) + label).shuffle(streams[label])
    total = sum(map(len, streams.values()))
    proportions = {label: len(streams[label]) / total for label in range(4)}
    used = {label: 0 for label in range(4)}
    order: list[str] = []
    # Explicit four-class coverage precedes quota interleaving.
    for label in sorted(streams):
        order.append(streams[label][used[label]])
        used[label] += 1
    while len(order) < total:
        position = len(order) + 1
        available = [label for label in range(4) if used[label] < len(streams[label])]
        label = max(available, key=lambda x: (position * proportions[x] - used[x], -x))
        order.append(streams[label][used[label]])
        used[label] += 1
    assert len(order) == len(set(order)) == total
    return order


def main() -> None:
    split = json.loads(MANIFEST.read_text(encoding="utf-8"))
    risks = json.loads(RISK.read_text(encoding="utf-8"))
    train = list(split["target_train"])
    validation = set(split["target_validation"])
    evaluation = set(split["target_evaluation"])
    buckets = {label: [] for label in range(4)}
    for sample_id in train:
        if sample_id not in risks:
            raise ValueError(f"missing target-train risk label: {sample_id}")
        buckets[cls(float(risks[sample_id]))].append(sample_id)
    if any(not buckets[label] for label in range(4)):
        raise ValueError(f"target train lacks a class: {[len(buckets[x]) for x in range(4)]}")
    OUT.mkdir(parents=True, exist_ok=True)
    base = {
        "protocol": "m7_class_stratified_strict_nested_anchor_v1",
        "city": "boston",
        "source_split_manifest_sha256": sha_bytes(MANIFEST.read_bytes()),
        "risk_labels_file_sha256": sha_bytes(RISK.read_bytes()),
        "target_train_count": len(train),
        "target_train_class_counts": {str(x): len(buckets[x]) for x in range(4)},
        "anchor_counts": {str(k): v for k, v in COUNTS.items()},
        "selection_rule": "train-label-only deterministic class-proportional interleaving; first four positions cover classes 0/1/2/3; every ratio is a prefix",
        "validation_or_test_labels_read": False,
    }
    for seed in SEEDS:
        order = build_order(buckets, seed)
        anchors = {str(ratio): order[:count] for ratio, count in COUNTS.items()}
        assert all(set(ids).issubset(train) and not set(ids) & validation and not set(ids) & evaluation for ids in anchors.values())
        assert all(set(anchors[str(a)]).issubset(anchors[str(b)]) for a, b in zip(COUNTS, tuple(COUNTS)[1:]))
        class_counts = {str(ratio): {str(label): sum(cls(float(risks[x])) == label for x in ids) for label in range(4)} for ratio, ids in anchors.items()}
        if any(any(value == 0 for value in values.values()) for values in class_counts.values()):
            raise AssertionError("a requested M7 ratio lacks class coverage")
        payload = base | {
            "training_seed": seed,
            "class_permutation_seeds": {str(label): (seed << 8) + label for label in range(4)},
            "ordered_anchor_pool_sha256": sha_bytes("".join(f"{x}\n" for x in order).encode()),
            "anchor_ids": anchors,
            "anchor_ids_sha256": {ratio: sha_bytes("".join(f"{x}\n" for x in ids).encode()) for ratio, ids in anchors.items()},
            "anchor_class_counts": class_counts,
            "nesting_checks": {f"{a}%_subset_{b}%": True for a, b in zip(COUNTS, tuple(COUNTS)[1:])},
            "integrity_checks": {"all_anchor_ids_in_target_train": True, "no_anchor_validation_overlap": True, "no_anchor_evaluation_overlap": True},
        }
        (OUT / f"seed_{seed}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = base | {"seeds": list(SEEDS), "files": {str(seed): f"seed_{seed}.json" for seed in SEEDS}}
    (OUT / "README.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
