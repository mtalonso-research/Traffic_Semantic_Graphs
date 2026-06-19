import argparse
import importlib
import sys
from argparse import Namespace
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict


VIEW_TARGET_NOISE_LEVEL = 0


BASELINE_DEFAULTS: Dict[str, Any] = {
    "l2d": False,
    "nup": False,
    "data_root": None,
    "clean": True,
    "noisy": None,
    "with_side_information": False,
    "mode": "all",
    "hidden_dim": 64,
    "embed_dim": 32,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.1,
    "risk_hidden_dim": 64,
    "prediction_mode": "classification",
    "num_classes": 4,
    "batch_size": 8,
    "num_workers": 4,
    "val_fraction": 0.2,
    "quant_bins": 32,
    "patience": 15,
    "train_autoencoder": False,
    "ae_epochs": 10,
    "ae_lr": 1e-4,
    "ae_weight_decay": 1e-5,
    "ae_train_num_neg": 1,
    "ae_val_num_neg": 4,
    "load_best_ae": False,
    "init_ae_checkpoint": None,
    "train_joint": False,
    "joint_epochs": 10,
    "joint_encoder_lr": 1e-4,
    "joint_head_lr": 1e-4,
    "joint_weight_decay": 1e-5,
    "joint_recon_weight": 1.0,
    "joint_risk_weight": 1.0,
    "train_risk": False,
    "risk_epochs": 10,
    "risk_lr": 1e-4,
    "risk_weight_decay": 1e-5,
    "risk_dropout": 0.5,
    "evaluate": False,
    "save_annotations": False,
    "sweep": False,
    "wandb": False,
    "best_model_path": "./models/risk_predictor/best_model.pt",
    "output_root": "./outputs",
    "seed": 42,
    "load_config": None,
    "run_id": None,
}


UST_DEFAULTS: Dict[str, Any] = {
    "l2d": False,
    "nup": False,
    "data_root": None,
    "clean": 5,
    "noisy": VIEW_TARGET_NOISE_LEVEL,
    "anchor_strategy": "random",
    "anchor_strat_bins": 5,
    "mode": "all",
    "hidden_dim": 64,
    "embed_dim": 32,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "activation": "relu",
    "dropout_rate": 0.1,
    "quant_bins": 32,
    "risk_hidden_dim": 64,
    "prediction_mode": "classification",
    "num_classes": 4,
    "use_proj_clean": False,
    "proj_hidden_dim": 0,
    "proj_dropout": 0.1,
    "proj_activation": "relu",
    "proj_residual": False,
    "proj_l2_normalize": False,
    "align_weight": 1.0,
    "align_loss_kind": "l2",
    "consistency_weight": 0.0,
    "consistency_kind": "kl",
    "batch_size": 8,
    "num_workers": 4,
    "val_fraction": 0.2,
    "patience": 15,
    "train_autoencoders": False,
    "ae_epochs": 10,
    "ae_lr": 1e-4,
    "ae_weight_decay": 1e-5,
    "init_ae_clean_checkpoint": None,
    "init_ae_noisy_checkpoint": None,
    "load_best_ae_clean": False,
    "load_best_ae_noisy": False,
    "ae_clean_ckpt_path": None,
    "ae_noisy_ckpt_path": None,
    "train_stage2": False,
    "train_noisy_proj_only": False,
    "stage2_epochs": 10,
    "stage2_lr": 1e-4,
    "stage2_weight_decay": 1e-5,
    "load_risk_head": False,
    "risk_head_ckpt_path": None,
    "load_proj_noisy": False,
    "proj_noisy_ckpt_path": None,
    "evaluate": False,
    "save_annotations": False,
    "sweep": False,
    "wandb": False,
    "best_model_path": "./models/risk_predictor/best_model.pt",
    "output_root": "./outputs",
    "seed": 42,
    "load_config": None,
    "run_id": None,
    "dataset_clean": None,
    "dataset_noisy": None,
}


def normalize_city(city: str) -> str:
    return city.strip().split("_")[-1].lower()


def city_pair(source_city: str, target_city: str) -> str:
    return f"{normalize_city(source_city)}_to_{normalize_city(target_city)}"


def default_view_root(args: Namespace) -> Path:
    return Path("data/NuPlan/city_views") / city_pair(args.source_city, args.target_city)


def resolved_view_root(args: Namespace) -> Path:
    if args.city_view_root is not None:
        return Path(args.city_view_root)
    return default_view_root(args)


def default_baseline_model_path(args: Namespace) -> str:
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    domain_city = source_city if args.domain == "source" else target_city
    return str(
        Path("./models/City/BaselineB")
        / city_pair(source_city, target_city)
        / f"{domain_city}_seed{args.seed}_best_model.pt"
    )


def default_ust_model_path(args: Namespace) -> str:
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    return str(
        Path("./models/City/UST")
        / city_pair(source_city, target_city)
        / f"anchor{args.anchor_pct}"
        / f"{source_city}_{target_city}_anchor{args.anchor_pct}_seed{args.seed}_best_model.pt"
    )


def require_city_view(view_root: Path) -> None:
    required = [
        view_root / "training_data" / "clean" / "graphs",
        view_root / "training_data" / "clean" / "risk_scores.json",
        view_root / "training_data" / "noisy_0" / "graphs",
        view_root / "training_data" / "noisy_0" / "risk_scores.json",
        view_root / "training_data" / "noisy_true" / "graphs",
        view_root / "training_data" / "noisy_true" / "risk_scores.json",
        view_root / "evaluation_data" / "clean" / "graphs",
        view_root / "evaluation_data" / "clean" / "risk_scores.json",
        view_root / "evaluation_data" / "noisy_0" / "graphs",
        view_root / "evaluation_data" / "noisy_0" / "risk_scores_true.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "City view is incomplete. Run scripts.city_split_processing first.\n"
            f"Missing:\n  - {joined}"
        )


def city_eval_sample_counts(view_root: Path, source_city: str, target_city: str) -> Dict[str, int]:
    source_graphs = view_root / "evaluation_data" / "clean" / "graphs"
    target_graphs = view_root / "evaluation_data" / f"noisy_{VIEW_TARGET_NOISE_LEVEL}" / "graphs"
    return {
        source_city: len(list(source_graphs.glob("*.json"))),
        target_city: len(list(target_graphs.glob("*.json"))),
    }


def namespace_from_defaults(defaults: Dict[str, Any], args: Namespace, updates: Dict[str, Any]) -> Namespace:
    values = dict(defaults)
    for key in values:
        if hasattr(args, key):
            values[key] = getattr(args, key)
    values.update(updates)
    return Namespace(**values)


def patch_legacy_annotations(module, enabled: bool) -> None:
    if not enabled:
        return

    def skip_log_annotations(*_args, **_kwargs):
        print("[city-train] Skipping legacy experiment_results/df.csv annotation logging.")

    module.log_annotations = skip_log_annotations


class CityLabelStream:
    def __init__(self, stream, source_city: str, target_city: str, eval_counts: Dict[str, int]):
        self.stream = stream
        source_count = eval_counts.get(source_city, 0)
        target_count = eval_counts.get(target_city, 0)
        self.replacements = [
            (
                f"========== noisy_{VIEW_TARGET_NOISE_LEVEL} evaluation results ==========",
                f"========== {target_city} evaluation results ==========\ntest_samples: {target_count}",
            ),
            (
                "========== clean evaluation results ==========",
                f"========== {source_city} evaluation results ==========\ntest_samples: {source_count}",
            ),
            (
                "========== 4Bb FIXED evaluation results ==========",
                (
                    "========== City UST evaluation results ==========\n"
                    f"test_samples/{source_city}: {source_count}\n"
                    f"test_samples/{target_city}: {target_count}\n"
                    f"test_samples/overall: {source_count + target_count}"
                ),
            ),
            (f"[eval:noisy_{VIEW_TARGET_NOISE_LEVEL}]", f"[eval:{target_city}]"),
            ("[eval:noisy]", f"[eval:{target_city}]"),
            ("[eval:clean]", f"[eval:{source_city}]"),
            (f" on noisy_{VIEW_TARGET_NOISE_LEVEL}", f" on {target_city}"),
            (" on clean", f" on {source_city}"),
            (f"noisy_{VIEW_TARGET_NOISE_LEVEL} evaluation avg loss", f"{target_city} evaluation avg loss"),
            ("clean evaluation avg loss", f"{source_city} evaluation avg loss"),
            (
                f"clean=clean, noisy=noisy_{VIEW_TARGET_NOISE_LEVEL}",
                f"source={source_city}, target={target_city}",
            ),
            ("CLEAN EVAL", f"{source_city.upper()} EVAL"),
            ("NOISY EVAL", f"{target_city.upper()} EVAL"),
            ("eval/clean/", f"eval/{source_city}/"),
            ("eval/noisy/", f"eval/{target_city}/"),
            ("CLEAN confusion matrix", f"{source_city.upper()} confusion matrix"),
            ("NOISY confusion matrix", f"{target_city.upper()} confusion matrix"),
        ]

    def write(self, text: str):
        for old, new in self.replacements:
            text = text.replace(old, new)
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def city_eval_labels(enabled: bool, source_city: str, target_city: str, eval_counts: Dict[str, int]):
    if not enabled:
        yield
        return

    stdout = CityLabelStream(sys.stdout, source_city, target_city, eval_counts)
    stderr = CityLabelStream(sys.stderr, source_city, target_city, eval_counts)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield


def add_city_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source_city", type=str, default="singapore")
    parser.add_argument("--target_city", type=str, default="boston")
    parser.add_argument(
        "--city_view_root",
        type=Path,
        default=None,
        help="Defaults to data/NuPlan/city_views/<source>_to_<target>.",
    )
    parser.add_argument(
        "--write_legacy_annotations",
        action="store_true",
        help="Allow the imported legacy script to update experiment_results/df.csv.",
    )


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--quant_bins", type=int, default=32)
    parser.add_argument("--risk_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="classification",
        choices=["regression", "classification"],
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience in validation epochs. Use <=0 to disable.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--save_annotations", action="store_true")
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_config", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", choices=["source", "target"], default="source")
    parser.add_argument("--train_autoencoder", action="store_true")
    parser.add_argument("--load_best_ae", action="store_true")
    parser.add_argument("--init_ae_checkpoint", type=str, default=None)
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--ae_lr", type=float, default=1e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-5)
    parser.add_argument("--ae_train_num_neg", type=int, default=1)
    parser.add_argument("--ae_val_num_neg", type=int, default=4)
    parser.add_argument("--train_joint", action="store_true")
    parser.add_argument("--joint_epochs", type=int, default=10)
    parser.add_argument("--joint_encoder_lr", type=float, default=1e-4)
    parser.add_argument("--joint_head_lr", type=float, default=1e-4)
    parser.add_argument("--joint_weight_decay", type=float, default=1e-5)
    parser.add_argument("--joint_recon_weight", type=float, default=1.0)
    parser.add_argument("--joint_risk_weight", type=float, default=1.0)
    parser.add_argument("--train_risk", action="store_true")
    parser.add_argument("--risk_epochs", type=int, default=10)
    parser.add_argument("--risk_lr", type=float, default=1e-4)
    parser.add_argument("--risk_weight_decay", type=float, default=1e-5)
    parser.add_argument("--risk_dropout", type=float, default=0.5)
    parser.add_argument("--evaluate", action="store_true")


def add_ust_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anchor_pct", type=int, default=5)
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default="random",
        choices=["random", "top_risk", "bottom_risk", "stratified", "nn"],
    )
    parser.add_argument("--anchor_strat_bins", type=int, default=5)
    parser.add_argument("--use_proj_clean", action="store_true")
    parser.add_argument("--proj_hidden_dim", type=int, default=0)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--proj_activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--proj_residual", action="store_true")
    parser.add_argument("--proj_l2_normalize", action="store_true")
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--align_loss_kind", type=str, default="l2", choices=["l2", "smoothl1", "cosine"])
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--consistency_kind", type=str, default="kl", choices=["mse", "kl"])
    parser.add_argument("--train_autoencoders", action="store_true")
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--ae_lr", type=float, default=1e-4)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-5)
    parser.add_argument("--init_ae_clean_checkpoint", type=str, default=None)
    parser.add_argument("--init_ae_noisy_checkpoint", type=str, default=None)
    parser.add_argument("--load_best_ae_clean", action="store_true")
    parser.add_argument("--load_best_ae_noisy", action="store_true")
    parser.add_argument("--ae_clean_ckpt_path", type=str, default=None)
    parser.add_argument("--ae_noisy_ckpt_path", type=str, default=None)
    parser.add_argument("--train_stage2", action="store_true")
    parser.add_argument("--train_noisy_proj_only", action="store_true")
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_weight_decay", type=float, default=1e-5)
    parser.add_argument("--load_risk_head", action="store_true")
    parser.add_argument("--risk_head_ckpt_path", type=str, default=None)
    parser.add_argument("--load_proj_noisy", action="store_true")
    parser.add_argument("--proj_noisy_ckpt_path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")


def run_baseline(args: Namespace) -> None:
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    view_root = resolved_view_root(args)
    require_city_view(view_root)

    best_model_path = args.best_model_path or default_baseline_model_path(args)
    is_source = args.domain == "source"
    domain_city = source_city if is_source else target_city

    legacy_args = namespace_from_defaults(
        BASELINE_DEFAULTS,
        args,
        {
            "data_root": str(view_root),
            "clean": is_source,
            "noisy": None if is_source else VIEW_TARGET_NOISE_LEVEL,
            "best_model_path": best_model_path,
        },
    )

    print(
        f"[city-train] BaselineB domain={args.domain} city={domain_city} "
        f"view={view_root} model={best_model_path}"
    )
    module = importlib.import_module("scripts.4A_ae_risk")
    patch_legacy_annotations(module, enabled=not args.write_legacy_annotations)
    eval_counts = city_eval_sample_counts(view_root, source_city, target_city) if legacy_args.evaluate else {}
    if legacy_args.evaluate:
        print(
            f"[city-train] Evaluation labels: clean -> {source_city}; "
            f"noisy_{VIEW_TARGET_NOISE_LEVEL}/noisy -> {target_city}"
        )
        print(
            f"[city-train] Test samples: {source_city}={eval_counts[source_city]}, "
            f"{target_city}={eval_counts[target_city]}"
        )
    with city_eval_labels(legacy_args.evaluate, source_city, target_city, eval_counts):
        module.run_task(legacy_args)


def run_ust(args: Namespace) -> None:
    source_city = normalize_city(args.source_city)
    target_city = normalize_city(args.target_city)
    view_root = resolved_view_root(args)
    require_city_view(view_root)

    best_model_path = args.best_model_path or default_ust_model_path(args)
    legacy_args = namespace_from_defaults(
        UST_DEFAULTS,
        args,
        {
            "data_root": str(view_root),
            "clean": args.anchor_pct,
            "noisy": VIEW_TARGET_NOISE_LEVEL,
            "best_model_path": best_model_path,
        },
    )

    if legacy_args.load_best_ae_clean and not legacy_args.ae_clean_ckpt_path:
        raise SystemExit("--load_best_ae_clean requires --ae_clean_ckpt_path")
    if legacy_args.load_best_ae_noisy and not legacy_args.ae_noisy_ckpt_path:
        raise SystemExit("--load_best_ae_noisy requires --ae_noisy_ckpt_path")
    if legacy_args.load_risk_head and not legacy_args.risk_head_ckpt_path:
        raise SystemExit("--load_risk_head requires --risk_head_ckpt_path")
    if legacy_args.load_proj_noisy and not legacy_args.proj_noisy_ckpt_path:
        raise SystemExit("--load_proj_noisy requires --proj_noisy_ckpt_path")

    print(
        f"[city-train] UST source={source_city} target={target_city} "
        f"anchor_pct={args.anchor_pct} view={view_root} model={best_model_path}"
    )
    module = importlib.import_module("scripts.5A_ust_risk")
    patch_legacy_annotations(module, enabled=not args.write_legacy_annotations)
    eval_counts = city_eval_sample_counts(view_root, source_city, target_city) if legacy_args.evaluate else {}
    if legacy_args.evaluate:
        print(
            f"[city-train] Evaluation labels: clean -> {source_city}; "
            f"noisy_{VIEW_TARGET_NOISE_LEVEL}/noisy -> {target_city}"
        )
        print(
            f"[city-train] Test samples: {source_city}={eval_counts[source_city]}, "
            f"{target_city}={eval_counts[target_city]}"
        )
    with city_eval_labels(legacy_args.evaluate, source_city, target_city, eval_counts):
        module.run_task(legacy_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "City-domain training wrapper. It reuses existing model code with a "
            "clean city view, without modifying the original training loops."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Train/evaluate BaselineB on one city domain.")
    add_city_args(baseline)
    add_shared_model_args(baseline)
    add_baseline_args(baseline)
    baseline.set_defaults(func=run_baseline)

    ust = subparsers.add_parser("ust", help="Train/evaluate UST from source city to target city.")
    add_city_args(ust)
    add_shared_model_args(ust)
    add_ust_args(ust)
    ust.set_defaults(func=run_ust)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
