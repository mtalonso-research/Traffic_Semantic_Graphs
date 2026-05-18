import subprocess
import argparse
from tqdm import tqdm
import logging
from utils.custom_logging import CustomFormatter
logger = logging.getLogger(__name__)

def _build_args_from_config(config: dict, arg_map: dict) -> list[str]:
    """Convert config dict into a flat CLI arg list using arg_map."""
    args: list[str] = []
    for key, value in config.items():
        if key == "script":
            continue
        flag = arg_map.get(key)
        if not flag:
            continue

        # bool flags: include only when True
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue

        # non-bool: include when not None
        if value is not None:
            args.extend([flag, str(value)])
    return args


def _run_command(full_command: list[str], verbose: bool) -> int:
    """Run command; suppress output unless error (unless verbose)."""
    try:
        if verbose:
            print(f"Running VERBOSE command: {' '.join(full_command)}\n")
            process = subprocess.Popen(full_command, text=True)
            process.wait()
            return process.returncode

        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"\n--- Command failed with exit code {process.returncode} ---")
            print(f"--- Command: {' '.join(full_command)} ---")
            print("\n--- STDOUT ---")
            print(stdout)
            print("\n--- STDERR ---")
            print(stderr)
        return process.returncode

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return -1


def run_experiment_baselineb(config, verbose=False):
    """
    Constructs and runs a command for a BaselineB experiment.
    Uses module invocation: python -m scripts.<script>
    """
    script_name = config["script"]
    base_command = ["python", "-m", f"scripts.{script_name}"]

    arg_map = {
        # dataset source
        "l2d": "--l2d",
        "nup": "--nup",
        "clean": "--clean",
        "noisy": "--noisy",

        # misc / task
        "seed": "--seed",
        "mode": "--mode",
        "prediction_mode": "--prediction_mode",
        "num_classes": "--num_classes",

        # loader / split
        "num_workers": "--num_workers",
        "batch_size": "--batch_size",
        "val_fraction": "--val_fraction",

        # stage controls
        "evaluate": "--evaluate",
        "train_autoencoder": "--train_autoencoder",
        "train_risk": "--train_risk",
        "load_best_ae": "--load_best_ae",

        # output
        "best_model_path": "--best_model_path",

        # epochs
        "ae_epochs": "--ae_epochs",
        "risk_epochs": "--risk_epochs",

        # model (AE)
        "hidden_dim": "--hidden_dim",
        "embed_dim": "--embed_dim",
        "num_encoder_layers": "--num_encoder_layers",
        "num_decoder_layers": "--num_decoder_layers",
        "activation": "--activation",
        "dropout_rate": "--dropout_rate",
        "quant_bins": "--quant_bins",

        # optim
        "ae_lr": "--ae_lr",
        "ae_weight_decay": "--ae_weight_decay",
        "risk_lr": "--risk_lr",
        "risk_weight_decay": "--risk_weight_decay",

        # risk head
        "risk_hidden_dim": "--risk_hidden_dim",
    }

    args = _build_args_from_config(config, arg_map)
    full_command = base_command + args
    full_command_text = " ".join(full_command)
    logger.debug(full_command_text)
    return _run_command(full_command, verbose=verbose)


def run_experiment_ust(config, verbose=False):
    """
    Constructs and runs a command for a UST experiment.
    Uses file invocation: python scripts/<script>.py
    """
    script_name = config["script"]
    base_command = ["python", f"scripts/{script_name}.py"]

    # This map matches the *actual* CLI flags in your UST script.
    arg_map = {
        # stage controls
        "train_stage2": "--train_stage2",
        "train_autoencoders": "--train_autoencoders",
        "evaluate": "--evaluate",

        # dataset source + types
        "l2d": "--l2d",
        "nup": "--nup",
        "clean": "--clean",  # anchor percentage
        "noisy": "--noisy",

        # AE ckpt loading
        "load_best_ae_clean": "--load_best_ae_clean",
        "ae_clean_ckpt_path": "--ae_clean_ckpt_path",
        "load_best_ae_noisy": "--load_best_ae_noisy",
        "ae_noisy_ckpt_path": "--ae_noisy_ckpt_path",

        # output / misc
        "best_model_path": "--best_model_path",
        "seed": "--seed",

        # loader / split
        "num_workers": "--num_workers",
        "batch_size": "--batch_size",
        "val_fraction": "--val_fraction",

        # AEs (match BaselineB where possible)
        "mode": "--mode",
        "embed_dim": "--embed_dim",
        "hidden_dim": "--hidden_dim",
        "num_encoder_layers": "--num_encoder_layers",
        "num_decoder_layers": "--num_decoder_layers",
        "activation": "--activation",
        "dropout_rate": "--dropout_rate",
        "quant_bins": "--quant_bins",

        # risk head / task
        "risk_hidden_dim": "--risk_hidden_dim",
        "prediction_mode": "--prediction_mode",
        "num_classes": "--num_classes",

        # stage2 hparams
        "stage2_epochs": "--stage2_epochs",
        "stage2_lr": "--stage2_lr",
        "stage2_weight_decay": "--stage2_weight_decay",

        # ---- optional knobs (kept here so the runner can control them later if desired) ----
        "use_proj_clean": "--use_proj_clean",
        "proj_hidden_dim": "--proj_hidden_dim",
        "proj_dropout": "--proj_dropout",
        "proj_activation": "--proj_activation",
        "proj_residual": "--proj_residual",
        "proj_l2_normalize": "--proj_l2_normalize",
        "align_weight": "--align_weight",
        "align_loss_kind": "--align_loss_kind",
        "consistency_weight": "--consistency_weight",
        "consistency_kind": "--consistency_kind",
        "train_noisy_proj_only": "--train_noisy_proj_only",
        "load_risk_head": "--load_risk_head",
        "risk_head_ckpt_path": "--risk_head_ckpt_path",
        "load_proj_noisy": "--load_proj_noisy",
        "proj_noisy_ckpt_path": "--proj_noisy_ckpt_path",
        "save_annotations": "--save_annotations",
        "wandb": "--wandb",
        "sweep": "--sweep",
        "run_id": "--run_id",
        "output_root": "--output_root",
        "load_config": "--load_config",
        "data_root": "--data_root",
    }

    args = _build_args_from_config(config, arg_map)
    full_command = base_command + args
    return _run_command(full_command, verbose=verbose)


def experiment_loop(
    exp,
    noises=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    seeds=[42, 1024, 31, 25, 334, 723, 567, 7, 121, 9],
    anchors=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
    zero_workers=False,
    verbose=False,
):
    BEST = {
        "hidden_dim": 64,
        "embed_dim": 64,
        "num_encoder_layers": 3,
        "num_decoder_layers": 1,
        "activation": "relu",
        "dropout_rate": 0.2806601930842774,
        "quant_bins": 32,
        "risk_hidden_dim": 256,
        "ae_lr": 0.00018938449208842956,
        "ae_weight_decay": 0.00007085783220313448,
        "risk_lr": 0.0000990628176417087,
        "risk_weight_decay": 0.0000010465199907668271,
        "batch_size": 4,
        "val_fraction": 0.2,
        "num_workers": 4,
        "mode": "all",
        "prediction_mode": "classification",
        "num_classes": 4,
    }

    if exp == "BaselineB":
        total_runs = len(noises) * len(seeds)
        logger.info(f"Total runs: {total_runs}")
        pbar = tqdm(total=total_runs, desc="Running BaselineB Experiments") if not verbose else None

        for noise in noises:
            base_config = {
                "script": "4A_ae_risk",
                "nup": True,
                "mode": BEST["mode"],
                "prediction_mode": BEST["prediction_mode"],
                "num_classes": BEST["num_classes"],

                "train_autoencoder": True,
                "train_risk": True,
                "load_best_ae": True,
                "evaluate": True,

                "hidden_dim": BEST["hidden_dim"],
                "embed_dim": BEST["embed_dim"],
                "num_encoder_layers": BEST["num_encoder_layers"],
                "num_decoder_layers": BEST["num_decoder_layers"],
                "activation": BEST["activation"],
                "dropout_rate": BEST["dropout_rate"],
                "quant_bins": BEST["quant_bins"],

                "risk_hidden_dim": BEST["risk_hidden_dim"],

                "batch_size": BEST["batch_size"],
                "val_fraction": BEST["val_fraction"],
                "num_workers": BEST["num_workers"],

                "ae_epochs": 60,
                "risk_epochs": 20,

                "ae_lr": BEST["ae_lr"],
                "ae_weight_decay": BEST["ae_weight_decay"],
                "risk_lr": BEST["risk_lr"],
                "risk_weight_decay": BEST["risk_weight_decay"],
            }

            if noise == 0:
                base_config["clean"] = True
            else:
                base_config["noisy"] = noise

            if zero_workers:
                base_config["num_workers"] = 0

            # Create a readable summary for base_config
            base_config_text = "\n".join([f"\t{key}:{value}" for key, value in base_config.items()])
            logger.debug(f"Config for noise={noise}:\n{base_config_text}")

            for seed_value in seeds:
                logger.info(f"Seed:{seed_value}")
                if pbar:
                    pbar.set_description(f"BaselineB (noise={noise}, seed={seed_value})")

                current_config = base_config.copy()
                current_config["seed"] = seed_value
                current_config["best_model_path"] = (
                    f"./models/BaselineB/"
                    f'{"clean" if noise == 0 else f"noisy{noise}"}/'
                    f'{"clean" if noise == 0 else f"noisy{noise}"}_seed{seed_value}.pt'
                )
                logger.info(f"best_model_path: {current_config['best_model_path']}")

                rc = run_experiment_baselineb(current_config, verbose=verbose)
                if rc != 0 and pbar:
                    pbar.write(f"Run FAILED for noise={noise}, seed={seed_value}")
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    elif exp == "UST":
        total_runs = len(noises) * len(seeds) * len(anchors)
        pbar = tqdm(total=total_runs, desc="Running UST Experiments") if not verbose else None

        for noise in noises:
            for seed in seeds:
                base_config = {
                    "script": "5A_ust_risk",
                    "train_stage2": True,
                    "evaluate": True,

                    # dataset
                    "nup": True,
                    "noisy": noise,

                    # match BaselineB BEST where possible
                    "mode": BEST["mode"],
                    "prediction_mode": BEST["prediction_mode"],
                    "num_classes": BEST["num_classes"],

                    # AEs
                    "hidden_dim": BEST["hidden_dim"],
                    "embed_dim": BEST["embed_dim"],
                    "num_encoder_layers": BEST["num_encoder_layers"],
                    "num_decoder_layers": BEST["num_decoder_layers"],
                    "activation": BEST["activation"],
                    "dropout_rate": BEST["dropout_rate"],
                    "quant_bins": BEST["quant_bins"],

                    # risk head capacity
                    "risk_hidden_dim": BEST["risk_hidden_dim"],

                    # loader/split
                    "batch_size": BEST["batch_size"],
                    "val_fraction": BEST["val_fraction"],
                    "num_workers": BEST["num_workers"],

                    # stage2 training hyperparams (match BaselineB risk optimizer)
                    "stage2_epochs": 20,
                    "stage2_lr": BEST["risk_lr"],
                    "stage2_weight_decay": BEST["risk_weight_decay"],

                    # load AEs from BaselineB
                    "load_best_ae_clean": True,
                    "load_best_ae_noisy": True,
                    "ae_clean_ckpt_path": f"./models/BaselineB/clean/clean_seed{seed}.pt",
                    "ae_noisy_ckpt_path": f"./models/BaselineB/noisy{noise}/noisy{noise}_seed{seed}.pt",
                }

                if zero_workers:
                    base_config["num_workers"] = 0

                for anchor in anchors:
                    if pbar:
                        pbar.set_description(f"UST (noise={noise}, seed={seed}, anchor={anchor})")

                    current_config = base_config.copy()
                    current_config["seed"] = seed
                    current_config["clean"] = anchor  # anchor is percent in UST script
                    current_config["best_model_path"] = (
                        f"./models/UST/noisy{noise}/anchor{anchor}/"
                        f"clean{anchor}_noisy{noise}_seed{seed}.pt"
                    )

                    rc = run_experiment_ust(current_config, verbose=verbose)
                    if rc != 0 and pbar:
                        pbar.write(f"Run FAILED for noise={noise}, seed={seed}, anchor={anchor}")
                    if pbar:
                        pbar.update(1)

        if pbar:
            pbar.close()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) # For now I always run in debug mode
    file_handler = logging.FileHandler("logging.log", mode="w")
    file_handler.setFormatter(CustomFormatter())
    logger.addHandler(file_handler)
    logger.info("Started")

    parser = argparse.ArgumentParser(description="Run a series of experiments.")
    parser.add_argument("--experiment", choices=["BaselineB", "UST"], required=True)
    parser.add_argument("--noises", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--anchors", nargs="+", type=int)
    parser.add_argument("--zero_workers", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Create a readable summary of args
    args_text = "\n".join([f"\t{key}: {value}"for key, value in vars(args).items()])
    logger.info(f"Passing arguments:\n{args_text}")

    kwargs = {
        "exp": args.experiment,
        "zero_workers": args.zero_workers,
        "verbose": args.verbose,
    }

    if args.noises is not None:
        kwargs["noises"] = args.noises
    if args.seeds is not None:
        kwargs["seeds"] = args.seeds
    if args.anchors is not None:
        kwargs["anchors"] = args.anchors

    # Create a readable summary of kwargs
    kwargs_text="\n".join([f"\t{key}: {value}" for key, value in kwargs.items()])
    logger.debug(f"Arugments passed to experiment loop:\n{kwargs_text}")
    experiment_loop(**kwargs)

    logging.info("Finished")