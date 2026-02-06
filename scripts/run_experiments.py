import subprocess
import os
import sys
import argparse
from tqdm import tqdm

def run_experiment_baselineb(config, verbose=False):
    """
    Constructs and runs a command for a BaselineB experiment.
    If verbose is False, output is suppressed unless an error occurs.
    """
    script_name = config['script']
    base_command = ["python", "-m", f"scripts.{script_name}"]
    args = []

    arg_map = {
        # dataset / split / misc
        'l2d': '--l2d', 'nup': '--nup', 'clean': '--clean', 'noisy': '--noisy',
        'seed': '--seed', 'mode': '--mode',
        'prediction_mode': '--prediction_mode', 'num_classes': '--num_classes',
        'num_workers': '--num_workers', 'batch_size': '--batch_size', 'val_fraction': '--val_fraction',

        # stages / eval
        'evaluate': '--evaluate', 'train_autoencoder': '--train_autoencoder',
        'train_risk': '--train_risk', 'load_best_ae': '--load_best_ae',

        # output
        'best_model_path': '--best_model_path', 'output_root': '--output_root',

        # epochs
        'ae_epochs': '--ae_epochs', 'risk_epochs': '--risk_epochs',

        # AE model
        'hidden_dim': '--hidden_dim', 'embed_dim': '--embed_dim',
        'num_encoder_layers': '--num_encoder_layers', 'num_decoder_layers': '--num_decoder_layers',
        'activation': '--activation', 'dropout_rate': '--dropout_rate',
        'quant_bins': '--quant_bins',

        # optim
        'ae_lr': '--ae_lr', 'ae_weight_decay': '--ae_weight_decay',
        'risk_lr': '--risk_lr', 'risk_weight_decay': '--risk_weight_decay',

        # risk head
        'risk_hidden_dim': '--risk_hidden_dim',
    }

    for key, value in config.items():
        if key == 'script':
            continue
        flag = arg_map.get(key)
        if flag:
            if isinstance(value, bool) and value:
                args.append(flag)
            elif not isinstance(value, bool) and value is not None:
                args.extend([flag, str(value)])

    full_command = base_command + args
    try:
        if verbose:
            print(f"Running VERBOSE command: {' '.join(full_command)}\n")
            process = subprocess.Popen(full_command, text=True)
            process.wait()
        else:
            process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def run_experiment_ust(config, verbose=False):
    """
    Constructs and runs a command for a UST experiment.
    If verbose is False, output is suppressed unless an error occurs.
    """
    script_name = config['script']
    base_command = ["python", f"scripts/{script_name}.py"]
    args = []

    arg_map = {
        'train_stage2': '--train_stage2', 'evaluate': '--evaluate', 'nup': '--nup',
        'clean': '--clean', 'noisy': '--noisy', 'load_best_ae_clean': '--load_best_ae_clean',
        'ae_clean_ckpt_path': '--ae_clean_ckpt_path', 'load_best_ae_noisy': '--load_best_ae_noisy',
        'ae_noisy_ckpt_path': '--ae_noisy_ckpt_path', 'best_model_path': '--best_model_path',
        'num_workers': '--num_workers',
        'embed_dim': '--embed_dim', 'hidden_dim': '--hidden_dim',
        'num_encoder_layers': '--num_encoder_layers', 'num_decoder_layers': '--num_decoder_layers',
        'risk_hidden_dim': '--risk_hidden_dim',
        'stage2_epochs': '--stage2_epochs',
        'stage2_lr': '--stage2_lr', 'stage2_weight_decay': '--stage2_weight_decay',
        'seed': '--seed',
    }

    for key, value in config.items():
        if key == 'script':
            continue
        flag = arg_map.get(key)
        if flag:
            if isinstance(value, bool) and value:
                args.append(flag)
            elif not isinstance(value, bool) and value is not None:
                args.extend([flag, str(value)])

    full_command = base_command + args
    try:
        if verbose:
            print(f"Running VERBOSE command: {' '.join(full_command)}\n")
            process = subprocess.Popen(full_command, text=True)
            process.wait()
        else:
            process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def experiment_loop(
    exp,
    noises=[5,10,15,20,25,30,35,40,45,50,55,60],
    seeds=[42, 1024, 31, 25, 334, 723, 567, 7, 121, 9],
    anchors=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
    zero_workers=False,
    verbose=False
):
    # Best-run sweep params (from your W&B best-run overview)
    BEST = {
        # AE / encoder
        "hidden_dim": 64,
        "embed_dim": 64,
        "num_encoder_layers": 3,
        "num_decoder_layers": 1,
        "activation": "relu",
        "dropout_rate": 0.2806601930842774,
        "quant_bins": 32,

        # risk head
        "risk_hidden_dim": 256,

        # optim
        "ae_lr": 0.00018938449208842956,
        "ae_weight_decay": 0.00007085783220313448,
        "risk_lr": 0.0000990628176417087,
        "risk_weight_decay": 0.0000010465199907668271,

        # loader
        "batch_size": 4,
        "val_fraction": 0.2,
        "num_workers": 4,

        # misc
        "mode": "all",
        "prediction_mode": "classification",
        "num_classes": 4,
    }

    if exp == 'BaselineB':
        total_runs = len(noises) * len(seeds)
        pbar = tqdm(total=total_runs, desc="Running BaselineB Experiments") if not verbose else None

        for noise in noises:
            base_config = {
                'script': '4A_ae_risk',

                # dataset selection
                'nup': True,
                'noisy': noise,
                'prediction_mode': BEST["prediction_mode"],
                'num_classes': BEST["num_classes"],
                'mode': BEST["mode"],

                # training flags
                'train_autoencoder': True,
                'train_risk': True,
                'load_best_ae': True,
                'evaluate': True,

                # model (match sweep-best)
                'hidden_dim': BEST["hidden_dim"],
                'embed_dim': BEST["embed_dim"],
                'num_encoder_layers': BEST["num_encoder_layers"],
                'num_decoder_layers': BEST["num_decoder_layers"],
                'activation': BEST["activation"],
                'dropout_rate': BEST["dropout_rate"],
                'quant_bins': BEST["quant_bins"],

                # risk head (match sweep-best)
                'risk_hidden_dim': BEST["risk_hidden_dim"],

                # loader/split (match sweep-best)
                'batch_size': BEST["batch_size"],
                'val_fraction': BEST["val_fraction"],
                'num_workers': BEST["num_workers"],

                # IMPORTANT: keep your original epochs
                'ae_epochs': 60,
                'risk_epochs': 20,

                # optim (match sweep-best)
                'ae_lr': BEST["ae_lr"],
                'ae_weight_decay': BEST["ae_weight_decay"],
                'risk_lr': BEST["risk_lr"],
                'risk_weight_decay': BEST["risk_weight_decay"],
            }

            if zero_workers:
                base_config['num_workers'] = 0

            for seed_value in seeds:
                if pbar:
                    pbar.set_description(f"BaselineB (noise={noise}, seed={seed_value})")

                current_config = base_config.copy()
                current_config['best_model_path'] = f'./models/BaselineB/noisy{noise}/noisy{noise}_seed{seed_value}.pt'
                current_config['seed'] = seed_value

                return_code = run_experiment_baselineb(current_config, verbose=verbose)
                if return_code != 0 and pbar:
                    pbar.write(f"Run FAILED for noise={noise}, seed={seed_value}")
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    elif exp == 'UST':
        total_runs = len(noises) * len(seeds) * len(anchors)
        pbar = tqdm(total=total_runs, desc="Running UST Experiments") if not verbose else None

        for noise in noises:
            for seed in seeds:
                base_config = {
                    'script': '5A_ust_risk',
                    'train_stage2': True,
                    'evaluate': True,
                    'nup': True,

                    'load_best_ae_clean': True,
                    'load_best_ae_noisy': True,
                    'noisy': noise,

                    'ae_clean_ckpt_path': f'./models/BaselineB/clean/clean_seed{seed}.pt',
                    'ae_noisy_ckpt_path': f'./models/BaselineB/noisy{noise}/noisy{noise}_seed{seed}.pt',

                    # encoder/AE architecture (match sweep-best)
                    'embed_dim': BEST["embed_dim"],
                    'hidden_dim': BEST["hidden_dim"],
                    'num_encoder_layers': BEST["num_encoder_layers"],
                    'num_decoder_layers': BEST["num_decoder_layers"],

                    # risk head (match sweep-best)
                    'risk_hidden_dim': BEST["risk_hidden_dim"],

                    # IMPORTANT: keep your original epochs for UST stage2
                    'stage2_epochs': 20,

                    # stage2 optim (match sweep-best risk head optim)
                    'stage2_lr': BEST["risk_lr"],
                    'stage2_weight_decay': BEST["risk_weight_decay"],

                    # workers (match sweep-best unless overridden)
                    'num_workers': BEST["num_workers"],
                }

                if zero_workers:
                    base_config['num_workers'] = 0

                for anchor in anchors:
                    if pbar:
                        pbar.set_description(f"UST (noise={noise}, seed={seed}, anchor={anchor})")

                    current_config = base_config.copy()
                    current_config['seed'] = seed
                    current_config['clean'] = anchor
                    current_config['best_model_path'] = f'./models/UST/noisy{noise}/anchor{anchor}/clean{anchor}_noisy{noise}_seed{seed}.pt'

                    return_code = run_experiment_ust(current_config, verbose=verbose)
                    if return_code != 0 and pbar:
                        pbar.write(f"Run FAILED for noise={noise}, seed={seed}, anchor={anchor}")
                    if pbar:
                        pbar.update(1)

        if pbar:
            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a series of experiments.")
    parser.add_argument("--experiment", type=str, choices=['BaselineB', 'UST'], help="The name of the experiment set to run.")
    parser.add_argument("--noises", nargs='+', type=int, default=None, help="List of noise levels to iterate through. Overrides the default.")
    parser.add_argument("--seeds", nargs='+', type=int, default=None, help="List of seeds to iterate through. Overrides the default.")
    parser.add_argument("--anchors", nargs='+', type=int, default=None, help="List of anchor percentages for the UST experiment. Overrides the default.")
    parser.add_argument("--zero_workers", action='store_true', help="If set, the number of data loader workers will be forced to 0.")
    parser.add_argument("--verbose", action='store_true', help="Print all output from child scripts for debugging.")

    args = parser.parse_args()

    loop_kwargs = {'exp': args.experiment, 'zero_workers': args.zero_workers, 'verbose': args.verbose}
    if args.noises is not None:
        loop_kwargs['noises'] = args.noises
    if args.seeds is not None:
        loop_kwargs['seeds'] = args.seeds
    if args.anchors is not None:
        loop_kwargs['anchors'] = args.anchors

    experiment_loop(**loop_kwargs) 