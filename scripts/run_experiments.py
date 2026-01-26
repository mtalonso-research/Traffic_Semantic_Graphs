import subprocess
import os
import sys
import argparse
from tqdm import tqdm

def run_experiment_baselineb(config):
    """
    Constructs and runs a command for a BaselineB experiment.
    Output is suppressed unless an error occurs.
    """
    script_name = config['script']
    base_command = ["python", "-m", f"scripts.{script_name}"]
    args = []

    arg_map = {
        'l2d': '--l2d', 'nup': '--nup', 'clean': '--clean', 'noisy': '--noisy',
        'seed': '--seed', 'prediction_mode': '--prediction_mode', 'num_workers': '--num_workers',
        'evaluate': '--evaluate', 'train_autoencoder': '--train_autoencoder', 'train_risk': '--train_risk',
        'best_model_path': '--best_model_path', 'ae_epochs': '--ae_epochs', 'risk_epochs': '--risk_epochs',
    }

    for key, value in config.items():
        if key == 'script': continue
        flag = arg_map.get(key)
        if flag:
            if isinstance(value, bool) and value:
                args.append(flag)
            elif not isinstance(value, bool):
                args.extend([flag, str(value)])

    full_command = base_command + args
    try:
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

def run_experiment_ust(config):
    """
    Constructs and runs a command for a UST experiment.
    Output is suppressed unless an error occurs.
    """
    script_name = config['script']
    base_command = ["python", f"scripts/{script_name}.py"]
    args = []

    arg_map = {
        'train_stage2': '--train_stage2', 'evaluate': '--evaluate', 'nup': '--nup',
        'clean': '--clean', 'noisy': '--noisy', 'load_best_ae_clean': '--load_best_ae_clean',
        'ae_clean_ckpt_path': '--ae_clean_ckpt_path', 'load_best_ae_noisy': '--load_best_ae_noisy',
        'ae_noisy_ckpt_path': '--ae_noisy_ckpt_path', 'best_model_path': '--best_model_path',
        'num_workers': '--num_workers', 'embed_dim': '--embed_dim', 'hidden_dim': '--hidden_dim',
        'num_encoder_layers': '--num_encoder_layers', 'num_decoder_layers': '--num_decoder_layers',
        'risk_hidden_dim': '--risk_hidden_dim', 'stage2_epochs': '--stage2_epochs',
        'stage2_lr': '--stage2_lr', 'stage2_weight_decay': '--stage2_weight_decay', 'seed': '--seed',
    }

    for key, value in config.items():
        if key == 'script': continue
        flag = arg_map.get(key)
        if flag:
            if isinstance(value, bool) and value:
                args.append(flag)
            elif not isinstance(value, bool):
                args.extend([flag, str(value)])

    full_command = base_command + args
    try:
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


def experiment_loop(exp,noises=[5,10,15,20,25,30,35,40,45,50,55,60],
                    seeds=[42, 1024, 31, 25, 334, 723, 567, 7, 121, 9],
                    anchors=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
                    zero_workers=False):
    if exp == 'BaselineB':
        total_runs = len(noises) * len(seeds)
        with tqdm(total=total_runs, desc="Running BaselineB Experiments") as pbar:
            for noise in noises:
                base_config = {
                    'script': '4A_ae_risk', 'nup': True, 'noisy': noise, 'prediction_mode': 'classification',
                    'train_autoencoder': True, 'train_risk': True, 'evaluate': True, 'hidden_dim': 64,
                    'embed_dim': 254, 'num_encoder_layers': 2, 'num_decoder_layers': 1, 'activation': 'relu',
                    'dropout_rate': 0.1, 'risk_hidden_dim': 32, 'quant_bins': 32, 'ae_epochs': 60,
                    'ae_lr': 8.913123740996388e-05, 'ae_weight_decay': 5.552716770961925e-06, 'risk_epochs': 20,
                    'risk_lr': 0.00029895883375268494, 'risk_weight_decay': 1.3705141698203051e-06,
                }
                if zero_workers: base_config['num_workers'] = 0

                for seed_value in seeds:
                    pbar.set_description(f"BaselineB (noise={noise}, seed={seed_value})")
                    current_config = base_config.copy()
                    current_config['best_model_path'] = f'./models/4B/4B_noisy{noise}_class_best_model_{seed_value}.pt'
                    current_config['seed'] = seed_value
                    
                    return_code = run_experiment_baselineb(current_config)
                    if return_code != 0:
                        pbar.write(f"Run FAILED for noise={noise}, seed={seed_value}")
                    pbar.update(1)

    elif exp == 'UST':
        total_runs = len(noises) * len(seeds) * len(anchors)
        with tqdm(total=total_runs, desc="Running UST Experiments") as pbar:
            for noise in noises:
                for seed in seeds:
                    base_config = {
                        'script': '5A_ust_risk', 'train_stage2': True, 'evaluate': True, 'nup': True,
                        'load_best_ae_clean': True, 'load_best_ae_noisy': True, 'noisy': noise,
                        'ae_clean_ckpt_path': f'./models/4B/clean/4B_clean_class_best_model_{seed}.pt',
                        'ae_noisy_ckpt_path': f'./models/4B/noisy{noise}/4B_noisy{noise}_class_best_model_{seed}.pt',
                        'embed_dim': 254, 'hidden_dim': 64, 'num_encoder_layers': 2, 'num_decoder_layers': 1,
                        'risk_hidden_dim': 32, 'stage2_epochs': 20, 'stage2_lr': 0.00029895883375268494,
                        'stage2_weight_decay': 1.3705141698203051e-06,
                    }
                    if zero_workers: base_config['num_workers'] = 0

                    for anchor in anchors:
                        pbar.set_description(f"UST (noise={noise}, seed={seed}, anchor={anchor})")
                        current_config = base_config.copy()
                        current_config['seed'] = seed
                        current_config['clean'] = anchor
                        current_config['best_model_path'] = f'./models/4C/4C_clean{anchor}_noisy{noise}_best_model_{seed}.pt'
                        
                        return_code = run_experiment_ust(current_config)
                        if return_code != 0:
                            pbar.write(f"Run FAILED for noise={noise}, seed={seed}, anchor={anchor}")
                        pbar.update(1)

def main():
    """
    Defines the command-line interface and runs the main experiment loop.
    """
    parser = argparse.ArgumentParser(description="Run a series of experiments.")
    parser.add_argument("experiment", type=str, choices=['BaselineB', 'UST'], help="The name of the experiment set to run.")
    parser.add_argument("--noises", nargs='+', type=int, default=None, help="List of noise levels to iterate through. Overrides the default.")
    parser.add_argument("--seeds", nargs='+', type=int, default=None, help="List of seeds to iterate through. Overrides the default.")
    parser.add_argument("--anchors", nargs='+', type=int, default=None, help="List of anchor percentages for the UST experiment. Overrides the default.")
    parser.add_argument("--zero_workers", action='store_true', help="If set, the number of data loader workers will be forced to 0.")

    args = parser.parse_args()

    loop_kwargs = {'exp': args.experiment, 'zero_workers': args.zero_workers}
    if args.noises is not None: loop_kwargs['noises'] = args.noises
    if args.seeds is not None: loop_kwargs['seeds'] = args.seeds
    if args.anchors is not None: loop_kwargs['anchors'] = args.anchors
    
    experiment_loop(**loop_kwargs)

if __name__ == "__main__":
    main()

