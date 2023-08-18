import subprocess
import pymongo
import itertools
from local_coeff_computation import EXPT_NAME
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import random

def run_experiments(expt_configs):
    # Create combinations of the configurations
    config_combinations = list(itertools.product(
        expt_configs["model_filepath"], 
        expt_configs["sgld_gamma"], 
        expt_configs["sgld_num_iter"], 
        expt_configs["sgld_noise_std"])
    )
    print(f"Num experiments: {len(config_combinations)}")
    seed = expt_configs["seed"]
    # config_fp = expt_configs["config_filepath"]
    processes = []
    for model_fp, gamma, num_iter, noise_std in config_combinations:
        model_basedir = os.path.dirname(model_fp)
        config_fp = os.path.join(model_basedir, "commandline_args.json")
        cmd = [
            "python", "local_coeff_computation.py", 
            "compute_local_learning_coefficient", "with",
            f"model_filepath={model_fp}",
            f"config_filepath={config_fp}",
            f"sgld_gamma={gamma}", 
            f"sgld_num_iter={num_iter}",
            f"sgld_noise_std={noise_std}", 
            f"seed={seed}"
        ]
        process = subprocess.Popen(cmd)
        processes.append(process)
        # Limit the number of concurrent processes
        if len(processes) >= 6:
            for process in processes:
                process.wait()
            processes = []
    # If there are remaining processes, wait for them
    for process in processes:
        process.wait()

if __name__ == '__main__':
    
    directories = [
        "./spartan_outputs/expt20230814_60000_512_t200_lr0.01_SGLDITER400_GAMMA100_sgd",
        "./spartan_outputs/expt20230814_60000_512_t200_lr0.01_SGLDITER400_GAMMA100_entropy-sgd"
    ]
    model_fps = []
    for directory in directories:
        model_fps += [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')][:30]
    random.shuffle(model_fps)
    print(f"Total num models: {len(model_fps)}")

    expt_configs = {
        "model_filepath": model_fps,
        "sgld_gamma": [10.0, 100.0, 1000.0],
        "sgld_num_iter": [100, 250, 400, 600, 800, 1000],
        "sgld_noise_std": [1e-5], 
        "seed": 49
    }

    run_experiments(expt_configs)