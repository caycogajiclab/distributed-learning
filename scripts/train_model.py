#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full CCNet Training Script

Created on 03/09/2024
Modified on 18/02/2026

@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

Description:
    Runs full CCNet training after cortex and cerebellum have been pretrained
    separately by pretrain_cortex.py and pretrain_cereb.py.
    Loads pretrained weights, applies a perturbation to the dataset, then
    trains the full model.
"""
import os
import sys
import shutil
from typing import Tuple
from copy import deepcopy
import json
import argparse
import numpy as np
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="Training CCNet script")
parser.add_argument("--config", type=str, required=True, 
                    help="Path to the JSON configuration file")
parser.add_argument("-f", "--force", action="store_true", default=False, 
                    help="Force execution without interruptions")
parser.add_argument("--id",    type=str, default=None, 
                    help="Running ID, for connecting job and saved data")
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"],
                    help="Device to use for training (set by run_jobs.py)")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if args.id:
    config['name'] = '.'.join([args.id, config['name']])
assert 'work_dir' in config, "Config must include 'work_dir' (where to save experiments)."
WORK_DIR = config.get("work_dir")
_job_registry = Path(WORK_DIR) / "job_registry.json"
_job_id       = int(args.id) if args.id is not None else None

from ccdl.utils import *
from ccdl.ccnet import CerebelloCorticalNet
from ccdl.trainer import CCNetTrainer
from ccdl.reaching import ReachingDataset
from script_utils import (
    build_defaults, merge_params, params_to_json_safe,
    save_script_and_params, get_pretrain_file, setup_trainer,
    prepare_training_settings, _registry_mark_iteration
)

config_file_keys = ['name', 'seed', 'rnn_hidden_size', 'model_class']
assert all([var in config.keys() for var in config_file_keys]), \
    f"The JSON file must provide {config_file_keys}. Please modify {args.config}"
assert config['name'], "Name must be provided in the JSON file"

_n = int(os.environ.get("OMP_NUM_THREADS", 1))
torch.set_num_threads(_n)
torch.set_num_interop_threads(1)

if args.device == 'cuda':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

dtype      = config.get('dtype', 'float32')
npdtype    = getattr(np, dtype)
torchdtype = getattr(torch, dtype)

def preparing_params(config_dict) -> Tuple[dict, list]:
    """
    Build the full params dict for training.

    Delegates shared logic (defaults, merge, device injection)
    to script_utils, then adds the train_settings block specific to this script.
    'train_params' is required in the config.

    Args:
        config_dict (dict): Configuration dictionary loaded from JSON.

    Returns:
        tuple: (params dict, list of keys that fell back to defaults)
    """
    model_class = config_dict['model_class']
    module      = globals()[model_class]

    defaults = build_defaults(module)
    defaults['cx_pretrain_path']    = None
    defaults['cb_pretrain_path']    = None
    defaults['perturbation_params'] = ReachingDataset.default_perturbation_parameters()

    params, used_defaults = merge_params(config_dict, defaults, device, torchdtype)

    assert 'train_params' in config_dict, \
        "'train_params' must be provided in the config for train_model.py."
    
    params['train_params'] = config_dict['train_params']
    
    return params, used_defaults


def training_method(name, folder, params, iteration=None) -> None:
    """
    Run one full-model training experiment.

    Optionally loads pretrained cortex and cerebellum, applies the perturbation,
    then calls trainer.train(). cx_pretrain_path and cb_pretrain_path are both
    optional: if None, the corresponding component is trained from scratch.
    When iterations > 1 and a pretrain path points to a directory, the file
    corresponding to the current iteration index is loaded.

    Args:
        name      (str):  Experiment name.
        folder    (str):  Parent folder.
        params    (dict): Full params dict (output of preparing_params).
        iteration (int):  Iteration index for multi-run experiments.
    """
    seed = params.get('seed', None)
    if seed is not None:
        iter_seed = seed if iteration is None else seed + iteration
        np.random.seed(iter_seed)
        torch.manual_seed(iter_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(iter_seed)
            
    if iteration in [None, 0]:
        save_script_and_params(__file__, folder, params, args)
    
    title = name
    if iteration is not None:
        name   = f'{iteration}'
        title += f'_{iteration}'

    experiment = Experiment2(folder, name, force=args.force)
    name    = experiment.name
    folder  = experiment.folder
    logfile = os.path.join(folder, "experiment.log")
    params  = deepcopy(params)

    def printfunc(text, formatting=None, end=None):
        printsave(text, logfile, True, formatting, end)

    printfunc('Experiment correctly initialized.')
    printfunc(f'Name: {title}\n')

    reaching = ReachingDataset(params['dataset_params'])
    experiment.save("ReachingDataset", reaching, is_parameter=True)
    trainer = setup_trainer(reaching, params)

    printfunc("\nRunning FULL MODEL training.\n", ['red', 'bold'])

    # Load pretrained cortex
    cx_path = params.get('cx_pretrain_path', None)
    id = iteration if iteration is not None else '0'
    if cx_path:
        cx_file = get_pretrain_file(cx_path, id)
        assert os.path.exists(cx_file), f"Pretrained cortex not found: {cx_file}"
        trainer.load_cortex(load_var(cx_file))
        printfunc(f"Loaded pretrained cortex from: {cx_file}")

    # Load pretrained cerebellum
    cb_path = params.get('cb_pretrain_path', None)
    if cb_path:
        cb_file = get_pretrain_file(cb_path, id, extension='.sd')
        try:
            cereb_sd = torch.load(cb_file, map_location=device)
        except Exception:
            raise IOError(
                f"Cerebellum model not found/unusable. "
                f"Please check the path or set it to null. path: '{cb_file}'"
            )
        trainer.load_cereb(cereb_sd)
        printfunc(f"Loaded pretrained cerebellum from: {cb_file}")

    # Apply perturbation
    printfunc("Applying perturbation:")
    printfunc(params['perturbation_params'])
    reaching.apply_perturbation(params['perturbation_params'])

    train_settings = prepare_training_settings(
        params, 'settings', logfile, title, tensorboard_suffix='_train'
    )
    experiment.save("settings", train_settings, is_parameter=True)

    outputs, performance = trainer.train(
        reaching,
        train_params=params['train_params'],
        model_state=params['model_state'],
        force=args.force,
        **train_settings
    )

    experiment.save("train_outputs", outputs)
    save_var(outputs['metrics'], os.path.join(folder, 'metrics.pkl'))
    experiment.update_performance("train_performance", performance)

    printfunc(f"Full training completed and data correctly saved (folder: {folder}).")
    del outputs, performance


if __name__ == '__main__':
    name       = config['name']
    iterations = config.get('iterations', 1)
    iterations = iterations if iterations > 0 else 1

    params, used_defaults = preparing_params(config)
    if used_defaults:
        print(f"Using default values for: {used_defaults}")

    param_to_iterate = config.get('param_to_iterate')
    if param_to_iterate:
        param_name        = param_to_iterate['name']
        values_to_iterate = param_to_iterate['values']
    else:
        param_name        = None
        values_to_iterate = [None]

    for vi, value in enumerate(values_to_iterate):
        val_params = deepcopy(params)
        val_name   = deepcopy(name)
        
        if param_name:
            keys    = param_name.split('.')
            current = val_params
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            str_value = f"{value:.0e}" if isinstance(value, float) else str(value)
            val_name  = '.'.join([val_name, str(keys[-1]), str_value])

        if iterations == 1:
            training_method(val_name, WORK_DIR, val_params)
            _registry_mark_iteration(
                _job_registry, _job_id,
                value_index=vi, iteration=0,
                n_values=len(values_to_iterate), iterations_per_value=1,
                job_name=val_name,
            )
        else:
            expected_folder = os.path.join(WORK_DIR, val_name)
            if os.path.isdir(expected_folder):
                folder = expected_folder
            else:
                folder, val_name = generate_folder2(WORK_DIR, val_name, args.force)

            cx_path = val_params.get('cx_pretrain_path', None)
            if cx_path and os.path.isdir(cx_path):
                assert len(os.listdir(cx_path)) >= iterations, \
                    "Number of pretrained cortex models < number of iterations"
            cb_path = val_params.get('cb_pretrain_path', None)
            if cb_path and os.path.isdir(cb_path):
                assert len(os.listdir(cb_path)) >= iterations, \
                    "Number of pretrained cerebellum models < number of iterations"
            
            # shutil.copy(args.config, os.path.join(folder, "config.json"))

            # Read already-completed iterations from registry (for resume)
            completed_pairs = set()
            if _job_id is not None and _job_registry.exists():
                try:
                    entries = json.loads(_job_registry.read_text())
                    for e in entries:
                        if e["job"] == _job_id:
                            completed_pairs = {
                            (p[0], p[1]) for p in e.get("completed", [])
                            if p[0] == vi          # only this value's iterations
                        }
                            break
                except Exception:
                    pass  # registry unreadable: run everything

            for j in range(iterations):
                if (vi, j) in completed_pairs:
                    print(f"  Value {vi}, iteration {j} already completed, skipping.")
                    continue
                training_method(val_name, folder, val_params, iteration=j)
                _registry_mark_iteration(
                    _job_registry, _job_id,
                    value_index=vi, iteration=j,
                    n_values=len(values_to_iterate), iterations_per_value=iterations,
                    job_name=val_name,
                )

    print("Done.")
