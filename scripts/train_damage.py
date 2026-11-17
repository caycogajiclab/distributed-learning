#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_damage_windows.py

Normal + damage training across multiple damage times.
Drop-in sibling of train_washout.py / train_continual.py / train_model.py.
CLI: --config, --id, --device, --force (identical).

Expected config additions:
"total_epochs": 300
"damage_epochs_list": [0, 40, 200]
"damage_model_state": {...}
"train_params": {
    ...
    "total_epochs": 5000
}

For each damage_epoch D the script runs `iterations` seeds, producing one
subfolder per D under the same parent folder. Each run is split as:
1. normal phase: D
2. damage phase: total_epochs - D

Here damage_epoch means the onset time of damage, i.e. the number of epochs
run in the normal condition before switching to damage_model_state.

Registry tracks (vi, di) pairs: vi = param_to_iterate value index,
di = damage-time index. Completed pairs are skipped on resume.
"""

import os
import sys
from typing import Tuple
from copy import deepcopy
import json
import argparse
import numpy as np
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="Damage-window CCNet Training Script")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("-f", "--force", action="store_true", default=False)
parser.add_argument("--id", type=str, default=None)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if args.id:
    config['name'] = '.'.join([args.id, config['name']])
assert 'work_dir' in config, "Config must include 'work_dir'."
WORK_DIR = config.get("work_dir")
_job_registry = Path(WORK_DIR) / "job_registry.json"
_job_id = int(args.id) if args.id is not None else None

from ccdl.utils import *
from ccdl.ccnet import CerebelloCorticalNet
from ccdl.trainer import CCNetTrainer
from ccdl.reaching import ReachingDataset
from script_utils import (
    build_defaults, merge_params, params_to_json_safe,
    save_script_and_params, get_pretrain_file, setup_trainer,
    prepare_training_settings, _registry_mark_iteration
)

_required_keys = ['name', 'seed', 'rnn_hidden_size', 'model_class']
assert all(k in config for k in _required_keys), \
    f"Config must provide {_required_keys}."
assert config['name'], "name must be non-empty."

_n = int(os.environ.get("OMP_NUM_THREADS", 1))
torch.set_num_threads(_n)
torch.set_num_interop_threads(1)

if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

dtype = config.get('dtype', 'float32')
npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)


def preparing_params(config_dict: dict) -> Tuple[dict, list]:
    """Build the params dict from the JSON config for damage training.

    Adds the damage-specific settings (``damage_model_state``, ``total_epochs``,
    ``damage_epochs_list``) on top of the shared defaults/merge logic. Returns
    (params, keys that used defaults).
    """
    model_class = config_dict['model_class']
    module = globals()[model_class]

    defaults = build_defaults(module)
    defaults['cx_pretrain_path'] = None
    defaults['cb_pretrain_path'] = None
    defaults['perturbation_params'] = ReachingDataset.default_perturbation_parameters()

    params, used_defaults = merge_params(config_dict, defaults, device, torchdtype)

    assert 'train_params' in config_dict, \
        "'train_params' must be provided in the config."
    assert 'damage_model_state' in config_dict, \
        "'damage_model_state' must be provided in the config."
    assert 'total_epochs' in config_dict, \
        "'total_epochs' must be provided in the config."

    params['train_params'] = config_dict['train_params']
    params['damage_model_state'] = config_dict['damage_model_state']
    params['total_epochs'] = config_dict['total_epochs']
    params['damage_epochs_list'] = config_dict.get(
        'damage_epochs_list', [0, 40, 200]
    )

    return params, used_defaults


def training_method(name: str, folder: str, params: dict, iteration=None) -> None:
    """Run one normal+damage training experiment across the damage times and
    save its outputs to ``folder``."""
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
        name = f'{iteration}'
        title += f'_{iteration}'

    experiment = Experiment2(folder, name, force=args.force)
    name = experiment.name
    folder = experiment.folder
    logfile = os.path.join(folder, "experiment.log")
    params = deepcopy(params)

    def printfunc(text, formatting=None, end=None):
        printsave(text, logfile, True, formatting, end)

    printfunc('Experiment correctly initialized.')
    printfunc(f'Name: {title}\n')

    reaching = ReachingDataset(params['dataset_params'])
    experiment.save("ReachingDataset", reaching, is_parameter=True)
    trainer = setup_trainer(reaching, params, damage=True)
    
    reaching.apply_perturbation(params['perturbation_params'])
    

    id_str = str(iteration) if iteration is not None else '0'
    cx_path = params.get('cx_pretrain_path', None)
    if cx_path:
        cx_file = get_pretrain_file(cx_path, id_str)
        assert os.path.exists(cx_file), f"Pretrained cortex not found: {cx_file}"
        trainer.load_cortex(load_var(cx_file))
        printfunc(f"Loaded pretrained cortex from: {cx_file}")

    cb_path = params.get('cb_pretrain_path', None)
    if cb_path:
        cb_file = get_pretrain_file(cb_path, id_str, extension='.sd')
        try:
            cereb_sd = torch.load(cb_file, map_location=device)
        except Exception:
            raise IOError(
                f"Cerebellum model not found/unusable. "
                f"Please check the path or set it to null. path: '{cb_file}'"
            )

        trainer.load_cereb(cereb_sd)
        printfunc(f"Loaded pretrained cerebellum from: {cb_file}")

    with torch.no_grad():
        trainer.model.eval(**trainer.cx_only_state)
        unpert_inputs, unpert_targets = reaching.generate_unperturbed_dataset(
            params['settings'].get('datatype', None)
        )

        unpert_outputs, _, _ = trainer._model_call(unpert_inputs)
        unpert_mse = trainer.criterion(unpert_outputs, unpert_targets).item()
    experiment.save("unperturbed_loss", unpert_mse)
    printfunc(f"Unperturbed baseline loss: {unpert_mse:.6e}")

    # ── Phase loop ────────────────────────────────────────────────────────────
    train_params = params['train_params']
    train_settings = prepare_training_settings(params, 'settings', logfile, title)
    experiment.save("settings", train_settings, is_parameter=True)

    all_outputs, all_metrics, all_performance = {}, {}, {}

    total_epochs = params['total_epochs']
    damage_epoch = params['damage_epoch']
    normal_epochs = damage_epoch
    damage_epochs = total_epochs - damage_epoch

    assert total_epochs >= damage_epoch >= 0
    assert damage_epochs >= 0

    phases = [
        {'name': 'normal', 'epochs': normal_epochs, 'model_state': params['model_state']},
        {'name': 'damage', 'epochs': damage_epochs, 'model_state': params['damage_model_state']},
    ]

    printfunc(f"\nRunning DAMAGE training ({len(phases)} phase(s)).\n",
             ['red', 'bold'])

    for i, phase in enumerate(phases):
        phase_name = phase.get('name', i)
        phase_epochs = phase['epochs']

        phase_train_params = deepcopy(train_params)
        phase_train_params['epochs'] = phase_epochs

        printfunc(
            f"\n{'='*56}\n Phase: {phase_name} ({phase_epochs} epochs)\n{'='*56}",
            ['bold', 'blue']
        )

        if phase_epochs == 0:
            all_outputs[phase_name] = None
            all_metrics[phase_name] = None
            all_performance[phase_name] = {
                'epochs': 0,
                'best_loss': None,
                'tot_epochs': 0
            }
            printfunc(f" Phase {i} ({phase_name}) skipped because epochs = 0.")
            continue

        phase_train_params = deepcopy(train_params)
        phase_train_params['epochs'] = phase_epochs

        outputs, performance = trainer.train(
            reaching,
            train_params=phase_train_params,
            model_state=phase['model_state'],
            force=args.force,
            **train_settings
        )

        all_outputs[phase_name] = outputs
        all_metrics[phase_name] = outputs.get('metrics', None)
        all_performance[phase_name] = performance

        printfunc(f" Phase {i} ({phase_name}) completed.")

    experiment.save("train_outputs", all_outputs)
    experiment.save("metrics", all_metrics)
    experiment.update_performance("train_performance", performance)
    save_var(all_metrics, os.path.join(folder, 'metrics.pkl'))

    printfunc(f"\nDamage training completed. Results in: {folder}",
             ['bold', 'green'])


if __name__ == '__main__':
    name = config['name']
    iterations = max(config.get('iterations', 1), 1)

    params, used_defaults = preparing_params(config)
    if params['settings']['debug_step']:
        print(f"Damage windows: {params['damage_epochs_list']}")
    if used_defaults:
        print(f"Using default values for: {used_defaults}")

    damage_epochs_list = params['damage_epochs_list']
    n_damage = len(damage_epochs_list)

    param_to_iterate = config.get('param_to_iterate')
    if param_to_iterate:
        param_name = param_to_iterate['name']
        values_to_iterate = param_to_iterate['values']
    else:
        param_name = None
        values_to_iterate = [None]

    n_values = len(values_to_iterate)

    for vi, value in enumerate(values_to_iterate):
        val_params = deepcopy(params)
        val_name = deepcopy(name)

        if param_name:
            keys = param_name.split('.')
            current = val_params
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            str_value = f"{value:.0e}" if isinstance(value, float) else str(value)
            val_name = '.'.join([val_name, str(keys[-1]), str_value])

        expected_folder = os.path.join(WORK_DIR, val_name)
        folder = expected_folder if os.path.isdir(expected_folder) \
            else generate_folder2(WORK_DIR, val_name, args.force)[0]

        for path_key, label in [('cx_pretrain_path', 'cortex'),
                                ('cb_pretrain_path', 'cerebellum')]:
            p = val_params.get(path_key, None)
            if p and os.path.isdir(p):
                assert len(os.listdir(p)) >= iterations, \
                    f"Fewer pretrained {label} models than iterations."

        completed_pairs = set()
        if _job_id is not None and _job_registry.exists():
            try:
                entries = json.loads(_job_registry.read_text())
                for e in entries:
                    if e["job"] == _job_id:
                        completed_pairs = {
                            (p[0], p[1]) for p in e.get("completed", [])
                            if p[0] == vi
                        }
                        break
            except Exception:
                pass

        total_epochs = val_params['total_epochs']

        for di, damage_epoch in enumerate(damage_epochs_list):
            if (vi, di) in completed_pairs:
                print(f" Value {vi}, damage_epoch {damage_epoch} already completed, skipping.")
                continue

            assert total_epochs >= damage_epoch >= 0
            normal_epochs = damage_epoch
            damage_epochs = total_epochs - damage_epoch
            assert damage_epochs >= 0

            print(f"\n{'='*80}\n {val_name} | damage_epoch={damage_epoch} "
                  f"({di+1}/{n_damage})\n{'='*80}")

            val_params['damage_epoch'] = damage_epoch

            if val_params['settings']['debug_step']:
                print(
                    f"total_epochs={total_epochs}, damage_epoch={damage_epoch}, "
                    f"normal_epochs={normal_epochs}, damage_epochs={damage_epochs}"
                )
            damage_folder = generate_folder2(folder, f"{damage_epoch}", args.force)[0]

            for j in range(iterations):
                training_method(val_name, damage_folder, val_params, iteration=j)

            _registry_mark_iteration(
                _job_registry, _job_id,
                value_index=vi, iteration=di,
                n_values=n_values, iterations_per_value=n_damage,
                job_name=f"{val_name}/{damage_epoch}",
            )

    print("Done.")
