#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_continual.py

Continual learning training script for CCNet. Drop-in sibling of train_model.py.
CLI: --config, --id, --device, --force  (identical to train_model.py)

Config schema (new keys on top of the standard train_model template):

  "continual_task": [                         # option A: explicit phase list
      {"perturbation_params": {...}, "epochs": 500},
      ...
  ]

  OR

  "continual_task_params": {                  # option B: random generation (seeded)
      "seed": 70,
      "n_perts": 7,
      "range": [-40, 40, 9],                  # args for np.linspace
      "basic_pert": {
          "perturbation_params": {"angle": 40, "amplitude_factor": 1, "translation": [0,0]},
          "epochs": 500
      }
  }

  If both are present, "continual_task" takes priority.

  Each phase in continual_task must have:
      "perturbation_params": {...}
      "epochs": <int>
      "name": <str>   (optional, defaults to {i}, i=1,N)

  "train_params"  – shared, fixed across all phases and all models.
  "model_state"   – shared, fixed across all phases.

Outputs (saved once at the end):
  "train_outputs"  – dict keyed by phase name, each value = trainer.train() outputs
  "metrics"        – dict keyed by phase name
  "unperturbed_loss" – scalar, computed before any perturbation
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

parser = argparse.ArgumentParser(description="Continual CCNet Training Script")
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

# ── Required config keys ──────────────────────────────────────────────────────
_required_keys = ['name', 'seed', 'rnn_hidden_size', 'model_class']
assert all(k in config for k in _required_keys), \
    f"Config must provide {_required_keys}."
assert config['name'], "name must be non-empty."

# ── Thread / device setup ─────────────────────────────────────────────────────
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
npdtype  = getattr(np, dtype)
torchdtype = getattr(torch, dtype)


# ── continual_task resolution ─────────────────────────────────────────────────
def resolve_continual_task(config_dict: dict) -> list:
    """
    Return a list of phase dicts, each with keys:
        name, perturbation_params, epochs

    Priority: explicit 'continual_task' > generate from 'continual_task_params'.
    When generating, np.random is seeded so that running identical configs from
    different simulation entries always produces the same task sequence.
    """
    if 'continual_task' in config_dict:
        task = deepcopy(config_dict['continual_task'])
        for i, phase in enumerate(task):
            phase.setdefault('name', i)
        return task

    assert 'continual_task_params' in config_dict, \
        "Config must provide either 'continual_task' or 'continual_task_params'."

    tp = config_dict['continual_task_params']
    np.random.seed(tp['seed'])
    angles = np.linspace(*tp['range']).astype('int32')
    sequence = np.random.choice(angles, size=tp['n_perts'], replace=False)

    task = []
    for i, angle in enumerate(sequence):
        phase = deepcopy(tp['basic_pert'])
        phase['perturbation_params']['angle'] = int(angle)
        phase.setdefault('name', i)
        task.append(phase)
        
    # adding final washout phase
    phase = deepcopy(tp['basic_pert'])
    phase['perturbation_params']['angle'] = 0
    phase.setdefault('name', len(sequence))
    task.append(phase)
    
    return task


# ── Parameter preparation ─────────────────────────────────────────────────────
def preparing_params(config_dict: dict) -> Tuple[dict, list]:
    """Build the params dict from the JSON config for continual-learning training.

    Resolves the continual task sequence (:func:`resolve_continual_task`) on top
    of the shared defaults/merge logic. Returns (params, keys that used defaults).
    """
    model_class = config_dict['model_class']
    module = globals()[model_class]

    defaults = build_defaults(module)
    defaults['cx_pretrain_path'] = None
    defaults['cb_pretrain_path'] = None

    params, used_defaults = merge_params(config_dict, defaults, device, torchdtype)

    assert 'train_params' in config_dict, \
        "'train_params' must be provided in the config (shared across all phases)."

    params['train_params']     = config_dict['train_params']
    params['continual_task']   = resolve_continual_task(config_dict)

    return params, used_defaults


# ── Single-iteration training ─────────────────────────────────────────────────
def training_method(name: str, folder: str, params: dict, iteration=None) -> None:
    """Run one continual-learning training experiment over the task sequence and
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
    name   = experiment.name
    folder = experiment.folder
    logfile = os.path.join(folder, "experiment.log")
    params  = deepcopy(params)

    def printfunc(text, formatting=None, end=None):
        printsave(text, logfile, True, formatting, end)

    printfunc('Experiment correctly initialized.')
    printfunc(f'Name: {title}\n')

    # ── Dataset + trainer ────────────────────────────────────────────────────
    reaching = ReachingDataset(params['dataset_params'])
    experiment.save("ReachingDataset", reaching, is_parameter=True)
    trainer = setup_trainer(reaching, params)

    # ── Load pretrained cortex ────────────────────────────────────────────────
    id_str = str(iteration) if iteration is not None else '0'
    cx_path = params.get('cx_pretrain_path', None)
    if cx_path:
        cx_file = get_pretrain_file(cx_path, id_str)
        assert os.path.exists(cx_file), f"Pretrained cortex not found: {cx_file}"
        trainer.load_cortex(load_var(cx_file))
        printfunc(f"Loaded pretrained cortex from: {cx_file}")

    # ── Load pretrained cerebellum ────────────────────────────────────────────
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

    # ── Unperturbed baseline loss ─────────────────────────────────────────────
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
    phases     = params['continual_task']
    train_params = params['train_params']   # fixed across all phases and models

    printfunc(f"\nRunning CONTINUAL training ({len(phases)} phase(s)).\n",
              ['red', 'bold'])

    # Save the common settings once (they don't change per phase)
    train_settings = prepare_training_settings(
        params, 'settings', logfile, title
    )
    experiment.save("settings", train_settings, is_parameter=True)

    all_outputs, all_metrics, all_performance = {}, {}, {}

    for i, phase in enumerate(phases):
        phase_name   = phase.get('name', i)
        pert_params  = phase['perturbation_params']
        phase_epochs = phase['epochs']

        # Build per-phase train_params: same as base, only epochs may differ
        phase_train_params = deepcopy(train_params)
        phase_train_params['epochs'] = phase_epochs

        printfunc(
            f"\n{'='*56}\n Phase: {phase_name} ({phase_epochs} epochs)\n{'='*56}",
            ['bold', 'blue']
        )
        printfunc(f" Perturbation: {pert_params}")

        reaching.apply_perturbation(pert_params)

        outputs, performance = trainer.train(
            reaching,
            train_params=phase_train_params,
            model_state=params['model_state'],
            force=args.force,
            **train_settings
        )

        all_outputs[phase_name] = outputs
        all_metrics[phase_name] = outputs.get('metrics', None)
        all_performance[phase_name] = performance

        printfunc(f" Phase {i} ({phase_name}) completed.")

    # ── Save outputs once ─────────────────────────────────────────────────────
    experiment.save("train_outputs", all_outputs)
    experiment.save("metrics",       all_metrics)
    experiment.update_performance(f"train_performance", performance)
    save_var(all_metrics, os.path.join(folder, 'metrics.pkl'))

    printfunc(f"\nContinual training completed. Results in: {folder}",
              ['bold', 'green'])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    name = config['name']
    iterations = max(config.get('iterations', 1), 1)

    params, used_defaults = preparing_params(config)
    if params['settings']['debug_step']:
        print(f"Continual task generated:\n{params['continual_task']}")
    if used_defaults:
        print(f"Using default values for: {used_defaults}")

    param_to_iterate = config.get('param_to_iterate')
    if param_to_iterate:
        param_name = param_to_iterate['name']
        values_to_iterate = param_to_iterate['values']
    else:
        param_name = None
        values_to_iterate = [None]

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
            folder = expected_folder if os.path.isdir(expected_folder) \
                else generate_folder2(WORK_DIR, val_name, args.force)[0]

            for path_key, label in [('cx_pretrain_path', 'cortex'),
                                     ('cb_pretrain_path', 'cerebellum')]:
                p = val_params.get(path_key, None)
                if p and os.path.isdir(p):
                    assert len(os.listdir(p)) >= iterations, \
                        f"Fewer pretrained {label} models than iterations."

            # Resume: read completed iterations from registry
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
