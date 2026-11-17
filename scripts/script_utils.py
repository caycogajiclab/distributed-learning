# -*- coding: utf-8 -*-
"""
Created on 18/02/2026

@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

Shared utilities for the training scripts.
Provides parameter building, serialisation, trainer setup, and I/O helpers
that are identical across scripts.
"""

import os
import json
import shutil
from copy import deepcopy
from typing import Tuple
from pathlib import Path
import fcntl

import torch

from ccdl.trainer import CCNetTrainer
from ccdl.pre_trainer import CCNetPreTrainer
from ccdl.trainer_damage import CCNetDamageTrainer
from ccdl.reaching import ReachingDataset

def build_defaults(model_class_module) -> dict:
    """
    Return the full defaults dict for all optional config keys.

    Net params are collected from any method named 'default*' on the model
    class module, excluding plasticity-related ones (which depend on structure).

    Args:
        model_class_module: The imported model class (e.g. CerebelloCorticalNet).
        device (torch.device): Injected as default into dataset/net params.
        torchdtype: Torch dtype injected as default into dataset/net params.

    Returns:
        dict: Defaults for all optional config keys.
    """
    default_net_param_names = [
        x for x in dir(model_class_module)
        if 'default' in x
        and x not in ['default_plasticity_specs', 'default_modules_plasticity']
    ]
    return {
        'net_params':     {n: getattr(model_class_module, n)() for n in default_net_param_names},
        'dataset_params': ReachingDataset.default_parameters(),
        'model_state': {
            'cereb_feedback':   True,
            'cereb_readout':    True,
            'plasticity_specs': None,
            'plant':            None,
        },
        'settings': {
            'opt':              None,
            'outsave_step':     200,
            'print_steps':      20,
            'save_w_steps':     100,
            'record_gradients': False,
            'verbose':          False,
            'debug':            0,
            'tensorboard_params': {}
        },
    }


def merge_params(config_dict, defaults, device, torchdtype) -> Tuple[dict, list]:
    """
    Build the base params dict shared by all training scripts.

    Config values take priority over defaults. device, dtype, and ct are
    injected via setdefault so explicit values in the config are respected.
    RecCCNet cereb_specs are assembled here if model_class is RecCCNet.

    Args:
        config_dict (dict):  Config loaded from JSON.
        defaults    (dict):  Output of build_defaults(), optionally extended.
        device      (torch.device): Runtime device.
        torchdtype:          Runtime torch dtype.

    Returns:
        tuple: (params dict, list of keys that fell back to defaults)
    """
    
    model_class = config_dict['model_class']

    params = {
        'name':            config_dict['name'],
        'rnn_hidden_size': config_dict['rnn_hidden_size'],
        'model_class':     model_class,
    }
    
    if 'seed' in config_dict:
        params['seed'] = config_dict['seed']
    
    if 'iterations' in config_dict:
        params['iterations'] = config_dict['iterations']

    used_defaults = []
    for key in defaults:
        if key in config_dict:
            params[key] = config_dict[key]
        else:
            params[key] = deepcopy(defaults[key])
            used_defaults.append(key)

    params['dataset_params'].setdefault('device', device)
    params['dataset_params'].setdefault('dtype',  torchdtype)
    params['net_params'].setdefault('device', device)
    params['net_params'].setdefault('dtype',  torchdtype)

    if model_class == 'RecCCNet':
        assert 'cereb_size' in config_dict, \
            "Please provide 'cereb_size' in the config for RecCCNet"
        params['net_params']['cereb_specs'] = [
            config_dict['cereb_size'],
            params['net_params']['hidden_layer_specs'],
            params['net_params']['rnn_specs'],
        ]

    return params, used_defaults


def prepare_training_settings(params, settings_key, logfile, title, tensorboard_suffix='') -> dict:
    """
    Build the runtime settings dict passed to trainer.train().

    Takes the base settings from params[settings_key], which already contains
    train_params and model_state (set in preparing_params), and injects the
    runtime-only fields that are only available inside training_method:
    logfile, text label, and the resolved tensorboard log_dir.

    The tensorboard folder is cleared if it already exists.

    Args:
        params           (dict): Full params dict.
        settings_key     (str):  Key to read from params ('settings' usually).
        logfile          (str):  Path to the experiment log file.
        title            (str):  Human-readable experiment title.
        tensorboard_suffix (str): Appended to the tensorboard log_dir subfolder name.

    Returns:
        dict: Settings dict ready to be unpacked into trainer.train().
    """
    import shutil
    settings = deepcopy(params[settings_key])
    settings['logfile'] = logfile
    settings['text']    = title

    if settings.get('tensorboard_params',{}) != {}:
        tb_base = settings['tensorboard_params']['log_dir']
        tb_dir  = os.path.join(tb_base, title + tensorboard_suffix)
        settings['tensorboard_params']['log_dir'] = tb_dir
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)

    return settings


def params_to_json_safe(params) -> dict:
    """Return a JSON-serialisable copy of params, with device/dtype as strings."""
    temp = deepcopy(params)
    if temp['dataset_params'].get('dtype'):
        temp['dataset_params']['dtype'] = str(temp['dataset_params']['dtype'])
    if temp['net_params'].get('dtype'):
        temp['net_params']['dtype'] = str(temp['net_params']['dtype'])
    if temp['dataset_params'].get('device'):
        temp['dataset_params']['device'] = str(temp['dataset_params']['device'])
    if temp['net_params'].get('device'):
        temp['net_params']['device'] = str(temp['net_params']['device'])
    return temp


def save_script_and_params(script_file, folder, params, args, save_script=True):
    """
    Save a copy of the calling script and the params (pkl + json) to folder.

    Args:
        script_file (str):  Path to the calling script (__file__).
        folder      (str):  Destination folder.
        params      (dict): Params dict to save.
        args:               Parsed argparse namespace (needs args.force).
        save_script (bool): Whether to save a copy of the calling script.
    """
    if save_script:
        try:
            script_dest = os.path.join(folder, 'script.py')
            shutil.copy(script_file, script_dest)
            with open(script_dest, 'a') as f:
                f.write('# original script path: ' + script_file)
        except Exception:
            if args.force:
                raise FileNotFoundError("Error occurred while copying the script.")
            else:
                input(f"{script_file} not found, you are probably using interactive console. "
                    "Press ENTER to continue, CTRL+C to exit.")
    
    with open(os.path.join(folder, "params.json"), 'w') as f:
        json.dump(params_to_json_safe(params), f, indent=4)
        f.truncate()


def get_pretrain_file(path, iteration=None, extension='') -> str:
    """
    Resolve a pretrain path to a concrete file.

    If path is a directory and iteration is given, resolves to path/<iteration>[extension].
    Otherwise returns path as-is (single-file case).

    Args:
        path      (str): File path or directory.
        iteration (int): Current iteration index.
        extension (str): File extension to append when resolving from a directory (e.g. '.sd').

    Returns:
        str: Resolved file path.
    """
    if iteration is not None and os.path.isdir(path):
        filename = f"{iteration}{extension}" if extension else str(iteration)
        return os.path.join(path, filename)
    return path


def setup_trainer(reaching, params, pretrain=False, damage=False):
    """
    Initialise a CCNetCxLocalTrainer from a ReachingDataset and params dict.

    Args:
        reaching: ReachingDataset instance (used to infer input size).
        params (dict): Must contain net_params, rnn_hidden_size, model_class.

    Returns:
        CCNetCxLocalTrainer: Initialised trainer.
    """
    net_params         = params['net_params']
    net_params['args'] = [reaching.input_size[2], params['rnn_hidden_size']]
    if pretrain:
        trainer_class = CCNetPreTrainer 
    else:
        if damage:
            trainer_class = CCNetDamageTrainer
        else: 
            trainer_class = CCNetTrainer
    
    return trainer_class(
        net_params,
        model_class=params['model_class'],
        opt=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        verbose=params['settings']['verbose'],
    )

def _registry_mark_iteration(job_registry, job_id, value_index: int, iteration: int,
                              n_values: int, iterations_per_value: int,
                              job_name: str = None):
    """Mark one (value_index, iteration) pair as completed."""
    if job_id is None or not job_registry.exists():
        return
    lock_path = str(job_registry) + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            entries = json.loads(job_registry.read_text())
            for e in entries:
                if e["job"] == job_id:
                    total = n_values * iterations_per_value
                    if e.get("total") is None:
                        e["total"] = total
                    if job_name and e.get("name") is None:
                        e["name"] = job_name
                    pair = [value_index, iteration]
                    if pair not in e["completed"]:
                        e["completed"].append(pair)
                    break
            job_registry.write_text(json.dumps(entries, indent=2))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)