# -*- coding: utf-8 -*-
"""
pretrain_cortex.py
"""

import os
import sys
import json
import copy
import shutil
import itertools
import argparse
from datetime import datetime
from copy import deepcopy
from ccdl.ccnet import CerebelloCorticalNet

# ---------------------------------------------------------------------------
# Argument parsing + sys.path setup at module level,
# mirroring the per-process import order of the original training_template.py
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cortex pretraining script")
parser.add_argument("--home",        required=True,  help="Project home directory (the repository root)")
parser.add_argument("--workdir",     required=True,  help="Base data directory (e.g. the repo's data/ folder)")
parser.add_argument("--output",      required=True,  help="Output folder name, relative to --workdir")
parser.add_argument("--config-dir",  required=True,  help="Folder containing template_config.json and configs_list.json")
parser.add_argument("-f", "--force", action="store_true", default=False)
args = parser.parse_args()

HOME    = args.home
WORKDIR = args.workdir
print("using home path: " + HOME)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import torch
import numpy as np

from ccdl.utils import *

from ccdl.pre_trainer import CCNetPreTrainer
from ccdl.reaching import ReachingDataset
from ccdl.paths import DATA_DIR


# ---------------------------------------------------------------------------
# Config preparation
# ---------------------------------------------------------------------------

def _generate_params_combinations(starting_dictionary, iter_params):
    if isinstance(iter_params, dict):
        combinations = list(itertools.product(*iter_params.values()))
        combo_keys   = [iter_params.keys() for _ in range(len(combinations))]
    elif isinstance(iter_params, list):
        combinations = [list(p.values()) for p in iter_params]
        combo_keys   = [p.keys()         for p in iter_params]
    else:
        raise IOError('Please provide either grid_dictionary or train_list')

    updated_dicts = []
    for combo, keys in zip(combinations, combo_keys):
        updated_dict = copy.deepcopy(starting_dictionary)
        for i, key in enumerate(keys):
            parts = key.split('.')
            cur = updated_dict
            for subkey in parts[:-1]:
                if subkey not in cur:
                    cur[subkey] = {}
                cur = cur[subkey]
            cur[parts[-1]] = combo[i]
        updated_dicts.append(updated_dict)
    return updated_dicts


def prepare_configs(configs_list_file, template_config_file, output_dir, home):
    with open(configs_list_file, 'r') as f:
        params_conf_list = json.load(f)
    template_params = json.load(open(template_config_file))
    template_params['home']     = home
    template_params['work_dir'] = output_dir
    params_list = _generate_params_combinations(template_params, params_conf_list)

    configs_dir = os.path.join(output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    specs = {}
    for index, (full_conf, new_conf) in enumerate(zip(params_list, params_conf_list)):
        config_file_path = os.path.join(configs_dir, f'config_{index}.json')
        with open(config_file_path, 'w') as cf:
            json.dump(full_conf, cf, indent=4)
            cf.truncate()
        specs[f'config_{index}.json'] = new_conf
    json.dump(specs, open(os.path.join(output_dir, 'specs.json'), 'w'), indent=4)
    return params_list


# ---------------------------------------------------------------------------
# preparing_params
# ---------------------------------------------------------------------------

def preparing_params(config_dict, device, torchdtype):
    model_class = config_dict['model_class']

    def get_default_params(name=None):
        module = globals()[model_class]
        default_net_params_dicts = list(filter(lambda x: 'default' in x, dir(module)))
        default_values = {
            'cx_pretrain_path': None,
            'cb_pretrain_path': None,
            'net_params': {
                n: getattr(module, n)()
                for n in default_net_params_dicts
                if n not in ['default_plasticity_specs', 'default_modules_plasticity']
            },
            'dataset_params':      ReachingDataset.default_parameters(),
            'perturbation_params': ReachingDataset.default_perturbation_parameters(),
            'model_state': {
                'cereb_feedback':   True,
                'cereb_readout':    True,
                'plasticity_specs': None,
                'plant':            None,
            },
            'settings': {
                'batch_train':      False,
                'opt':              None,
                'outsave_step':     200,
                'print_steps':      20,
                'save_w_steps':     100,
                'record_gradients': False,
                'verbose':          False,
                'debug':            0,
                'tensorboard_params': {},
            },
        }
        default_values['dataset_params'].update({'device': device, 'dtype': torchdtype})
        default_values['net_params'].update({'ct': False, 'device': device, 'dtype': torchdtype})

        if name is None:
            return default_values
        assert name in default_values, f"The name {name} is not present in the default parameters"
        return default_values[name]

    default_config_available_params = get_default_params().keys()

    params = {
        'name':            config_dict['name'],
        'rnn_hidden_size': config_dict['rnn_hidden_size'],
        'model_class':     model_class,
    }
    if 'iterations' in config_dict:
        params['iterations'] = config_dict['iterations']

    params.update({
        name: config_dict[name] if name in config_dict else get_default_params(name)
        for name in default_config_available_params
    })

    used_default_params = [n for n in default_config_available_params if n not in config_dict]

    params['dataset_params'].update({'device': device, 'dtype': torchdtype})
    params['net_params'].update({'ct': False, 'device': device, 'dtype': torchdtype})

    if 'pretrain_params' in config_dict:
        params['pretrain_settings'] = config_dict['settings']
        extra_params = {
            'train_params': config_dict['pretrain_params'],
            'model_state':  params['model_state'],
        }
        params['pretrain_settings'].update(extra_params)

    if 'train_params' in config_dict:
        params['train_settings'] = config_dict['settings']
        extra_params = {
            'train_params': config_dict['train_params'],
            'model_state':  params['model_state'],
        }
        params['train_settings'].update(extra_params)

    return params, used_default_params


# ---------------------------------------------------------------------------
# training_method
# ---------------------------------------------------------------------------

def training_method(name, folder, params, device, force, iteration=None):
    if iteration is not None:
        name = f'{iteration}'

    experiment = Experiment2(folder, name, force=force)
    name   = experiment.name
    folder = experiment.folder

    logfile = os.path.join(folder, "experiment.log")
    params  = deepcopy(params)

    def printfunc(text, formatting=None, end=None):
        printsave(text, logfile, True, formatting, end)

    printfunc('Experiment correctly initialized.')
    printfunc(f'Name: {name}\n')

    try:
        script_file = __file__
        script_name = os.path.join(folder, 'script.py')
        shutil.copy(script_file, script_name)
        with open(script_name, 'a') as f:
            f.write('# original script path: ' + script_file)
    except Exception:
        if force:
            raise FileNotFoundError("Error occurred while copying the script.")
        else:
            input(
                f"{__file__} not found, you are probably using interactive console. If not, "
                "and you want to save a copy of the script, please check which is the issue. "
                "\nPress ENTER to continue, CTRL+C to exit."
            )

    save_var(params, os.path.join(folder, "params.pkl"))

    temp_params = deepcopy(params)
    temp_params['dataset_params']['device'] = str(temp_params['dataset_params']['device'])
    temp_params['dataset_params']['dtype']  = str(temp_params['dataset_params']['dtype'])
    temp_params['net_params']['device']     = str(temp_params['net_params']['device'])
    temp_params['net_params']['dtype']      = str(temp_params['net_params']['dtype'])
    with open(os.path.join(folder, "params.json"), 'w') as cf:
        json.dump(temp_params, cf, indent=4)
        cf.truncate()
    del temp_params

    reaching        = ReachingDataset(params['dataset_params'])
    in_features     = reaching.input_size[2]
    rnn_hidden_size = params['rnn_hidden_size']
    net_params      = params['net_params']
    net_params['args'] = [in_features, rnn_hidden_size]

    trainer = CCNetPreTrainer(net_params, model_class=params['model_class'],
                           opt=torch.optim.Adam, criterion=torch.nn.MSELoss)
    experiment.save("ReachingDataset", reaching, is_parameter=True)

    assert 'pretrain_settings' in params, \
        "Either pretrain_params or train_params must be provided"

    printfunc("\nRunning model pretrain.\n", ['red', 'bold'])
    params['pretrain_settings']['logfile'] = logfile
    params['pretrain_settings']['text']    = name + ' | Pretraining'

    if params['pretrain_settings']['tensorboard_params'] != {}:
        tb_log_dir = os.path.join(
            params['pretrain_settings']['tensorboard_params']['log_dir'], name + '_pretrain')
        params['pretrain_settings']['tensorboard_params']['log_dir'] = tb_log_dir
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)

    experiment.save("pretrain_settings", params['pretrain_settings'], is_parameter=True)

    outputs, performance = trainer.train(
        reaching, pretrain=True, **params['pretrain_settings'], plot=False, force=force)

    save_var(outputs['cortex_sd'], os.path.join(folder, 'cortex_state_dict.pkl'))
    experiment.save("pretrain_outputs", outputs)
    save_var(outputs['metrics'], os.path.join(folder, 'metrics.pkl'))
    experiment.update_performance("pretrain_performance", performance)
    printfunc("Pretraining completed and data correctly saved.")
    del outputs, performance



# ---------------------------------------------------------------------------
# Per-config runner — replicates the per-subprocess behavior of
# training_template.py: seed reset, device setup, preparing_params, loop
# ---------------------------------------------------------------------------

def run_config(config, config_index, output_folder, force):
    # Prefix name with index, same as --id arg in training_template.py
    config = dict(config)
    config['name'] = '.'.join([str(config_index), config['name']])
    name = config['name']

    # Seed — set at the same execution point as in training_template.py module level
    seed = config['seed'] if config['seed'] else 124
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = config['use_cuda']
    if use_cuda:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            device = torch.device('cuda')
            print('Using GPU')
        else:
            device = torch.device('cpu')
            print('Using CPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    dtype      = 'float32'
    torchdtype = getattr(torch, dtype)

    WORK_DIR = config['work_dir'] if config['work_dir'] else output_folder

    config_file_keys = ['name', 'seed', 'use_cuda', 'rnn_hidden_size', 'model_class']
    assert all(var in config for var in config_file_keys), \
        f"The JSON file must provide {config_file_keys}."
    assert config['name'], "Name must be provided in the JSON file"

    params, default_params_used = preparing_params(config, device, torchdtype)
    if default_params_used:
        print('Some parameters were not present in the config file. Using default values for: '
              + str(default_params_used))

    iterations = config.get('iterations', 1)
    iterations = iterations if iterations > 0 else 1

    if iterations == 1:
        training_method(name, WORK_DIR, params, device, force)
    else:
        folder, name = generate_folder2(WORK_DIR, name, True)
        with open(os.path.join(folder, "params.json"), 'w') as cf:
            temp_params = deepcopy(params)
            temp_params['dataset_params']['device'] = str(device)
            temp_params['dataset_params']['dtype']  = str(torchdtype)
            temp_params['net_params']['device']     = str(device)
            temp_params['net_params']['dtype']      = str(torchdtype)
            json.dump(temp_params, cf, indent=4)
            del temp_params
            cf.truncate()
        for j in range(iterations):
            training_method(name, folder, params, device, force, iteration=j)

def config_is_complete(config_folder, iterations):
    """Return True only if every iteration has a cortex_state_dict.pkl."""
    if iterations == 1:
        return os.path.exists(os.path.join(config_folder, 'cortex_state_dict.pkl'))
    return all(
        os.path.exists(os.path.join(config_folder, str(j), 'cortex_state_dict.pkl'))
        for j in range(iterations)
    )


if __name__ == '__main__':
    output_folder = os.path.join(WORKDIR, args.output)
    config_dir    = args.config_dir
    force         = args.force

    if os.path.exists(output_folder) and force:
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)


    configs_list_file    = os.path.join(config_dir, 'configs_list.json')
    template_config_file = os.path.join(config_dir, 'template_config.json')

    if not os.path.exists(configs_list_file):
        print(f"Error: configs_list.json not found in {config_dir}")
        sys.exit(1)
    if not os.path.exists(template_config_file):
        print(f"Error: template_config.json not found in {config_dir}")
        sys.exit(1)

    print("Preparing configuration files...")
    configs = prepare_configs(configs_list_file, template_config_file, output_folder, HOME)
    print(f"Configuration files prepared successfully.")
    print(f"Config folder: {os.path.join(output_folder, 'configs')}")
    print(f"Found {len(configs)} configuration(s) to run.")

    for i, config in enumerate(configs):
        config_name   = '.'.join([str(i), config['name']])
        config_folder = os.path.join(output_folder, config_name)
        iterations    = config.get('iterations', 1) or 1
        config_file   = os.path.join(output_folder, 'configs', f'config_{i}.json')

        print(f"\n{'='*56}")
        print(f" Running job {i} / {len(configs) - 1}")
        print(f" Config: {config_file}")
        print(f" Start:  {datetime.now()}")
        print(f"{'='*56}")

        # Skip fully completed configs
        if config_is_complete(config_folder, iterations):
            print(f"Job {i} already completed, skipping.")
            continue

        # Partial run: erase the config folder so seed replay is exact
        if os.path.exists(config_folder):
            print(f"Job {i} partially completed, erasing and restarting.")
            shutil.rmtree(config_folder)

        try:
            run_config(config, i, output_folder, force=True)
            print(f"Job {i} completed successfully.")
        except Exception as e:
            print(f"Warning: Job {i} exited with an error: {e}. Continuing to next job.")

    print(f"\nAll jobs completed. Results saved in: {output_folder}")