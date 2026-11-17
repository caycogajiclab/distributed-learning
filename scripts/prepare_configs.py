# -*- coding: utf-8 -*-
"""
Created on 01/02/2024
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu
"""
import itertools
import copy
from warnings import warn
import json
import argparse
import os

def generate_params_combinations(starting_dictionary, iter_params, verbose=False):
    """
    Generate distinct updated dictionaries for all combinations of values from grid_dictionary.

    Args:
        starting_dictionary (dict): The dictionary to be updated.
        grid_dictionary (dict): The dictionary containing keys and lists of values.

    Returns:
        list: A list of updated dictionaries with all combinations.
    """
    
    if isinstance(iter_params, dict):
        combinations = list(itertools.product(*iter_params.values()))
        combo_keys = [iter_params.keys() for i in range(len(combinations))]
    elif isinstance(iter_params, list):
        combinations = [list(iter_param_i.values()) for iter_param_i in iter_params]
        combo_keys = [iter_param_i.keys() for iter_param_i in iter_params]
    else:
        raise IOError('Please provide either grid_dictionary or train_list')
    updated_dicts = []

    for k, (combo, keys) in enumerate(zip(combinations, combo_keys)):
        updated_dict = copy.deepcopy(starting_dictionary)
        for i, key in enumerate(keys):
            keys = key.split('.')
            current_dict = updated_dict
            for subkey in keys[:-1]:
                if subkey not in current_dict:
                    current_dict[subkey] = {}
                current_dict = current_dict[subkey]
            if keys[-1] not in current_dict.keys() and verbose:
                warn(f'ATTENTION: in config {k}, parameter {key} is not present in the template_config file. '
                     'This might be due to a typo in the config_list, please check it')
            current_dict[keys[-1]] = combo[i]

        updated_dicts.append(updated_dict)

    return updated_dicts

def update_configs(params_conf_list_file, template_config_file, output_dir, 
                   home, cx_pre=None, cb_pre=None, rnn_pre=None, verbose=False):


    with open(params_conf_list_file, 'r') as params_file:
        params_conf_list = json.load(params_file)
    template_params = json.load(open(template_config_file))
    template_params['home'] = home
    template_params['work_dir'] = output_dir
    params_list = generate_params_combinations(template_params, params_conf_list, verbose=verbose)

    configs_dir = os.path.join(output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    specs = {}
    for index, (full_conf,new_conf) in enumerate(zip(params_list,params_conf_list)):
        if full_conf.get('rnn_only', False):
            assert isinstance(rnn_pre, str), 'Please provide a path to the cortex pretraining folder'
            assert os.path.isdir(rnn_pre), 'Cortex pretraining folder does not exist'
            full_conf['cx_pretrain_path'] = rnn_pre
            if index==0:
                print(f"Assigning cortex pretraining path to config file: cx_pretrain_path = '{full_conf['cx_pretrain_path']}'")
        else:
            if cx_pre is not None:
                assert isinstance(cx_pre, str), 'Please provide a path to the cortex pretraining folder'
                assert os.path.isdir(cx_pre), 'Cortex pretraining folder does not exist'
                full_conf['cx_pretrain_path'] = cx_pre
                if index==0:
                    print(f"Assigning cortex pretraining path to config file: cx_pretrain_path = '{full_conf['cx_pretrain_path']}'")
            if cb_pre is not None:
                assert isinstance(cb_pre, str), 'Please provide a path to the cerebellum pretraining folder'
                assert os.path.isdir(cb_pre), 'Cerebellum pretraining folder does not exist'
                full_conf['cb_pretrain_path'] = cb_pre
                if index==0:
                    print(f"Assigning cerebellum pretraining path to config file: cb_pretrain_path = '{full_conf['cb_pretrain_path']}'")
        config_file_path = f'{configs_dir}/config_{index}.json'
        with open(config_file_path, 'w') as config_file:
            json.dump(full_conf, config_file, indent=4)
            config_file.truncate()
        # saving only the differences
        specs[f'config_{index}.json'] = new_conf
    json.dump(specs, open(f'{output_dir}/specs.json', 'w'), indent=4)
    

if __name__ == "__main__":
    print(f"Preparing configs json. Current folder: {os.getcwd()}")
    parser = argparse.ArgumentParser(description='Update configuration files based on parameters')
    parser.add_argument('params_conf_list_file', help='Path to the parameters configuration list file')
    parser.add_argument('template_config_file', help='Path to the template configuration file')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('home', help='Path to the home directory')
    parser.add_argument('--cx-pre', default=None, help='Path to the cortex pretraining folder')
    parser.add_argument('--cb-pre', default=None, help='Path to the cerebellum pretraining folder')
    parser.add_argument('--rnn-pre', default=None, help='Path to the RNN pretraining folder')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if not isinstance(args.cx_pre, type(None)):
        print("cx_pre provided: ",args.cx_pre)
    if not isinstance(args.cb_pre, type(None)):
        print("cb_pre provided: ",args.cb_pre)
    update_configs(args.params_conf_list_file, args.template_config_file, 
                   args.output_dir, args.home, cx_pre=args.cx_pre, 
                   cb_pre=args.cb_pre, rnn_pre=args.rnn_pre, verbose=args.verbose)
