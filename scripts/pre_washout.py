# -*- coding: utf-8 -*-
"""
pre_washout.py

Perturbation + washout training for two networks from cx_pre pretrainings:
  - RNN        : cortex weights plastic, trained on perturbed task
  - fb.ae_fixed: all weights frozen, loss recorded only (no_grad forward)

Phase structure:
  1. Perturbation : 50  epochs, angle=40
  2. Washout      : 100 epochs, angle=0

Called by generate_data.py with --home, --workdir, --pretrainings, --output.
"""

import os
import sys
import json
import shutil
import traceback
import argparse
from datetime import datetime
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--home",         required=True)
parser.add_argument("--workdir",      required=True)
parser.add_argument("--pretrainings", required=True)
parser.add_argument("--output",       required=True)
parser.add_argument("-f", "--force",  action="store_true", default=False)
args = parser.parse_args()

import torch
import numpy as np

from ccdl.utils    import *
from ccdl.pre_trainer    import CCNetPreTrainer
from ccdl.reaching import ReachingDataset

# ---------------------------------------------------------------------------
# Hardcoded phases
# ---------------------------------------------------------------------------
TASKS = [
    {
        'name': 'perturbation',
        'perturbation_params': {'angle': 40, 'amplitude_factor': 1., 'translation': [0., 0.]},
        'epochs': 50,
    },
    {
        'name': 'washout',
        'perturbation_params': {'angle': 0,  'amplitude_factor': 1., 'translation': [0., 0.]},
        'epochs': 100,
    },
]

# ---------------------------------------------------------------------------
# Frozen-net trainer: no_grad forward, no backward/optimizer step
# ---------------------------------------------------------------------------
class CCNetFrozenTrainer(CCNetPreTrainer):
    def _training_loop(self, epoch, optimizers, scheduler, record_gradients, debug):
        self.model.train(**self.model_state, pretraining=self.pretrain)
        inputs, targets = self.dataset
        self.reset_loss()
        input_2 = self.correction_dataset if self.training_params.get('correction_fb') else None
        with torch.no_grad():
            outputs, activity, _ = self._model_call(inputs, input_2=input_2, 
                                                    input_2_type=self.input_2_type)
            loss = self.compute_loss(outputs, targets, activity)
        self.mse_losses.append(self.mse_loss)
        self.tot_losses.append(loss.item())
        return loss, outputs

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    'RNN': {
        'pretrain_subfolder': 'RNN',
        'trainer_class':      CCNetPreTrainer,
        'inter_specs_patch':  {},
        'model_state_patch': {
            'modules_plasticity': {
                'cortex_hw':       True,
                'cortex_hb':       True,
                'cortex_in':       False,
                'cortex_out':      False,
                'cereb_to_cortex': True,
                'cereb':           False,
            },
        },
        'train_params_patch': {'lr_scheduler': None, 'correction_fb': False},
    },
    'fb.ae_fixed.20': {
        'pretrain_subfolder': 'cortex',
        'trainer_class':      CCNetFrozenTrainer,
        'inter_specs_patch':  {'gain': 20},
        'model_state_patch':  {},       # model_state from pretraining used as-is
        'train_params_patch': {'lr_scheduler': None, 'correction_fb': True},
    },
    'fb.ae_fixed.15': {
        'pretrain_subfolder': 'cortex',
        'trainer_class':      CCNetFrozenTrainer,
        'inter_specs_patch':  {'gain': 15},
        'model_state_patch':  {},       # model_state from pretraining used as-is
        'train_params_patch': {'lr_scheduler': None, 'correction_fb': True},
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_json_serializable(obj):
    """Recursively convert non-JSON-serializable values (torch types) to strings."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

# ---------------------------------------------------------------------------
# run_model: owns the full iteration loop for one network config
# ---------------------------------------------------------------------------
def run_model(model_name, pre_folder, output_folder, device, torchdtype, force):
    cfg              = MODEL_CONFIGS[model_name]
    pretrain_folder  = os.path.join(pre_folder, cfg['pretrain_subfolder'])

    # Load params from this config's own pretraining folder
    with open(os.path.join(pretrain_folder, 'train_info.json')) as f:
        params = json.load(f)
        
    # Apply all config patches to pretrain_settings, then promote it to
    # params['settings'] and attach the task list. pretrain_settings is
    # removed to avoid the duplication noted in the original file.
    settings = params.pop('pretrain_settings')
    if cfg['model_state_patch']:
        settings['model_state'].update(cfg['model_state_patch'])
        if 'modules_plasticity' in cfg['model_state_patch']:
            settings['model_state']['modules_plasticity'] = \
                deepcopy(cfg['model_state_patch']['modules_plasticity'])
    settings['train_params'].update(cfg['train_params_patch'])
    params['settings'] = settings
    params['tasks']    = TASKS
    params['settings']['record_angle'] = True
    
    params['dataset_params']['device'] = device
    params['dataset_params']['dtype']  = torchdtype
    params['net_params']['device']     = device
    params['net_params']['dtype']      = torchdtype

    # count iterations
    n_iterations = len([f for f in os.listdir(pretrain_folder)
                        if os.path.isfile(os.path.join(pretrain_folder, f))
                        and os.path.splitext(f)[0].isdigit()])
    assert n_iterations > 0, f"No pretraining files found in {pretrain_folder}"

    print(f"\n[{model_name}] {n_iterations} iteration(s) to run.")
    
    model_dir = os.path.join(output_folder, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'params.json'), 'w') as f:
        json.dump(_to_json_serializable(params), f, indent=4)
        
    for iteration in range(n_iterations):
        print(f"\n{'='*56}\n [{model_name}] Iteration {iteration+1}/{n_iterations}"
              f" | {datetime.now()}\n{'='*56}")

        # Skip if already done
        data_dir    = os.path.join(model_dir, 'data')
        done_marker = os.path.join(data_dir, str(iteration))
        if os.path.exists(done_marker) and not force:
            pr_print(['bold', 'green'], f"  Already present, skipping.")
            continue

        try:
            _run_single_iteration(
                iteration, model_name, cfg,
                pretrain_folder, params,
                data_dir, force,
            )
        except Exception as e:
            print(f"  Warning: iteration {iteration} failed: {e}")
            traceback.print_exc()


def _run_single_iteration(iteration, model_name, cfg, pretrain_folder,
                           params, data_dir, force):

    experiment    = Experiment2(data_dir, str(iteration), force=force)
    saving_folder = experiment.folder
    logfile       = os.path.join(saving_folder, "experiment.log")

    def printfunc(text, formatting=None, end=None):
        printsave(text, logfile, True, formatting, end)

    printfunc(f"[{model_name}] Iteration {iteration}", ['bold', 'blue'])
    start = datetime.now()

    # Build net params and apply inter_specs patch
    net_params = deepcopy(params['net_params'])
    net_params['inter_specs'].update(cfg['inter_specs_patch'])

    reaching = ReachingDataset(params['dataset_params'])
    net_params['args'] = [reaching.input_size[2], params['rnn_hidden_size']]

    # Build trainer and load pretraining
    trainer = cfg['trainer_class'](net_params, model_class=params['model_class'],
                                   opt=torch.optim.Adam, criterion=torch.nn.MSELoss)
    trainer.logfile = logfile
    sd = load_var(os.path.join(pretrain_folder, str(iteration)))
    trainer.load_cortex(sd)
    printfunc(f"Pretraining loaded.")
    
    with torch.no_grad():
        trainer.model.train(**params['settings']['model_state'], pretraining=False)
        unpert_inputs, unpert_targets = reaching.generate_unperturbed_dataset(
            params['settings']['datatype']
        )
        _, input_2 = reaching.generate_unperturbed_dataset('angles')
        if params['settings']['train_params'].get('correction_fb'):
            unpert_outputs, _, _ = trainer._model_call(unpert_inputs, input_2=input_2, input_2_type='angles')
        else:
            unpert_outputs, _, _ = trainer._model_call(unpert_inputs)
        unpert_mse = trainer.criterion(unpert_outputs, unpert_targets).item()
    experiment.save("unperturbed_loss", unpert_mse)

    all_mse, all_loss, all_angles, all_takeoff_angles = [], [], [], []

    for phase in TASKS:
        reaching.apply_perturbation(phase['perturbation_params'])

        # build phase-specific settings: deepcopy so epoch mutation doesn't leak
        phase_settings = deepcopy(params['settings'])
        phase_settings['train_params']['epochs'] = phase['epochs']

        printfunc(f"  Phase: {phase['name']} ({phase['epochs']} epochs)", ['bold'])

        outputs, _ = trainer.train(
            reaching,
            pretrain=False,
            plot=False,
            force=True,
            outsave_step=0,
            save_w_steps=0,
            record_gradients=False,
            save_cortex=False,
            verbose=False,
            debug=False,
            tensorboard_params={},
            text=f"{model_name} | {phase['name']}",
            logfile=logfile,
            **{k: phase_settings[k]
               for k in ('train_params', 'model_state', 'datatype', 'print_steps', 'record_angle')},
            skip_init_loss=True
        )

        all_mse  += outputs['mse_losses']
        all_loss += outputs['tot_losses']
        all_angles += outputs.get('angular_error_rad', [])
        all_takeoff_angles += outputs.get('takeoff_angular_error_rad', [])
        del outputs

    
    # --- Save losses ---
    experiment.save("train_outputs", {'mse': all_mse, 'loss': all_loss, 'angles': all_angles, 'takeoff_angles': all_takeoff_angles})
    
    printfunc(f"Done. Total time: {datetime.now() - start}", ['bold', 'green'])


# ---------------------------------------------------------------------------
# Main: one call per model config
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(124)
    torch.manual_seed(124)

    device        = torch.device('cpu')
    torchdtype    = torch.float32
    output_folder = os.path.join(args.workdir, args.output)
    os.makedirs(output_folder, exist_ok=True)

    for model_name in MODEL_CONFIGS:
        run_model(model_name, args.pretrainings, output_folder,
                  device, torchdtype, args.force)
    
    # Copy this script to the main output folder for reproducibility
    shutil.copy(__file__, os.path.join(output_folder, 'script.py'))

    print(f"\nAll done. Results in: {output_folder}")