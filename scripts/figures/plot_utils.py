"""Shared plotting utilities for the figure-generation scripts.

Axis/legend styling, per-figure source-data save/load helpers, and the
preprocessed-metrics loaders used across ``scripts/figures/``.
"""
import os, pickle
import json
import shutil
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.cm import rainbow

from ccdl.utils import load_var, save_var, Experiment2
from ccdl.reaching import ReachingDataset
from ccdl.ccnet import CerebelloCorticalNet
from ccdl.paths import DATA_DIR


def load_or_build_plot_data(name, plots_data_folder, update_plots=False, builder=None, save_data=True):
    os.makedirs(plots_data_folder, exist_ok=True)
    data_path = os.path.join(plots_data_folder, f"{name}.pkl")

    if not update_plots and os.path.exists(data_path):
        return load_var(data_path)

    if builder is None:
        raise ValueError("builder must be provided when plot data is not cached.")

    plot_data = builder()

    if save_data and plot_data is not None:
        save_var(plot_data, data_path)

    return plot_data


# Opt-in cascade: when a spec is missing, fall back to rebuilding just that one
# (prepare() -> cached metrics -> raw data) instead of raising. Loud, never
# silent — the original bug was a *silent* fallback. Off by default so a plain
# from-source run stays strict and never touches raw.
SPEC_FALLBACK = os.environ.get('SPEC_FALLBACK', '').lower() in ('1', 'true', 'yes', 'on')
# T2 (rebuild source_data without raw): reuse an existing spec when a
# model/trajectory figure's prepare() can't reach its raw inputs. Gated so T1
# (raw present) stays strict and never silently reuses a stale spec.
REUSE_SPEC_ON_MISSING_RAW = os.environ.get('REUSE_SPEC_ON_MISSING_RAW', '').lower() in ('1', 'true', 'yes', 'on')


def load_or_build_spec(name, plots_data_folder, prepare, rebuild=False):
    spec_path = os.path.join(plots_data_folder, f"{name}_spec.pkl")

    if not rebuild:
        if os.path.exists(spec_path):
            with open(spec_path, 'rb') as f:
                return pickle.load(f)
        # Spec is missing.
        if not SPEC_FALLBACK:
            raise FileNotFoundError(
                f"{name}_spec.pkl not found in source_data and rebuild=False "
                f"(looked in {plots_data_folder}). Set REBUILD_SOURCE_DATA=True to "
                f"rebuild all specs, or SPEC_FALLBACK=1 to rebuild only the missing "
                f"ones from cached metrics / raw data."
            )
        warnings.warn(
            f"{name}_spec.pkl missing; SPEC_FALLBACK set -> rebuilding it from "
            f"cached metrics / raw data."
        )
        return prepare()

    if REUSE_SPEC_ON_MISSING_RAW and os.path.exists(spec_path):
        try:
            return prepare()
        except FileNotFoundError as exc:
            warnings.warn(
                f"{name}: rebuild source unavailable ({exc}); reusing existing "
                f"{name}_spec.pkl (REUSE_SPEC_ON_MISSING_RAW set)."
            )
            with open(spec_path, 'rb') as f:
                return pickle.load(f)

    return prepare()


# =============================================================================
# SMOOTHING HELPERS
# =============================================================================

def moving_median(data, window_size):
    if window_size == 0:
        return data

    if len(data.shape) == 2:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data = data.T
        data = data.rolling(window=window_size, min_periods=1).median()
        data = data.values.T
    elif len(data.shape) == 1:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        data = data.rolling(window=window_size, min_periods=1).median()
        data = data.values
    else:
        raise ValueError("data must be 1D or 2D")

    return data


def moving_average(data, window_size):
    if window_size == 0:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def apply_gaussian_smoothing(data, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d

    if sigma == 0:
        return data

    if len(data.shape) == 2:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        smoothed_data = np.apply_along_axis(
            gaussian_filter1d,
            axis=1,
            arr=data,
            sigma=sigma,
            radius=max(4, int(sigma * 5)),
        )
    elif len(data.shape) == 1:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        smoothed_data = gaussian_filter1d(
            data,
            sigma=sigma,
            radius=max(4, int(sigma * 2)),
        )
    else:
        raise ValueError("data must be 1D or 2D")

    return smoothed_data


# =============================================================================
# PARAMS / DATASET HELPERS
# =============================================================================

def parse_model_name(name):
    parts = name.split('.')
    return '.'.join(parts[1:]) if parts and parts[0].isdigit() else name


def load_train_params(path, single_training=False):
    if single_training:
        try:
            params_file = os.path.join(path, "params.json")
            with open(params_file) as f:
                params = json.load(f)
        except FileNotFoundError:
            params_file = os.path.join(path, "0", "params.json")
            with open(params_file) as f:
                params = json.load(f)
        return params

    dicts = []
    for filename in ["template_config.json", "configs_list.json"]:
        file = os.path.join(path, filename)
        with open(file, "r") as f:
            dicts.append(json.load(f))
    return tuple(dicts)

def get_dataset(params, pert_params=None, datatype=None):
    dataset_params = params["dataset_params"]
    dataset_params["device"] = "cpu"

    if not datatype:
        try:
            datatype = params["train_settings"]["datatype"]
        except Exception:
            try:
                datatype = params["pretrain_settings"]["datatype"]
            except Exception:
                datatype = params["settings"]["datatype"]

    reaching = ReachingDataset(dataset_params)

    if pert_params:
        reaching.apply_perturbation(pert_params)
        return reaching.generate_perturbed_dataset(datatype)

    return reaching.generate_unperturbed_dataset(datatype)


def get_rescaling(params, datatype=None, mode="normal", verbose=False):
    if not datatype:
        try:
            datatype = params["train_settings"]["datatype"]
            if verbose:
                print(f"Using training datatype {datatype}")
        except Exception:
            try:
                datatype = params["pretrainsettings"]["datatype"]
            except Exception:
                datatype = params["settings"]["datatype"]
            if verbose:
                print(f"Using default datatype {datatype}")

    dataset = get_dataset(params, datatype=datatype)
    target = dataset[1]

    if mode.lower() == "minmax":
        return target.max().item() - target.min().item()
    if mode.lower() == "normal":
        return target.std().item()

    raise ValueError("Please choose a mode between 'minmax' and 'normal' for rescaling.")


def fb_nuc_norm(model):
    return torch.linalg.matrix_norm(model.cortex.rnn.input2h_2.weight, ord='nuc').item()


# Function to load data
def load_data(data_folder, selected=None):
    full_data = load_var(os.path.join(data_folder, 'data.pkl'))
    data = {k: v for k, v in full_data.items() if k in selected} if selected else {k: v for k, v in full_data.items()}
    epochs = full_data.pop('epochs')
    angles = full_data.pop('angles')
    return data, epochs, angles


def load_continual_metrics(
    data_folder,
    update,
    pretrainings_folder,
    metrics=['mse', 'nocb_error', 'cortex_error', 'changes', 'take_off_angle', 'endpoint_distance'],
    normalize_changes=False,
    save_name='data.pkl',
):
    epoch_flag = True
    data_flag = False
    std_flag = True

    skip_folders = {
        'specs.json', 'config.json', 'completed_jobs', 'jobs_ids.txt', 'scripts',
        'submitted_jobs', 'snapshot', 'configs', 'trainings_metrics',
        'trainings_metrics_continual'
    }

    data_path = os.path.join(DATA_DIR, 'metrics', os.path.basename(data_folder), save_name)
    if os.path.exists(data_path):
        if update:
            os.remove(data_path)
        else:
            data_flag = True

    if data_flag:
        data = load_var(data_path)
        print("Metrics data found and loaded.")
        return data

    data = {}

    sim_folders = [
        file for file in sorted(os.listdir(data_folder))
        if file not in skip_folders
        and os.path.isdir(os.path.join(data_folder, file))
    ]

    for file in sim_folders:
        exp_folder = os.path.join(data_folder, file)
        iterations = {}

        for f in sorted(os.listdir(exp_folder), key=lambda x: int(x) if x.isdigit() else float('inf')):
            if not f.isdigit():
                continue

            print(f'Loading file {file}/{f}', end='\r')
            exp = Experiment2(exp_folder, f, load=True)
            params = load_train_params(exp_folder, single_training=True)
            if std_flag:
                dataset_params = params['dataset_params']
                dataset_params['device'] = 'cpu'
                reaching = ReachingDataset(dataset_params)
                std = reaching.generate_unperturbed_dataset('cart')[1].std().item()
                std_flag = False

            continual_task = params['continual_task']

            if normalize_changes:
                pretrain_sd = load_var(os.path.join(pretrainings_folder, f))
                W_0 = pretrain_sd['rnn.h2h.weight'].abs().mean().item()
            else:
                W_0 = 1.0

            if epoch_flag:
                epochs = []

            train_outputs = exp.train_outputs
            phase_names = list(train_outputs.keys())

            extended_metrics = None
            for j, phase_name in enumerate(phase_names):
                train_output_j = train_outputs[phase_name]

                if extended_metrics is None:
                    extended_metrics = {
                        key: [] for key in train_output_j['metrics'].keys() if key in metrics
                    }
                    if 'changes' in metrics and 'changes' not in extended_metrics:
                        extended_metrics['changes'] = []

                if 'changes' in metrics:
                    changes = np.array(train_output_j['changes']['cortex.rnn.h2h.weight']) / W_0
                    extended_metrics['changes'].append(changes)

                for k, v in train_output_j['metrics'].items():
                    if k not in metrics:
                        continue

                    if k in ['mse', 'nocb_error', 'cortex_error']:
                        v = [value / std for value in v]

                    new_v = v if j == 0 else v[1:]
                    extended_metrics[k].append(new_v)

                if epoch_flag:
                    epochs.append(len(new_v))

            extended_metrics = {
                k: np.concatenate(extended_metrics[k]) for k in extended_metrics.keys()
            }

            if epoch_flag:
                angles = [val['perturbation_params']['angle'] for val in continual_task]

            iterations[f] = deepcopy(extended_metrics)
            del exp, extended_metrics
            epoch_flag = False
        key = file.split('.', 1)[1]
        data[key] = iterations
        print(f'Loading file {file} done.')

    print('Saving data.pkl...', end='\r')
    data['epochs'] = epochs
    data['angles'] = angles
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    save_var(data, data_path)
    print('Saving data.pkl... Done.')
    return data


def plot_reachings_evolution(folder, name, targets_lw=5, trial_lw=2, targets_alpha=0.15, trial_alpha=0.7, plot_specs=None, lims=None, dpi=100,
                 selected_task=2, plot_axes=True, fontsize=16, cmap=rainbow, figsize=(26, 7), gain=None, empty_endpoints=True,markersize=360, plot_fb=False,
                 legend_position=(1.05, 1.0), title='', colorbar='last', return_fig=False, contrast=False, excluded=[], dots=None, debug=False, plot_unpert=True, marker='s', edge=3.2,alpha_targets=1, capture=None, replay=None):

        if replay is not None:
            assert not dots, 'dots overlay not supported in replay mode (not captured)'
            assert not gain, 'gain/effective_gain not supported in replay mode (requires the model)'
            weights_evolution_flag = True

            lims = [[-0.12,0.175],[0.2,0.48]] if lims is None else lims
            plot_specs = {'epochs_list': [0,1,2,5,10,20,50], 'steps': 5} if not plot_specs else plot_specs
            epochs_list = np.sort(plot_specs['epochs_list'])
            steps = plot_specs['steps']

            n_tasks = replay[0]['n_tasks']
            tasks = np.arange(n_tasks)
            entries_by_panel = {}
            for entry in replay:
                entries_by_panel.setdefault(entry['panel_index'], {})[entry['task']] = entry

            colors = cmap(np.linspace(0, 1, n_tasks)) if cmap else ['k' for i in range(n_tasks)]
            if empty_endpoints:
                markeredgecolors = colors
                markerfacecolors = ['none' for i in range(n_tasks)]
                s_unpert = 1.2*markersize
                s_pert = markersize
            else:
                markeredgecolors = ['k' for i in range(n_tasks)]
                markerfacecolors = colors
                s_unpert = 1.2*markersize
                s_pert = markersize

            titles = [
                'Full network',
                'Removing Adapter readout',
                'No-adapter'
            ]
            if plot_fb:
                fig, axs = plt.subplots(1,3, figsize=figsize, dpi=dpi)
            else:
                fig, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi)
                del titles[1]
            plt.gca().set_aspect('equal', adjustable='box')
            fig.suptitle(title, fontsize=1.2*fontsize)

            for i in range(len(axs)):

                n = len(epochs_list)
                alphas = {epoch: alpha for (epoch, alpha) in zip(epochs_list,np.linspace(*plot_specs['color_range'],n))}

                panel_entries = entries_by_panel.get(i, {})

                for task in tasks:
                    if selected_task != 'all' and task !=selected_task:
                        continue
                    entry = panel_entries.get(task)
                    if entry is None:
                        continue
                    comps = {c['component']: c for c in entry['components']}
                    if plot_unpert and 'unperturbed_target' in comps:
                        c = comps['unperturbed_target']
                        axs[i].scatter(c['x'], c['y'], alpha = 0.6, edgecolor=markeredgecolors[task],
                                marker="^", s=s_unpert,color=colors[task], zorder=2, label=f'Initial targets'
                                    if task==1 else None, facecolor='none',linewidths=edge)

                    if 'perturbed_target' in comps:
                        c = comps['perturbed_target']
                        axs[i].scatter(c['x'], c['y'], color = colors[task], alpha = alpha_targets,
                                            marker=marker, s=s_pert, edgecolor=markeredgecolors[task],
                                            facecolor=markerfacecolors[task], linewidths=edge, zorder=2,
                                            label=f'Perturbed targets' if task==1 else None)
                if i==(len(axs)-1):
                    axs[i].legend(fontsize=fontsize, handlelength=4, bbox_to_anchor=legend_position)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['bottom'].set_linewidth(3)
                axs[i].spines['left'].set_linewidth(3)
                axs[i].spines['bottom'].set_visible(plot_axes)
                axs[i].spines['left'].set_visible(plot_axes)
                axs[i].set_aspect('equal')
                if plot_axes:
                    axs[i].set_xticks([-0.1,-0.05,0.0,0.05,0.1])
                    axs[i].set_yticks([0.25,0.3,0.35,0.4])
                    axs[i].tick_params(axis='both', which='major', labelsize=fontsize, width=4,length=8)
                else:
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                axs[i].set_xlim(*lims[0])
                axs[i].set_ylim(*lims[1])
                axs[i].set_title(titles[i],fontsize=0.8*fontsize)

                comps_by_task = {}
                for task in tasks:
                    if selected_task != 'all' and selected_task is not None:
                        if task!=selected_task:
                            continue
                    if task in excluded:
                        continue
                    entry = panel_entries.get(task)
                    if entry is None:
                        continue
                    comps_by_task[task] = {c['component']: c for c in entry['components']}

                for j in epochs_list:
                    for task in tasks:
                        comps = comps_by_task.get(task)
                        if comps is None:
                            continue
                        c = comps.get(f'epoch{j}')
                        if c is None:
                            continue
                        t_lw = trial_lw
                        if j == epochs_list[-1]:
                            t_lw = trial_lw
                        if j == epochs_list[0] and epochs_list[0]==0:
                            t_lw = 1.4*trial_lw
                        zorder=3
                        if contrast and j == epochs_list[-1]:
                            zorder=4
                            axs[i].plot(c['x'], c['y'],c='k',lw=contrast*t_lw, alpha=1, solid_capstyle='round', zorder=zorder)
                        axs[i].plot(c['x'], c['y'],c=colors[task],lw=t_lw, alpha=alphas[j], solid_capstyle='round', label="Output trajectories" if task==1 else None, zorder=zorder)

                if colorbar=='each':
                    custom_cmap = ListedColormap([[0, 0, 0, alpha] for alpha in alphas.values()])
                    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(-0.5, n-0.5))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=axs[i])
                    cbar.set_ticks(np.arange(0,n))
                    cbar.set_ticklabels((np.array([steps*epoch for epoch in epochs_list])).astype(int).astype(str),fontsize=fontsize)
                    cbar.set_label('Epoch', fontsize=fontsize)

            if colorbar !='each' and weights_evolution_flag:
                custom_cmap = ListedColormap([0, 0, 0, alpha] for alpha in alphas)
                sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(-0.5, n-0.5))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=axs[-1])
                cbar.set_ticks(np.arange(0,n))
                cbar.set_ticklabels(([steps*epoch for epoch in epochs_list]).astype(int).astype(str),fontsize=0.8*fontsize)
                cbar.set_label('Epoch', fontsize=fontsize)

            plt.show(block=not return_fig)
            outputs = []
            if return_fig:
                outputs.append(fig)
            return tuple(outputs)

        exp = Experiment2(folder, name, load=True)
        weights_evolution_flag = True
        try:
            weights = exp.train_outputs['parameters']
        except:
            weights_evolution_flag = False
            weights = exp.train_outputs['model_sd']
            warnings.warn('Weights are loaded from model_sd.pkl. No evolution to plot, just final performance')
        n_params = len(weights)
        file = os.path.join(folder, 'params.json')
        with open(file, 'r') as f:
            params = json.load(f)

        # making sure things are on cpu
        params['dataset_params']['device'] = 'cpu'
        params['net_params']['device'] = 'cpu'

        epochs = params["train_params"]["epochs"]
        save_w_steps = params['settings']["save_w_steps"]
        if debug:
            print('save_w_steps', save_w_steps)
            print('epochs', epochs)

        dataset_params = params['dataset_params']
        perturbation_params = params['perturbation_params']
        reaching = ReachingDataset(dataset_params)
        reaching.apply_perturbation(perturbation_params)
        inputs, un_targets = reaching.generate_unperturbed_dataset('cart')
        _, targets = reaching.generate_perturbed_dataset('cart')


        in_features = reaching.input_size[2]
        n_tasks = reaching.input_size[1]
        rnn_hidden_size = params['rnn_hidden_size']
        net_params = params['net_params']
        perturbation_params = params['perturbation_params']
        args=[in_features, rnn_hidden_size]
        model = CerebelloCorticalNet(*args, **net_params)
        model.load_state_dict(weights[-1])
        if gain:
            model.gain = gain
            effective_gain = gain*fb_nuc_norm(model)

        # model states
        model_state = params['model_state']
        no_cb_state = deepcopy(model_state)
        no_cb_state['cereb_readout'] = False
        no_cb_state['cereb_feedback'] = False
        if plot_fb:
            no_readout_state = deepcopy(model_state)
            no_readout_state['cereb_readout'] = False
            states = [model_state,no_readout_state,no_cb_state]
        else:
            states = [model_state,no_cb_state]

        if debug:
            for state in states:
                print(state)


        # plot settings

        tasks = np.arange(inputs.shape[1])
        lims = [[-0.12,0.175],[0.2,0.48]] if lims is None else lims
        s_task = 2 if not selected_task else selected_task
        plot_specs = {'epochs_list': [0,1,2,5,10,20,50], 'steps': 5} if not plot_specs else plot_specs
        epochs_list = np.sort(plot_specs['epochs_list'])
        steps = plot_specs['steps']

        selected_weights = {epoch: weights[epoch] for epoch in epochs_list}

        colors = cmap(np.linspace(0, 1, inputs.shape[1])) if cmap else ['k' for i in range(n_tasks)]
        if empty_endpoints:
            markeredgecolors = colors
            markerfacecolors = ['none' for i in range(n_tasks)]
            s_unpert = 1.2*markersize
            s_pert = markersize
        else:
            markeredgecolors = ['k' for i in range(n_tasks)]
            markerfacecolors = colors
            s_unpert = 1.2*markersize
            s_pert = markersize

        titles = [
            'Full network',
            'Removing Adapter readout',
            'No-adapter'
        ]
        if plot_fb:
            fig, axs = plt.subplots(1,3, figsize=figsize, dpi=dpi)
        else:
            fig, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi)
            del titles[1]
        plt.gca().set_aspect('equal', adjustable='box')
        fig.suptitle(title, fontsize=1.2*fontsize)

        for i in range(len(axs)):

            n = len(epochs_list)
            alphas = {epoch: alpha for (epoch, alpha) in zip(epochs_list,np.linspace(*plot_specs['color_range'],n))}

            for task in tasks:
                if selected_task != 'all' and task !=selected_task:
                    continue
                t_alpha = targets_alpha
                t_lw = targets_lw
                if plot_unpert:
                    axs[i].scatter(un_targets[-1, task, 0], un_targets[-1, task, 1], alpha = 0.6, edgecolor=markeredgecolors[task],
                            marker="^", s=s_unpert,color=colors[task], zorder=2, label=f'Initial targets'
                                if task==1 else None, facecolor='none',linewidths=edge)


                # plotting perturbed targets
                axs[i].scatter(targets[-1, task, 0], targets[-1, task, 1], color = colors[task], alpha = alpha_targets,
                                    marker=marker, s=s_pert, edgecolor=markeredgecolors[task],
                                    facecolor=markerfacecolors[task], linewidths=edge, zorder=2,
                                    label=f'Perturbed targets' if task==1 else None)
            if i==(len(axs)-1):
                axs[i].legend(fontsize=fontsize, handlelength=4, bbox_to_anchor=legend_position)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_linewidth(3)
            axs[i].spines['left'].set_linewidth(3)
            axs[i].spines['bottom'].set_visible(plot_axes)
            axs[i].spines['left'].set_visible(plot_axes)
            axs[i].set_aspect('equal')
            if plot_axes:
                axs[i].set_xticks([-0.1,-0.05,0.0,0.05,0.1])
                axs[i].set_yticks([0.25,0.3,0.35,0.4])
                axs[i].tick_params(axis='both', which='major', labelsize=fontsize, width=4,length=8)
            else:
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            axs[i].set_xlim(*lims[0])
            axs[i].set_ylim(*lims[1])
            axs[i].set_title(titles[i],fontsize=0.8*fontsize)

            if capture is not None:
                captured_components = {}

            for j, net in selected_weights.items():
                model.load_state_dict(net)
                model.eval(**states[i])
                out = model(inputs)
                t_lw = trial_lw
                if j == epochs_list[-1]:
                    t_lw = trial_lw
                if j == epochs_list[0] and epochs_list[0]==0:
                    t_lw = 1.4*trial_lw
                for task in tasks:
                    if selected_task != 'all' and selected_task is not None:
                        if task!=selected_task:
                            continue
                    if task in excluded:
                        continue
                    zorder=3
                    if contrast and j == epochs_list[-1]:
                        zorder=4
                        axs[i].plot(out[:,task,0], out[:,task,1],c='k',lw=contrast*t_lw, alpha=1, solid_capstyle='round', zorder=zorder)
                    axs[i].plot(out[:,task,0], out[:,task,1],c=colors[task],lw=t_lw, alpha=alphas[j], solid_capstyle='round', label="Output trajectories" if task==1 else None, zorder=zorder)
                    if capture is not None:
                        captured_components.setdefault(task, []).append({'component': f'epoch{j}', 'x': out[:, task, 0].detach().tolist(), 'y': out[:, task, 1].detach().tolist()})
                    if dots:
                        axs[i].scatter(out[::dots,task,0], out[::dots,task,1],c=colors[task],s=t_lw*3, alpha=t_alpha, zorder=zorder)
                del out

            if capture is not None:
                for task, components in captured_components.items():
                    if plot_unpert:
                        components.append({'component': 'unperturbed_target', 'x': [float(un_targets[-1, task, 0])], 'y': [float(un_targets[-1, task, 1])]})
                    components.append({'component': 'perturbed_target', 'x': [float(targets[-1, task, 0])], 'y': [float(targets[-1, task, 1])]})
                    capture.append({
                        'name': f'{titles[i]}_task{task}',
                        'panel_index': i,
                        'task': int(task),
                        'n_tasks': int(inputs.shape[1]),
                        'components': components,
                    })

            if colorbar=='each':
                # custom_cmap = ListedColormap([*colors[1][:3], alpha] for alpha in alphas)
                custom_cmap = ListedColormap([[0, 0, 0, alpha] for alpha in alphas.values()])
                sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(-0.5, n-0.5))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=axs[i])
                # vector = np.array(list(x_cmap[::30])+[x_cmap[-1]])
                # vector = (vector - vector.min())/(vector.max()-vector.min())
                cbar.set_ticks(np.arange(0,n))
                cbar.set_ticklabels((np.array([steps*epoch for epoch in epochs_list])).astype(int).astype(str),fontsize=fontsize)
                cbar.set_label('Epoch', fontsize=fontsize)



        if colorbar !='each' and weights_evolution_flag:
            # custom_cmap = ListedColormap([*colors[1][:3], alpha] for alpha in alphas)
            custom_cmap = ListedColormap([0, 0, 0, alpha] for alpha in alphas)
            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(-0.5, n-0.5))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[-1])
            cbar.set_ticks(np.arange(0,n))
            cbar.set_ticklabels(([steps*epoch for epoch in epochs_list]).astype(int).astype(str),fontsize=0.8*fontsize)
            cbar.set_label('Epoch', fontsize=fontsize)

        plt.show(block=not return_fig)
        del exp
        outputs = []
        if return_fig:
            outputs.append(fig)
        if gain:
            outputs.append(effective_gain)
        return tuple(outputs)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _append_scalar_series_rows(rows, common, series, component='mean'):
    x_values = series.get('x', series.get('xmean'))
    y_values = series.get('y', series.get('ymean'))
    x_fill = series.get('xfill')
    y_min = series.get('ymin')
    y_max = series.get('ymax')
    raw_y = series.get('raw_y', None)

    if x_values is not None and y_values is not None:
        for point_index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            rows.append({
                **common,
                'component': component,
                'pointindex': point_index,
                'x': float(x_value),
                'y': float(y_value),
            })

    if x_fill is not None and y_min is not None and y_max is not None:
        for point_index, (x_value, y_low, y_high) in enumerate(zip(x_fill, y_min, y_max)):
            rows.append({
                **common,
                'component': f'{component}_band',
                'pointindex': point_index,
                'x': float(x_value),
                'ylow': float(y_low),
                'yhigh': float(y_high),
            })

    if raw_y is not None and x_values is not None:
        raw_y = np.asarray(raw_y)

        if raw_y.ndim == 1:
            raw_y = raw_y[None, :]

        for replicate_index, replicate_values in enumerate(raw_y):
            for point_index, (x_value, y_value) in enumerate(zip(x_values, replicate_values)):
                rows.append({
                    **common,
                    'component': f'{component}_replicate',
                    'replicate': int(replicate_index),
                    'pointindex': point_index,
                    'x': float(x_value),
                    'y': float(y_value),
                })


def _plotspec_to_rows(data, name):
    rows = []
    meta = data.get('meta', {})

    for panel_index, panel in enumerate(data['panels']):
        panel_id = panel.get('metric', f'panel{panel_index}')

        for series_index, series in enumerate(panel.get('series', [])):
            common = {
                'figurename': name,
                'figuretype': data.get('figuretype', ''),
                'panelid': panel_id,
                'seriesid': series.get('name', f'series{series_index}'),
                'label': series.get('label', ''),
                'color': series.get('color', ''),
                'metric': series.get('metric', panel.get('metric', '')),
                'nruns': series.get('nruns', None),
                **{f'meta_{key}': value for key, value in meta.items()},
            }

            components = series.get('components', None)

            if components is None:
                _append_scalar_series_rows(
                    rows=rows,
                    common=common,
                    series=series,
                    component='mean',
                )
            else:
                for component in components:
                    component_name = component.get('component', 'component')
                    _append_scalar_series_rows(
                        rows=rows,
                        common=common,
                        series=component,
                        component=component_name,
                    )

    return rows


def save_plot(fig, name, plotsfolder, savepdf=False, savetiff=False, savedata=False,
              data=None, plotsdatafolder=None, dpi=400, transparent=True,
              bbox_inches='tight', csv_data=None):

    os.makedirs(plotsfolder, exist_ok=True)

    svg_folder = os.path.join(plotsfolder, 'svg')
    os.makedirs(svg_folder, exist_ok=True)
    fig.savefig(
        os.path.join(svg_folder, f'{name}.svg'),
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
    )

    if savepdf:
        pdf_folder = os.path.join(plotsfolder, 'pdf')
        os.makedirs(pdf_folder, exist_ok=True)
        fig.savefig(
            os.path.join(pdf_folder, f'{name}.pdf'),
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
        )

    if savetiff:
        tiff_folder = os.path.join(plotsfolder, 'tiff')
        os.makedirs(tiff_folder, exist_ok=True)
        fig.savefig(
            os.path.join(tiff_folder, f'{name}.tiff'),
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pil_kwargs={"compression":"tiff_lzw"}
        )

    if savedata and data is not None:
        data_folder = plotsdatafolder if plotsdatafolder is not None else plotsfolder
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder, f'{name}_spec.pkl'), 'wb') as f:
            pickle.dump(data, f)

        # CSV may draw from a different (e.g. combined) spec than the saved pkl.
        csv_source = csv_data if csv_data is not None else data
        if isinstance(csv_source, dict) and 'panels' in csv_source and 'figuretype' in csv_source:
            rows = _plotspec_to_rows(csv_source, name)

            if len(rows) > 0:
                pd.DataFrame(rows).to_csv(
                    os.path.join(data_folder, f'{name}_sourcedata.csv.gz'),
                    index=False,
                    compression='gzip',
                )


def add_colorbar(ax, tick_labels, colors, fontsize=16, label="learning rate", **kwargs):
    cmap = ListedColormap(list(colors.values()) if isinstance(colors, dict) else colors)
    n = len(colors)
    bounds = np.linspace(0, n, n + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        cax=ax,
        orientation=kwargs.get("orientation", "vertical"),
        fraction=kwargs.get("fraction", 0.3),
        pad=kwargs.get("pad", 0.07),
        shrink=kwargs.get("shrink", 0.95),
    )
    cbar.set_label(label, fontsize=0.8 * fontsize)
    cbar.set_ticks(np.arange(n).astype(int) + 0.5)
    cbar.set_ticklabels(tick_labels, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return cbar


def style_spines(ax, lw=1.3, ax_color=(0.23, 0.23, 0.23, 1.0), hide_top=True, hide_right=True):
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(lw)
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_color(ax_color)
    ax.spines["left"].set_color(ax_color)


def configure_axes(
    ax,
    xlabel,
    ylabel=None,
    grid=True,
    fontsize=20,
    xlims=None,
    ylims=None,
    yticks=None,
    xticks=None,
    yticklabels=None,
    xticklabels=None,
    logscale=False,
    grid_color=(0.92, 0.92, 0.92, 1.0),
    ax_color=(0.23, 0.23, 0.23, 1.0),
    spine_lw=0.8,
    tick_width_major=0.7,
    tick_width_minor=0.5,
    tick_length_major=4.5,
    tick_length_minor=2.5,
):
    if logscale:
        ax.set_yscale("log")

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    ax.set_xlabel(xlabel, fontsize=fontsize, color=ax_color)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=ax_color)

    if grid:
        ax.grid(True, color=grid_color, linewidth=0.64, zorder=0)

    style_spines(ax, lw=spine_lw, ax_color=ax_color)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        width=tick_width_major,
        length=tick_length_major,
        colors=ax_color,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=tick_width_minor,
        length=tick_length_minor,
        colors=ax_color,
    )


# =============================================================================
# METRICS LOADING
# =============================================================================

def load_metrics_pretraining(data_folder, save_folder='trainings_metrics', names=None, update=False,
                     drop = None, norm=False, norm_mode='normal',add_gradients=False, mild_rule=False, debug=False):
        
    save_folder = os.path.join(DATA_DIR, 'metrics', os.path.basename(data_folder), save_folder)
    try:
        if update:
            raise RuntimeError("Updating metrics")
        metrics_data = load_var(os.path.join(save_folder,'metrics_data.pkl'))
        if add_gradients:
            metrics = metrics_data.index.get_level_values('metric').unique()
            if sum(['gradients' in name for name in metrics])==0:
                raise ValueError(f' Folder {os.path.basename(data_folder)}: No gradients in metrics data, but requested. Updating metrics.')
        print(f' Folder {os.path.basename(data_folder)}: Metrics loaded.')
        return metrics_data
    except:
        print(f' Folder {data_folder}: Creating metrics...')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
    metrics_data_list = []
    activate_name_check = False
    use_order_variable = False
    if names is None:
        names = os.listdir(data_folder)
        activate_name_check=True
        use_order_variable = True
    for name in names:
        if activate_name_check:
            sim_name = name.split('.')
            order = sim_name[0]
            # check if its first element is a integer
            if sim_name[0].isdigit():
                # remove the first element and create the new one
                sim_name = '.'.join(sim_name[1:])
            else:
                continue
        else:
            sim_name = name
            order = None
        
        sim_folder = os.path.join(data_folder, name)
        params = load_train_params(sim_folder, single_training=True)
        n_iterations = params['iterations']
        if norm:
            rescaling = get_rescaling(params, norm_mode)
        
        exp = None
        for i in os.listdir(sim_folder):
            if i.isdigit():
                print('loading ',sim_name, f'({order}): {int(i)+1}/{n_iterations}', end='\r')
                i_folder = os.path.join(sim_folder, str(i))
                i_metrics = os.path.join(i_folder,'metrics.pkl')
                try:
                    metrics = load_var(i_metrics)
                except:
                    try:
                        exp = Experiment2(sim_folder, str(i), load=True)
                        metrics = deepcopy(exp.train_outputs['metrics'])
                    except:
                        print(f"{name}, {i}, can't be load, skipping.")
                        continue
                for metric_name, values in metrics.items():
                    temp_df = pd.DataFrame(values).T
                    temp_df = temp_df/rescaling if norm else temp_df
                    if use_order_variable:
                        temp_df['order'] = int(order)
                    temp_df['name'] = sim_name
                    temp_df['iteration'] = i
                    temp_df['metric'] = metric_name
                    # Append the temporary DataFrame to the main DataFrame
                    metrics_data_list.append(temp_df)
                
                if add_gradients:
                    if not exp:
                        exp = Experiment2(sim_folder, str(i), load=True)
                    gradients = deepcopy(exp.train_outputs['gradients'])
                    changes = deepcopy(exp.train_outputs['changes'])
                    epochs = exp.parameters['train_settings.settings.train_params.epochs']
                    del exp
                    # NEcessary for the if condition
                    exp = None
                    
                    for quantity, data in zip(['gradients','changes'],[gradients,changes]):
                        for name, d in data.items():
                            if isinstance(d,dict):
                                for k, v in d.items():
                                    if 'weight' in k:
                                        k = f'{quantity}.{name}.{k}'
                                        if len(v) == 2*epochs:
                                            v = v[::2]
                                        elif len(v) == epochs:
                                            pass
                                        elif len(v) == 0:
                                            continue
                                        elif mild_rule:
                                            if len(v) > epochs+1:
                                                v = v[::2]
                                            else:
                                                pass
                                        else:
                                            print(f"iteration {i}, skipping current {quantity} ({name}.{k}), wrong length.")
                                            print(f"{quantity} size: {len(v)}, epochs: {epochs}")
                                            continue
                                        layer_df = pd.DataFrame(v).T
                                        if use_order_variable:
                                            layer_df['order'] = int(order)
                                        layer_df['name'] = sim_name
                                        layer_df['iteration'] = i
                                        layer_df['metric'] = k
                                        metrics_data_list.append(layer_df)
                            elif isinstance(d,list):
                                layer_df = pd.DataFrame(d).T
                                if use_order_variable:
                                    layer_df['order'] = int(order)
                                layer_df['name'] = sim_name
                                layer_df['iteration'] = i
                                layer_df['metric'] = f'{quantity}.{name}'
                                metrics_data_list.append(layer_df)

                if debug:
                    break
        print(f'{sim_name} ({order}) completed.\n' if use_order_variable else f'{sim_name} completed.\n', end='')
        if debug:
            break
    metrics_data = pd.concat(metrics_data_list, ignore_index=True)
    if use_order_variable:
        metrics_data.set_index(['order','name', 'iteration', 'metric'], inplace=True)
    else:
        metrics_data.set_index(['name', 'iteration', 'metric'], inplace=True)
    if drop:
        metrics_data.drop(index=list(drop),level='metric',inplace=True)
    if use_order_variable:
        metrics_data.sort_index(level='order',inplace=True)
    if debug:
        return metrics_data
    save_var(metrics_data, os.path.join(save_folder,'metrics_data.pkl'))
    print('Done.')
    return metrics_data



def load_metrics(
    data_folder,
    save_folder="trainings_metrics",
    names=None,
    update=False,
    drop=None,
    norm=False,
    norm_mode="normal",
    add_gradients=False,
    mild_rule=False,
    debug=False,
):
    save_folder = os.path.join(data_folder, save_folder)

    try:
        if update:
            raise RuntimeError("Updating metrics")
        metrics_data = load_var(os.path.join(save_folder, "metrics_data.pkl"))
        if add_gradients:
            metrics = metrics_data.index.get_level_values("metric").unique()
            if not any("sum_gradients" in name for name in metrics):
                raise ValueError(
                    f"Folder {os.path.basename(data_folder)}: no gradients in metrics data, but requested. Updating metrics."
                )
        print(f"Folder {os.path.basename(data_folder)}: metrics loaded.")
        return metrics_data
    except Exception:
        print(f"Folder {data_folder}: creating metrics...")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    metrics_data_list = []
    activate_name_check = False
    use_order_variable = False

    if names is None:
        names = os.listdir(data_folder)
        activate_name_check = True
        use_order_variable = True

    for name in names:
        if activate_name_check:
            sim_name_parts = name.split(".")
            order = sim_name_parts[0]
            if sim_name_parts[0].isdigit():
                sim_name = ".".join(sim_name_parts[1:])
            else:
                continue
        else:
            sim_name = name
            order = None

        sim_folder = os.path.join(data_folder, name)

        if debug:
            try:
                params = load_train_params(sim_folder, single_training=True)
            except Exception:
                print(f"{name} can't be loaded, skipping (debug mode).")
                continue
        else:
            params = load_train_params(sim_folder, single_training=True)

        n_iterations = params["iterations"]

        if norm:
            rescaling = get_rescaling(params, mode=norm_mode)

        exp = None
        for i in os.listdir(sim_folder):
            if not i.isdigit():
                continue

            print(f"loading {sim_name}, folder {int(i) + 1}/{n_iterations}", end="\n")
            i_folder = os.path.join(sim_folder, str(i))
            i_metrics = os.path.join(i_folder, "metrics.pkl")

            try:
                metrics = load_var(i_metrics)
            except Exception:
                try:
                    exp = Experiment2(sim_folder, str(i), load=True)
                    metrics = deepcopy(exp.train_outputs["metrics"])
                except Exception:
                    print(f"{name}, {i} can't be loaded, skipping.")
                    continue

            for metric_name, values in metrics.items():
                temp_df = pd.DataFrame(values).T
                temp_df = temp_df / rescaling if norm else temp_df

                if use_order_variable:
                    temp_df["order"] = int(order)
                temp_df["name"] = sim_name
                temp_df["iteration"] = i
                temp_df["metric"] = metric_name
                metrics_data_list.append(temp_df)

            if add_gradients:
                if exp is None:
                    exp = Experiment2(sim_folder, str(i), load=True)

                gradients = deepcopy(exp.train_outputs["gradients"])
                changes = deepcopy(exp.train_outputs["changes"])
                epochs = exp.parameters["trainsettings"]["settings"]["trainparams"]["epochs"]
                exp = None

                for quantity, data in zip(["gradients", "changes"], [gradients, changes]):
                    for group_name, d in data.items():
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if "weight" not in k:
                                    continue
                                full_key = f"{quantity}.{group_name}.{k}"
                                if len(v) == 2 * epochs:
                                    v = v[::2]
                                elif len(v) == epochs:
                                    pass
                                elif len(v) == 0:
                                    continue
                                elif mild_rule:
                                    if len(v) == epochs + 1:
                                        v = v[::2]
                                else:
                                    print(f"iteration {i}, skipping current quantity {group_name}.{k}, wrong length.")
                                    print(f"quantity size {len(v)}, epochs {epochs}")
                                    continue

                                layer_df = pd.DataFrame(v).T
                                if use_order_variable:
                                    layer_df["order"] = int(order)
                                layer_df["name"] = sim_name
                                layer_df["iteration"] = i
                                layer_df["metric"] = full_key
                                metrics_data_list.append(layer_df)
                        elif isinstance(d, list):
                            layer_df = pd.DataFrame(d).T
                            if use_order_variable:
                                layer_df["order"] = int(order)
                            layer_df["name"] = sim_name
                            layer_df["iteration"] = i
                            layer_df["metric"] = f"{quantity}.{group_name}"
                            metrics_data_list.append(layer_df)

            print(
                f"{sim_name} order completed." if use_order_variable else f"{sim_name} completed.",
                end="\n",
            )

    metrics_data = pd.concat(metrics_data_list, ignore_index=True)
    if use_order_variable:
        metrics_data.set_index(["order", "name", "iteration", "metric"], inplace=True)
    else:
        metrics_data.set_index(["name", "iteration", "metric"], inplace=True)

    if drop:
        metrics_data.drop(index=list(drop), level="metric", inplace=True)
    if use_order_variable:
        metrics_data.sort_index(level="order", inplace=True)

    if debug:
        return metrics_data

    save_var(metrics_data, os.path.join(save_folder, "metrics_data.pkl"))
    print("Done.")
    return metrics_data


# =============================================================================
# NOTEBOOK HEADER SETUP
# =============================================================================

def notebook_header(HOME, display_width=70, font_family="Arial", seed=25):
    import warnings
    import sys, shutil, json
    from matplotlib import rcParams
    # from IPython.display import display, HTML
    import torch
    from copy import deepcopy
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    warnings.filterwarnings("ignore")

    rcParams["font.family"] = font_family
    rcParams["svg.fonttype"] = "none"
    # display(HTML(f"<style>.container {{ width:{display_width}% !important; }}</style>"))