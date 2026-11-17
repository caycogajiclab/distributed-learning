# %%
"""Figure generation for the distributed-learning (backprop) experiments."""
MACHINE = 'cerbero' # 'cerbero / frontex/ node41'

# %load_ext autoreload
# %autoreload 2


############################################
# SETTINGS
############################################

import os
from ccdl.paths import DATA_DIR, SOURCE_DATA_DIR, FIGURES_DIR, PROJECT_ROOT

WORKDIR = str(DATA_DIR)

import argparse

_bool = lambda v: str(v).lower() in ('1', 'true', 'yes', 'on')
_parser = argparse.ArgumentParser(description="Generate this script's figures (settings default to standalone behavior; generate_figures.py overrides them).")
_parser.add_argument('--plots-data-folder', default=str(SOURCE_DATA_DIR))
_parser.add_argument('--plots-folder', default=str(FIGURES_DIR))
_parser.add_argument('--update-metrics', type=_bool, default=False)
_parser.add_argument('--rebuild-source-data', type=_bool, default=False)
_parser.add_argument('--save-pdf', type=_bool, default=True)
_parser.add_argument('--save-tiff', type=_bool, default=False)
_args, _ = _parser.parse_known_args()

HOME = str(PROJECT_ROOT)
plots_data_folder = _args.plots_data_folder
plots_folder = _args.plots_folder
UPDATE_METRICS = _args.update_metrics
# UPDATE_METRICS implies a source-data rebuild (no point refreshing metrics without regenerating the spec).
REBUILD_SOURCE_DATA = _args.rebuild_source_data or UPDATE_METRICS
save_pdf = _args.save_pdf
save_tiff = _args.save_tiff
# source data is written exactly when it is (re)built — no separate save flag
save_data = REBUILD_SOURCE_DATA

load_folder = os.path.join(DATA_DIR, 'DL-BP')
config_folder = os.path.join(load_folder, 'snapshot')

############################################
# IMPORTS
############################################

from plot_utils import *

# --- progress reporting: emit one line per saved figure (parsed by generate_figures.py) ---
_orig_save_plot = save_plot
def save_plot(*args, **kwargs):
    _fig_name = kwargs.get('name', args[1] if len(args) > 1 else '?')
    _result = _orig_save_plot(*args, **kwargs)
    print(f"[figure] {_fig_name}", flush=True)
    return _result
# --- end progress reporting ---

notebook_header(HOME, display_width=70, font_family="Arial", seed=124)

import pandas as pd
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.cm import Reds
import matplotlib.colors as mcolors

from ccdl.utils import *
from ccdl.reaching import ReachingDataset
from ccdl.ccnet import CerebelloCorticalNet

##############################################################################################
# OTHER SETTINGS

device = torch.device('cpu')
dtype = 'float32'

npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)
rnn_hidden_size = 256
model_class = "CerebelloCorticalNet"

# %% [markdown]
# # Methods

# %%
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_rescaling(params, datatype=None, mode='normal', verbose=False):

    if not datatype:
        try:
            datatype = params['train_settings']['datatype']
            if verbose:
                print(f"Using training datatype: {datatype}")
        except:
            try:
                datatype = params['pretrain_settings']['datatype']
            except:
                datatype = params['settings']['datatype']
            if verbose:
                print(f"Using default datatype: {params['pretrain_settings']['datatype']}")
    
    dataset = get_dataset(params, datatype=datatype)
    target = dataset[1]
    if mode.lower() == 'minmax':
        return target.max().item() - target.min().item()
    elif mode.lower() == 'normal':
        return target.std().item()
    else:
        raise 'Please choose a mode between minmax and normal for rescaling!'

def load_train_params(path, single_training=False):
    r"""Load train/configs_list params from save folder"""
    # loading template_configs.json in path
    if single_training:
        try:
            params_file = os.path.join(path, 'params.json')
            with open(params_file) as f:
                params = json.load(f)
        except:
            params_file = os.path.join(path, '0', 'params.json')
            with open(params_file) as f:
                params = json.load(f)
        return params
    dicts = []
    for filename in ['template_config.json', 'configs_list.json']:
        file = os.path.join(path, filename)
        with open(file, 'r') as f:
            dicts.append(json.load(f))
    return tuple(dicts)

def load_metrics(data_folder, save_folder='trainings_metrics', names=None, update=False,
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
        if debug:
            try:
                params = load_train_params(sim_folder, single_training=True)
            except:
                print(name, "can't be load, skipping. (debug mode)")
                continue
        else:
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
                        pr_print(['red'],f"{name}, {i}, can't be load, skipping.")
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

        print(f'{sim_name} ({order}) completed.\n' if use_order_variable else f'{sim_name} completed.\n', end='')

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

def apply_gaussian_smoothing(data, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    if sigma == 0:
        return data
    # Handling 2D data
    if len(data.shape) == 2:
        # Convert to DataFrame for easy handling if not already one
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # Apply Gaussian smoothing across each row (axis=1)
        smoothed_data = np.apply_along_axis(gaussian_filter1d, axis=1, arr=data, sigma=sigma, radius=max(4,sigma//5))
    # Handling 1D data
    elif len(data.shape) == 1:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        smoothed_data = gaussian_filter1d(data, sigma=sigma, radius=max(4,sigma//2))
    else:
        raise ValueError('data must be 1D or 2D')
    
    return smoothed_data


# =============================================================================
# 1. Compute loss-based contributions
# =============================================================================
def compute_loss_contributions(folder, update=False):
    """
    Loads an experiment and extracts loss metrics for all iterations.
    
    The function assumes that the experiment (loaded via Experiment2) has a dictionary 
    in its train outputs under the key 'metrics', containing:
      - 'mse': the model's MSE loss,
      - 'cortex_error': the loss when using cortical (feedback) error,
      - 'nocb_error': the loss when cerebellar output is removed.
      
    The adapter contribution is computed as:
        a = g - m
    where g corresponds to the 'nocb_error' and m to the 'mse'.
    
    It is assumed that for each metric the experiment stores a list or array for all
    iterations (e.g. 5 iterations), with shape (n_epochs,) per iteration.
    
    Parameters:
        folder (str): The folder where the experiment is stored.
        name (str): The experiment name (used for loading outputs and parameters).
        debug (bool): If True, prints additional debug information.
        
    Returns:
        data (dict): A dictionary containing the following keys:
            - 'm':      np.array of shape (n_iterations, n_epochs) for mse.
            - 'fb':     np.array of shape (n_iterations, n_epochs) for cortex_error.
            - 'g':      np.array of shape (n_iterations, n_epochs) for nocb_error.
            - 'a':      np.array of shape (n_iterations, n_epochs) for adapter contribution (g - m).
    """
    cache_path = os.path.join(DATA_DIR, 'metrics', os.path.basename(os.path.dirname(folder)),
                              os.path.basename(folder), 'loss_contributions.pkl')
    if not update and os.path.exists(cache_path):
        return load_var(cache_path)

    data = {
        'm': [],
        'fb': [],
        'g': []
    }
    for iteration in range(5):
        
        # Load the experiment (assumes Experiment2 and load_var are defined elsewhere)
        exp = Experiment2(folder, str(iteration), load=True)
        
        try:
            metrics = exp.train_outputs['metrics']
        except Exception as e:
            raise ValueError("Could not find 'metrics' in train_outputs. Check experiment outputs.") from e
        
        # Extract the loss arrays for each metric.
        # It is assumed that each is stored as a list or array for each iteration.
        data['m'].append(np.array(metrics['mse']))
        data['fb'].append(np.array(metrics['cortex_error']))
        data['g'].append(np.array(metrics['nocb_error']))
    
    for k, v in data.items():
        data[k]=np.stack(v)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_var(data, cache_path)
    return data

# =============================================================================
# 2. Plot loss contributions (mean across iterations)
# =============================================================================
def plot_loss_contributions(data=None, epochs=None, start=0,stop=None, fontsize=20, lw=2.8, dpi=400, plot_fb=True, figsize=(8,4), alpha=1, replay=None):
    """
    Plots the mean loss-based contributions across iterations.

    The data dictionary should have the following keys (each with shape (n_iterations, n_epochs)):
      - 'm': mse loss.
      - 'fb': cortex error.
      - 'g': nocb_error.

    If an 'epochs' array is not provided, the function creates one assuming a simple range.

    Parameters:
        data (dict): Dictionary as returned by compute_loss_contributions().
        epochs (np.array, optional): Array of epoch indices. If None, will be set as np.arange(n_epochs).
        fontsize (int): Font size for labels and legend.
        lw (float): Line width for plot lines.
        dpi (int): DPI for the figure.

    Returns:
        fig, ax: The figure and axes objects.
    """
    if replay is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for s in replay:
            ax.plot(s['x'], s['y'], label=s['label'], color=s['color'], lw=lw, alpha=alpha)
        ax.spines['left'].set(linewidth=1.5)
        ax.spines['bottom'].set(linewidth=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=3, length=6, labelsize=fontsize)
        ax.set_ylabel('adaptation', fontsize=fontsize)
        ax.set_xlabel('epochs', fontsize=fontsize)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both',which='both',width=1.5, length= 4)
        ax.legend(fontsize=0.8 * fontsize, loc='best')
        return fig, ax

    # Determine the number of epochs from one of the arrays.
    n_iterations, n_epochs = data['m'].shape
    print(data['m'].shape)
    epochs = np.arange(n_epochs)[start:stop]
    # Compute the mean across iterations (axis=0).
    m_mean = data['m'].mean(axis=0)[start:stop]
    fb_mean = data['fb'].mean(axis=0)[start:stop]
    g_mean = data['g'].mean(axis=0)[start:stop]
    
    # Renormalization
    maxi = np.max(m_mean)
    mini = np.min(m_mean)
    m_contribution = 1 - (m_mean - mini) / (maxi - mini)
    g_contribution = 1 - (g_mean - mini) / (maxi - mini)
    fb_contribution = 1 - (fb_mean - mini) / (maxi - mini)
    a_contribution = m_contribution - g_contribution
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.plot(epochs, m_contribution, label='full model', color='k', lw=lw, alpha=alpha)
    ax.plot(epochs, g_contribution, label='controller memory', color='#0c7b11ff', lw=lw, alpha=alpha)
    if plot_fb:
        ax.plot(epochs, fb_contribution, label='controller + feedback', color='C1', lw=lw, alpha=alpha)
    ax.plot(epochs, a_contribution, label='adapter contribution', color='#629ee6ff', lw=lw, alpha=alpha)
    
    ax.spines['left'].set(linewidth=1.5)
    ax.spines['bottom'].set(linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=3, length=6, labelsize=fontsize)
    ax.set_ylabel('adaptation', fontsize=fontsize)
    ax.set_xlabel('epochs', fontsize=fontsize)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both',which='both',width=1.5, length= 4)
    ax.legend(fontsize=0.8 * fontsize, loc='best')

    return fig, ax


def plot_losses_quick_ccnet_partial(metrics_list=None, ax=None, stop=-1, contrast=False,plot_metric=None,
 mode='mean', lims=(1e-8,1e-1), colorbar=False,
                         alpha=0.8, title=None, median_w=0, gaussian_w=0, step=1, lw=1.6,
                         inset=False,stop_inset = 2000, loc=None,fontsize=12,
                         metrics=None, inset_specs=None, labels=None, colors=None, grid=True,
                         bbox_to_anchor=[1.05,0.9], plot_legend=True, axins_ylims=(1e-7,1e-2), capture=None, replay=None):
    from matplotlib.lines import Line2D

    assert (step >=1 and isinstance(step, int)), "Step must be a positive integer"

    if replay is not None:
        assert not contrast, 'contrast overlay not supported in replay mode (not captured)'
        assert not colorbar, 'colorbar path not supported in replay mode (not captured)'
        assert not inset, 'inset overlay not supported in replay mode (not captured)'
        assert mode != 'errorbars', 'errorbars mode not supported in replay mode (not captured)'

        metrics = ['mse', 'cortex_error', 'nocb_error'] if not metrics else metrics
        styles = {
            'mse': 'solid',
            'cortex_error': (0, (1.85*lw, 0.7*lw)),
            'nocb_error': (0, (2.15*lw, 0.9*lw))}
        metrics_labels = {
            'mse': 'full model',
            'cortex_error': 'controller + feedback',
            'nocb_error': 'controller memory',
        }
        if plot_metric is not None:
            linestyles = {metric: 'solid' for metric in metrics}
        else:
            linestyles = {metric: styles[metric] for metric in metrics}
        metrics = linestyles.keys()

        legend_elements = [Line2D([0], [0], linestyle=linestyles[metric], color='k', label=label, lw=lw) for metric, label in metrics_labels.items() if metric in metrics]

        element_order = [entry['name'] for entry in next(iter(replay.values()))]
        entries_by_metric_name = {metric: {e['name']: e for e in entries} for metric, entries in replay.items()}

        if title:
            ax.set_title(title, fontsize=1.3*fontsize)

        for name in element_order:
            for metric in sorted(replay.keys()):
                entry = entries_by_metric_name[metric].get(name)
                if entry is None:
                    continue
                zorder = 2 if metric=='nocb_error' else 3
                if 'RNN' in name:
                    zorder -= 1

                if metric == 'mse':
                    if entry['label'] is not None:
                        legend_elements.append(Line2D([0], [0], linestyle=linestyles[metric], color=entry['color'], label=entry['label'], lw=1.5*lw))
                else:
                    if entry['label'] == 'RNN':
                        continue

                t_alpha = None
                t_lw = lw if mode == 'mean' else 1.3*lw
                ax.plot(entry['x'], entry['y'], linestyle=linestyles[metric], color=entry['color'], lw=t_lw, alpha=t_alpha, zorder=zorder)

        ax.set_yscale('log')
        ax.set_ylim(lims)
        ax.set_xlim(-stop//20, 1.05*stop)
        ax.set_xticks(np.arange(0, stop+1, stop//2))
        grid_color = (233/255, 233/255, 233/255, 97/100)
        if grid:
            ax.grid(True, color=grid_color, linewidth=0.64)
        ax.set_ylabel('error (adaptation)', fontsize=fontsize)
        ax.set_xlabel('epoch', fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set(linewidth=1.3)
        ax.spines['left'].set(linewidth=1.3)
        ax.tick_params(axis='x', which='major', width=1.3, length=4)
        ax.tick_params(axis='y', which='major', width=1.3, length=4)
        ax.tick_params(axis='y', which='minor', width=0.9, length=2.5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_yticks([x for x in [1e-2,1e-4,1e-6,1e-8] if lims[0]<=x<=lims[1]])
        if plot_legend:
            ax.legend(handles=legend_elements,loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        return ax

    all_elements = metrics_list.index.get_level_values('name').unique()
    def _lr_sort_key(name):
        # RNN always last; otherwise sort by float value of lr token (e.g. 'lr1e-4' -> 1e-4)
        if 'RNN' in name:
            return float('inf')
        parts = name.split('.')
        for part in parts:
            if part.startswith('lr'):
                try:
                    return float(part[2:])
                except ValueError:
                    pass
        return 0.0
    elements = sorted(all_elements, key=_lr_sort_key)
    labels = {element: element for element in elements} if not labels else labels
    colors = {element: f'C{i}' for i, element in enumerate(elements)} if not colors else colors
    inset_specs = [0.6,0.5,0.36,0.45] if inset_specs is None else inset_specs
    metrics = ['mse', 'cortex_error', 'nocb_error'] if not metrics else metrics
    styles = {
        'mse': 'solid',
        'cortex_error': (0, (1.85*lw, 0.7*lw)),
        'nocb_error': (0, (2.15*lw, 0.9*lw))}
    
    metrics_labels = {
        'mse': 'full model',
        'cortex_error': 'controller + feedback',
        'nocb_error': 'controller memory',
    }
    
    if plot_metric is not None:
        linestyles = {metric: 'solid' for metric in metrics}
    else:
        linestyles = {metric: styles[metric] for metric in metrics}
    metrics = linestyles.keys()
    N = metrics_list.shape[1]
    if stop == -1:
        stop = N
    x = np.linspace(0, stop-1, stop)[::step]
    legend_elements = [Line2D([0], [0], linestyle=linestyles[metric], color='k', label=label, lw=lw) for metric, label in metrics_labels.items() if metric in metrics]
    if inset:
        #add inset cetered, on the top part of the plot
        axins = ax.inset_axes(inset_specs)
        axins.set_yscale('log')
        if grid:
            axins.yaxis.grid(True)
        axins.tick_params(axis='both',labelsize=fontsize)
        axins.set_ylim(axins_ylims)
        # axins.set_yticks([1e-3,1e-5,1e-7])
        x_inset = np.linspace(0, stop_inset, stop_inset + 1)
    
    RNN_name = None
    for l, element in enumerate(elements):
        selection = metrics_list.copy()
        condition = np.array([element==name for name in selection.index.get_level_values('name')])
        selection = selection.loc[condition]
        
        for metric, data in selection.groupby(['metric']):
            if isinstance(metric, tuple):
                metric = metric[0]
            if plot_metric is not None and metric != plot_metric:
                continue
            if metric not in metrics:
                continue
            zorder = 2 if metric=='nocb_error' else 3
            if 'RNN' in element:
                RNN_name = element
                zorder-=1
            if title:
                ax.set_title(title, fontsize=1.3*fontsize)
            plt.tight_layout()
            color = colors[element]
            
            if metric == 'mse':
                if not colorbar and labels[element] is not None:
                    legend_elements.append(Line2D([0], [0], linestyle=linestyles[metric], color=color, label=labels[element], lw=1.5*lw))
                else:
                    if labels[element] == 'RNN':
                        legend_elements.append(Line2D([0], [0], linestyle=linestyles[metric], color=color, label=labels[element], lw=1.5*lw))
            else:
                if labels[element] == 'RNN':
                    continue
                
            mean = data.values.mean(axis=0)[:stop]
            if mode in ['mean','both', 'errorbars']:
                t_alpha = None
                t_lw = lw if mode == 'mean' else 1.3*lw
                if median_w>0:
                    mean = moving_median(mean, median_w)
                if gaussian_w>0:
                    mean = apply_gaussian_smoothing(mean, gaussian_w)
                if capture is not None:
                    capture[metric].append({
                        'name': element, 'label': labels[element], 'color': color,
                        'x': x.tolist(), 'y': mean[::step].tolist(),
                    })
                if contrast:
                    # ax.plot(x,mean[::step], linestyle=linestyles[metric], color='k', lw=1.03*t_lw)
                    ax.plot(x,mean[::step], linestyle=linestyles[metric], color='w', lw=t_lw, zorder=zorder)
                ax.plot(x,mean[::step], linestyle=linestyles[metric], color=color, lw=t_lw, alpha=t_alpha, zorder=zorder)
            if mode == 'errorbars':
                steps = stop//10
                offset = l * 200
                idx = np.arange(offset, stop, steps)
                means = mean[idx]
                mins = np.minimum(data.values.min(axis=0)[idx],means)
                maxs = np.maximum(data.values.max(axis=0)[idx],means)
                x_errors = x[idx]
                # Plot asymmetric error bars every n time steps
                ax.errorbar(
                    x_errors,
                    means,
                    yerr=[means- mins, maxs - means],
                    fmt='o',
                    color=color,
                    markersize=2,
                    markerfacecolor=color,
                    markeredgewidth=0.7,
                    ecolor=color,
                    elinewidth=0.7,
                    capsize=3, alpha=1
)
            if inset:
                if mode in ['mean','both']:
                    mean = mean[:stop_inset+1]
                    if contrast:
                        # axins.plot(x_inset, mean, linestyle=linestyles[metric],color='k', lw=0.92*lw)
                        axins.plot(x_inset, mean, linestyle=linestyles[metric],color='w', lw=0.9*lw, zorder=zorder)
                    axins.plot(x_inset, mean, linestyle=linestyles[metric],color=color, lw=0.9*lw,alpha=alpha if not colors else None, zorder=zorder)
    ax.set_yscale('log')
    # ax.grid()
    ax.set_ylim(lims)
    ax.set_xlim(-stop//20, 1.05*stop)
    ax.set_xticks(np.arange(0, stop+1, stop//2))
    grid_color = (233/255, 233/255, 233/255, 97/100)
    if grid:
        ax.grid(True, color=grid_color, linewidth=0.64)
    ax.set_ylabel('error (adaptation)', fontsize=fontsize)
    ax.set_xlabel('epoch', fontsize=fontsize)
    # ax.spines['top'].set(linewidth=0.92, color=grid_color, zorder=-1)
    ax.spines['top'].set_visible(False)
    # ax.spines['right'].set(linewidth=0.92, color=grid_color)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set(linewidth=1.3)
    ax.spines['left'].set(linewidth=1.3)
    ax.tick_params(axis='x', which='major', width=1.3, length=4)
    ax.tick_params(axis='y', which='major', width=1.3, length=4)
    ax.tick_params(axis='y', which='minor', width=0.9, length=2.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_yticks([x for x in [1e-2,1e-4,1e-6,1e-8] if lims[0]<=x<=lims[1]])
    if plot_legend:
        ax.legend(handles=legend_elements,loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
    if colorbar:
        cb_colors = deepcopy(colors)
        if RNN_name:
            cb_colors.pop(RNN_name)
        cb_labels = deepcopy(labels)
        if RNN_name:
            cb_labels.pop(RNN_name)
        cb_labels = {k: v[5:] for k, v in cb_labels.items()}
        
        # Create a colorbar showing the effective gains
        cmap = ListedColormap(list(cb_colors.values()))
        n = len(cb_colors)
        bounds = np.linspace(0, n, n + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.3, pad=0.07, shrink=0.6)
        cbar.set_label('Effective gain', fontsize=fontsize)
        cbar.set_ticks(np.arange(n).astype('int') + 0.5)
        cbar.set_ticklabels(cb_labels.values(), fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
    return ax


def add_cbar(ax, tick_labels, colors, fontsize=16):
    cmap = ListedColormap(list(colors.values()))
    n = len(colors)
    bounds = np.linspace(0, n, n + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical', fraction=0.3, pad=0.07, shrink=0.95)
    cbar.set_label('learning rate', fontsize=fontsize)
    cbar.set_ticks(np.arange(n).astype(int) + 0.5)
    cbar.set_ticklabels(tick_labels, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

# %% [markdown]
# # data_processing

# %%
_dlbp_metrics_loaded = None

def get_dlbp_metrics():
    global _dlbp_metrics_loaded
    if _dlbp_metrics_loaded is None:
        mse_data = load_metrics(
            load_folder,
            norm=True,
            norm_mode='normal',
            update=UPDATE_METRICS,
            debug=False,
        )

        names = mse_data.index.get_level_values('name')

        weak_fb = mse_data.loc[
            np.array([
                ('gain.1' in name) for name in names
            ])
        ].copy()

        strong_fb = mse_data.loc[
            np.array([
                ('gain.20' in name) for name in names
            ])
        ].copy()

        _dlbp_metrics_loaded = {'weak_fb': weak_fb, 'strong_fb': strong_fb}
    return _dlbp_metrics_loaded

# %% [markdown]
# # 4b

# %%
def figure_4b(load_folder, start=0, stop=1000, fontsize=20, lw=4.5, dpi=100,
              plot_fb=False, figsize=(8, 4), ylim=(-0.05, 1.05)):
    def prepare():
        folder = os.path.join(load_folder, '0.gain.1.cx_lr.5e-07')
        data = compute_loss_contributions(folder, update=UPDATE_METRICS)

        n_iterations, n_epochs = data['m'].shape
        epochs = np.arange(n_epochs)[start:stop]
        m_mean = data['m'].mean(axis=0)[start:stop]
        fb_mean = data['fb'].mean(axis=0)[start:stop]
        g_mean = data['g'].mean(axis=0)[start:stop]
        maxi = np.max(m_mean)
        mini = np.min(m_mean)
        m_contribution = 1 - (m_mean - mini) / (maxi - mini)
        g_contribution = 1 - (g_mean - mini) / (maxi - mini)
        fb_contribution = 1 - (fb_mean - mini) / (maxi - mini)
        a_contribution = m_contribution - g_contribution

        series = [
            {'name': 'full_model', 'label': 'full model', 'color': 'k',
             'x': epochs.tolist(), 'y': m_contribution.tolist()},
            {'name': 'controller_memory', 'label': 'controller memory', 'color': '#0c7b11ff',
             'x': epochs.tolist(), 'y': g_contribution.tolist()},
            {'name': 'adapter_contribution', 'label': 'adapter contribution', 'color': '#629ee6ff',
             'x': epochs.tolist(), 'y': a_contribution.tolist()},
        ]
        if plot_fb:
            series.insert(2, {'name': 'controller_feedback', 'label': 'controller + feedback',
                               'color': 'C1', 'x': epochs.tolist(), 'y': fb_contribution.tolist()})

        return {
            'figuretype': 'loss_contributions',
            'meta': {'start': start, 'stop': stop, 'fontsize': fontsize, 'lw': lw,
                     'plot_fb': plot_fb, 'ylim': ylim},
            'panels': [{'metric': 'contribution', 'series': series}],
            'data': data,
        }

    def render(spec):
        meta = spec['meta']
        raw_data = spec.get('data')
        if raw_data is not None:
            fig, axs = plot_loss_contributions(raw_data, start=meta['start'], stop=meta['stop'],
                                                fontsize=meta['fontsize'], lw=meta['lw'], dpi=dpi,
                                                plot_fb=meta['plot_fb'], figsize=figsize)
            del spec['data']
        else:
            fig, axs = plot_loss_contributions(fontsize=meta['fontsize'], lw=meta['lw'], dpi=dpi,
                                                figsize=figsize, replay=spec['panels'][0]['series'])
        axs.set_ylim(meta['ylim'])
        return fig, axs

    spec = load_or_build_spec("4b", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
fig, spec = figure_4b(load_folder, fontsize=24
                      )
save_plot(
    fig, 
    "4b", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder,
    dpi=100, 
    transparent=True, 
    bbox_inches='tight'
)
plt.show()
plt.close(fig)

# %% [markdown]
# # 4d + S7 

# %%
def figure_4d_S7(data_key, data_title, split_plots, name, stop=25000, new_stop=1000,
                  lims=[1e-8, 1e-2], figsize=(7, 6), median_w=200, step=1,
                  fontsize=16, metrics=['mse', 'nocb_error'],
                  inset_specs=[0.53, 0.53, 0.46, 0.45], axins_ylims=[5e-7, 8e-3],
                  split_lims=[5e-7, 5e-2],
                  cbar_labels={
                      '1e-05': '1e-5', '1e-06': '1e-6', '2e-06': '2e-6',
                      '5e-06': '5e-6', '5e-07': '5e-7',
                  }, dpi=100):
    def prepare():
        mse_data = get_dlbp_metrics()[data_key]
        # Extract LR token, sort numerically, RNN last
        all_names = mse_data.index.get_level_values('name').unique()
        names = sorted(all_names, key=lambda n: float(n.split('.')[3]), reverse=True)
        lr_tokens = [n.split('.')[3] for n in names]           # e.g. ['5e-07', '1e-06', ...]
        cbar_ticks = [cbar_labels[lr] for lr in lr_tokens]
        colors = {name: Reds(1 - 0.8 * (len(names) - 1 - i) / len(names)) for i, name in enumerate(names)}
        labels = {name: None for name in names}

        used_stop = new_stop if split_plots else stop
        used_lims = split_lims if split_plots else lims
        grid = split_plots

        return {
            'figuretype': 'loss_curves',
            'meta': {
                'data_title': data_title, 'split_plots': split_plots,
                'used_stop': used_stop, 'used_lims': used_lims, 'grid': grid,
                'median_w': median_w, 'step': step, 'fontsize': fontsize,
                'metrics': metrics, 'cbar_ticks': cbar_ticks,
            },
            'panels': [{'metric': m, 'series': []} for m in metrics],
            'mse_data': mse_data, 'names': names, 'colors': colors, 'labels': labels,
        }

    def render(spec):
        meta = spec['meta']
        used_stop = meta['used_stop']
        used_lims = meta['used_lims']
        grid = meta['grid']
        colors = spec['colors']
        labels = spec['labels']
        raw_data = spec.get('mse_data')
        if raw_data is not None:
            for panel in spec['panels']:
                panel['series'] = []
        panel_by_metric = {panel['metric']: panel['series'] for panel in spec['panels']}

        pr_print(['bold', 'red'], data_title + ('.split' if split_plots else ''))

        if split_plots:
            fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi,
                                    gridspec_kw={'width_ratios': [30, 1], 'height_ratios': [1, 1]})
            for ax, plot_metric in zip(axs[:, 0], metrics):
                if raw_data is not None:
                    ax = plot_losses_quick_ccnet_partial(
                        raw_data, ax, stop=used_stop,
                        mode='mean', lims=used_lims, plot_metric=plot_metric,
                        metrics=[plot_metric], gaussian_w=0,
                        alpha=0.8, title=None, median_w=meta['median_w'], step=meta['step'],
                        lw=1.7, labels=labels, colors=colors, inset=False, grid=grid,
                        stop_inset=1000, loc=(0.2, 0.495), inset_specs=inset_specs,
                        fontsize=meta['fontsize'], colorbar=False, axins_ylims=axins_ylims,
                        plot_legend=True, bbox_to_anchor=[0.4, 0.8], contrast=False,
                        capture=panel_by_metric)
                else:
                    ax = plot_losses_quick_ccnet_partial(
                        ax=ax, stop=used_stop,
                        mode='mean', lims=used_lims, plot_metric=plot_metric,
                        metrics=[plot_metric],
                        title=None, median_w=meta['median_w'], step=meta['step'],
                        lw=1.7, grid=grid,
                        loc=(0.2, 0.495),
                        fontsize=meta['fontsize'], colorbar=False,
                        plot_legend=True, bbox_to_anchor=[0.4, 0.8], contrast=False,
                        replay={plot_metric: panel_by_metric[plot_metric]})
            for ax in axs[:, 1]:
                add_cbar(ax, meta['cbar_ticks'], colors, fontsize=meta['fontsize'])
        else:
            fig, axs = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] * 0.5), dpi=dpi,
                                    gridspec_kw={'width_ratios': [30, 1]})
            if raw_data is not None:
                axs[0] = plot_losses_quick_ccnet_partial(
                    raw_data, axs[0], stop=used_stop,
                    mode='mean', lims=used_lims, plot_metric=None,
                    metrics=meta['metrics'], gaussian_w=0,
                    alpha=0.8, title=None, median_w=meta['median_w'], step=meta['step'],
                    lw=1.7, labels=labels, colors=colors, inset=False, grid=grid,
                    stop_inset=1000, loc=(0.2, 0.495), inset_specs=inset_specs,
                    fontsize=meta['fontsize'], colorbar=False, axins_ylims=axins_ylims,
                    plot_legend=True, bbox_to_anchor=[0.3, 0.8], contrast=False,
                    capture=panel_by_metric)
            else:
                axs[0] = plot_losses_quick_ccnet_partial(
                    ax=axs[0], stop=used_stop,
                    mode='mean', lims=used_lims, plot_metric=None,
                    metrics=meta['metrics'],
                    title=None, median_w=meta['median_w'], step=meta['step'],
                    lw=1.7, grid=grid,
                    loc=(0.2, 0.495),
                    fontsize=meta['fontsize'], colorbar=False,
                    plot_legend=True, bbox_to_anchor=[0.3, 0.8], contrast=False,
                    replay=panel_by_metric)
            add_cbar(axs[1], meta['cbar_ticks'], colors, fontsize=meta['fontsize'])

        plt.tight_layout()
        if raw_data is not None:
            del spec['mse_data']
        return fig, axs

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
panels = {
    '4d': ('weak_fb', "weak feedback", False),
    'S7ac': ('weak_fb', "weak feedback", True),
    'S7bd': ('strong_fb', "strong feedback", True)
}



for panel_name, (data_key, data_title, split_plots) in panels.items():
    fig, spec = figure_4d_S7(
        data_key,
        data_title,
        split_plots,
        name=panel_name
    )
    save_plot(
        fig, 
        panel_name, 
        plots_folder, 
        savepdf=save_pdf, 
        savetiff=save_tiff,
        savedata=save_data, 
        data=spec, 
        plotsdatafolder=plots_data_folder,
        dpi=100, 
        transparent=True, 
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)

# %% [markdown]

# %%
def figure_4c_reaching(load_folder, iteration='1', figsize=(16, 4), dpi=400,
                        plot_specs={'epochs_list': [1, 2, 5, 10, 20, 50], 'steps': 500,
                                    'color_range': [0.1, 1]},
                        lims=[[-0.12, 0.12], [0.22, 0.44]], fontsize=13,
                        targets_lw=6, targets_alpha=0.1, excluded=[],
                        trial_alpha=0.55, trial_lw=1.25, debug=False, gain=None,
                        selected_task='all', markersize=140, edge=2, contrast=1.1,
                        plot_axes=False, colorbar='each', title='', legend_position=(1.7, 0.7)):
    def prepare():
        path = os.path.join(load_folder, '0.gain.1.cx_lr.5e-07')
        return {
            'figuretype': 'reaching_evolution',
            'meta': {
                'path': path, 'iteration': iteration, 'plot_specs': plot_specs, 'lims': lims,
                'fontsize': fontsize, 'targets_lw': targets_lw, 'targets_alpha': targets_alpha,
                'trial_alpha': trial_alpha, 'trial_lw': trial_lw, 'selected_task': selected_task,
                'gain': gain,
            },
            'panels': [{'metric': 'reaching', 'series': []}],
        }

    def render(spec):
        meta = spec['meta']
        panels = spec.setdefault('panels', [{'metric': 'reaching', 'series': []}])
        series = panels[0]['series']
        kwargs = dict(
            figsize=figsize, dpi=dpi, targets_lw=meta['targets_lw'],
            targets_alpha=meta['targets_alpha'], return_fig=True, selected_task=meta['selected_task'],
            trial_alpha=meta['trial_alpha'], trial_lw=meta['trial_lw'], lims=meta['lims'],
            markersize=markersize, edge=edge, plot_specs=meta['plot_specs'], cmap=rainbow,
            excluded=excluded, contrast=contrast, gain=meta['gain'],
            plot_axes=plot_axes, colorbar=colorbar, title=title, debug=debug,
            legend_position=legend_position)
        if series:
            result = plot_reachings_evolution(meta['path'], meta['iteration'], replay=series, **kwargs)
        else:
            result = plot_reachings_evolution(meta['path'], meta['iteration'], capture=series, **kwargs)
        fig = result[0]
        return fig, fig.axes

    spec = load_or_build_spec("4c", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
fig, spec = figure_4c_reaching(load_folder, iteration='1')
save_plot(
    fig, 
    "4c", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder,
    dpi=400, 
    transparent=True, 
    bbox_inches='tight'
)
plt.show()
plt.close(fig)


# %%
