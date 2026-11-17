# %%
"""Figure generation for the distributed-learning (local plasticity) experiments."""
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

load_folder = os.path.join(DATA_DIR, 'DL-local')
config_folder = os.path.join(load_folder, 'snapshot')
pretrain_folder = os.path.join(DATA_DIR, "pretrainings","cortex")

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

import json
import pandas as pd
import torch
from torch import nn
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow, Greens
from typing import List
import matplotlib.colors as mcolors

from ccdl.utils import *

##############################################################################################
# OTHER SETTINGS

device = torch.device('cpu')
dtype = 'float32'

npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)
rnn_hidden_size = 256
model_class = "CerebelloCorticalNet"

# %% [markdown]
# ## Functions
#

# %%
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm

def _dllocal_fb_meta_path():
    return os.path.join(plots_data_folder, 'DL-local_fb_meta.pkl')


def get_mean_fb_weights_nuc_norm(source):
    # Raw-free path: reuse the cached feedback nuc-norm (written by
    # get_dllocal_data during an UPDATE_METRICS build) so rendering does not
    # need the pretraining cortex checkpoints. Falls back to computing it from
    # `source` when the cache is absent (identical value).
    meta_path = _dllocal_fb_meta_path()
    if not UPDATE_METRICS and os.path.exists(meta_path):
        return load_var(meta_path)['fb_nuc_norm_floor']
    norm = []
    for it in os.listdir(source):
        if it.isnumeric():
            state_dict = load_var(os.path.join(source, it))
            norm.append(torch.linalg.matrix_norm(state_dict['rnn.input2h_2.weight'],ord='nuc'))
    return np.floor(np.mean(norm))

def get_rescaling(params, datatype=None, mode='normal', verbose=False):

    if not datatype:
        try:
            datatype = params['train_settings']['datatype']
            if verbose:
                print(f"Using training datatype: {datatype}")
        except:
            datatype = params['pretrain_settings']['datatype']
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
def compute_loss_contributions(mse_data, name):
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
    data = {
        'm': [],
        'fb': [],
        'g': []
    }
    selection = [i_name==name for i_name in mse_data.index.get_level_values('name')]
    selected_data = mse_data.loc[selection]
    for iteration in range(5):
        # Extract the loss arrays for each metric.
        # It is assumed that each is stored as a list or array for each iteration.
        selection_i = [i_name==str(iteration) for i_name in selected_data.index.get_level_values('iteration')]
        selected_data_i = selected_data.loc[selection_i]
        
        data['m'].append(selected_data_i.loc[selected_data_i.index.get_level_values('metric')=='mse'].values.squeeze(axis=0))
        data['fb'].append(selected_data_i.loc[selected_data_i.index.get_level_values('metric')=='cortex_error'].values.squeeze(axis=0))
        data['g'].append(selected_data_i.loc[selected_data_i.index.get_level_values('metric')=='nocb_error'].values.squeeze(axis=0))
    
    for k, v in data.items():
        data[k]=np.stack(v)
    return data

# =============================================================================
# 2. Plot loss contributions (mean across iterations)
# =============================================================================

def plot_loss_contributions(data=None, epochs=None, start=0,stop=None, fontsize=20, lw=1, dpi=400,
                            selection=None,clip_values=False,split=False,figsize=(8,4), invert_colors=False,
                            norm=True, contribution=True, ylims=None, plot_fb=False, gain_values=False,
                            legend_loc=None, wspace=0.5, cb_label=None, pretrain_folder=pretrain_folder, capture=None, replay=None):
    """
    Plots the mean loss-based contributions across iterations.
    
    The data dictionary should have the following keys (each with shape (n_iterations, n_epochs)):
      - 'm': mse loss.
      - 'fb': cortex error.
      - 'g': nocb_error.
      - 'a': adapter contribution (g - m).
      
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
    def settings(ax, fontsize, values, alphas, xlabel=True, ylims=None, bbox_to_anchor=legend_loc, colorb=False,ylabel=False, cb_label=cb_label):
        ax_width=0.8
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set(linewidth=ax_width, zorder=5) #, color = ax_color)
        ax.spines['left'].set(linewidth=ax_width, zorder=5) #, color = ax_color)
        if ylabel:
            ax.set_ylabel('adaptation', fontsize=fontsize) # color=ax_color
        if xlabel==True:
            ax.set_xlabel('epochs', fontsize=fontsize) # color=ax_color
        ax.legend(fontsize=0.8 * fontsize, bbox_to_anchor=bbox_to_anchor) # , labelcolor=ax_color
        ax.tick_params(axis='both', labelsize=fontsize, width=ax_width, length=4.5, zorder=5) # , colors=ax_color
        if ylims:
            ax.set_ylim(*ylims)
        
        colors = [(0, 0, 0, alphas[g]) for g in values]  # RGBA tuples
        # Create a colorbar showing values
        if colorb:
            cmap = ListedColormap(list(colors))
            n = len(colors)
            bounds = np.linspace(0, n, n + 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.3, pad=0.07, shrink=0.85)
            cbar.set_label('feedback strength' if not cb_label else cb_label, fontsize=fontsize)
            cbar.set_ticks(np.arange(n).astype('int') + 0.5)
            if gain_values:
                labels = [f"{val:.1f}" for val in np.array(values,dtype=float)*get_mean_fb_weights_nuc_norm(pretrain_folder)]
            else:
                labels = [v.replace('-', 'e-') if 'e' not in v else v for v in values]

            cbar.set_ticklabels(labels)
            cbar.ax.tick_params(labelsize=fontsize)# color=ax_color
        
    blue = '#629ee6ff'
    green = '#0c7b11ff'
    orange = '#c97c14ff'

    if replay is not None:
        assert not split, "split=True not supported in replay mode (not captured)"
        component_colors = {
            'full_model': 'k', 'controller_memory': green,
            'controller_feedback': orange, 'adapter_memory': blue,
        }
        component_zorders = {
            'full_model': 2, 'controller_memory': 3,
            'controller_feedback': 4, 'adapter_memory': 5,
        }
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        values = [entry['name'] for entry in replay]
        alphas = {entry['name']: entry['alpha'] for entry in replay}
        for entry in replay:
            for comp in entry['components']:
                ax.plot(comp['x'], comp['y'], label=comp['label'],
                        color=component_colors[comp['component']], lw=lw,
                        alpha=entry['alpha'], zorder=component_zorders[comp['component']])
        settings(ax, fontsize, values, alphas, xlabel=True, ylims=ylims, colorb=True, ylabel=True)
        return fig, ax

    values = list(data.keys())
    if selection:
        assert isinstance(selection,list), "selection should be a list"
        new_values = []
        for s in selection:
            for v in values:
                if v == s:
                    new_values.append(v)
    else:
        new_values = values
    
    values = new_values
    sorting = np.argsort(np.array(new_values, dtype='float'))
    values = [values[k] for k in sorting]
    if invert_colors:
        values = values[::-1]
        
    alphas = {v: a for (a,v) in zip(np.linspace(0.2,0.9,len(values)),values)}
    if split:
        fig, axs = plt.subplots(1,3,figsize=figsize, dpi=dpi,sharey=True, gridspec_kw={'width_ratios':[1,1,1.6]})
        plt.subplots_adjust(wspace=wspace)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    for value in values:
        alpha = alphas[value]
        # Determine the number of epochs from one of the arrays.
        n_iterations, n_epochs = data[value]['m'].shape
        epochs = np.arange(n_epochs)[start:stop]
        # Compute the mean across iterations (axis=0).
        m_mean = data[value]['m'].mean(axis=0)[start:stop]
        fb_mean = data[value]['fb'].mean(axis=0)[start:stop]
        g_mean = data[value]['g'].mean(axis=0)[start:stop]
        
        # Renormalization
        maxi = np.max(m_mean)
        mini = np.min(m_mean)
        if norm:
            if contribution:
                m_mean = 1 - (m_mean - mini) / (maxi - mini)
                g_mean = 1 - (g_mean - mini) / (maxi - mini)
                if clip_values:
                    g_mean = np.minimum(g_mean, m_mean)
                fb_mean = 1 - (fb_mean - mini) / (maxi - mini)
                a_mean = m_mean - g_mean
            else:
                m_mean = (m_mean - mini) / (maxi - mini)
                g_mean = (g_mean - mini) / (maxi - mini)
                fb_mean = (fb_mean - mini) / (maxi - mini)
                a_mean = g_mean - m_mean
        else:
            a_mean = g_mean - m_mean
        # if clip_values:
        #     a_contribution = np.maximum(a_contribution, 0)
        
        # Create the plot.
        
        if split:
            ax = axs[0]
            ax.plot(epochs, m_mean, label='full model' if value==values[-1] else None, color='k', lw=lw, alpha=alpha,zorder=2)
        else:
            ax.plot(epochs, m_mean, label='full model' if value==values[-1] else None, color='k', lw=lw, alpha=alpha,zorder=2)
        ax.plot(epochs, g_mean, label='controller memory' if value==values[-1] else None, color=green, lw=lw, alpha=alpha,zorder=3)
        
        if plot_fb:
            if split:
                ax = axs[1]
                ax.plot(epochs, m_mean, label='full model' if value==values[-1] else None, color='k', lw=lw, alpha=alpha,zorder=2)
            ax.plot(epochs, fb_mean, label='controller + feedback' if value==values[-1] else None, color=orange, lw=lw, alpha=alpha,zorder=4)
        
        if split:
            ax = axs[2]
            ax.plot(epochs, m_mean, label='full model' if value==values[-1] else None, color='k', lw=lw, alpha=alpha,zorder=2)
        ax.plot(epochs, a_mean, label='adapter memory' if value==values[-1] else None, color=blue, lw=lw, alpha=alpha,zorder=5)

        if capture is not None:
            is_last = (value == values[-1])
            components = [
                {'component': 'full_model', 'label': 'full model' if is_last else None,
                 'x': epochs.tolist(), 'y': m_mean.tolist()},
                {'component': 'controller_memory', 'label': 'controller memory' if is_last else None,
                 'x': epochs.tolist(), 'y': g_mean.tolist()},
            ]
            if plot_fb:
                components.append({'component': 'controller_feedback',
                                    'label': 'controller + feedback' if is_last else None,
                                    'x': epochs.tolist(), 'y': fb_mean.tolist()})
            components.append({'component': 'adapter_memory', 'label': 'adapter memory' if is_last else None,
                                'x': epochs.tolist(), 'y': a_mean.tolist()})
            capture.append({'name': str(value), 'alpha': alpha, 'components': components})

    if not split:
        axs = [ax]
    for k, ax in enumerate(axs):
        settings(ax, fontsize, values, alphas, xlabel = True, ylims=ylims, colorb= True if (k==2 or not split) else False, ylabel=True if k==0 else False)
    
    return fig, ax


# =============================================================================
# 2. Compute loss-based contributions
# =============================================================================

def compute_lr_loss_contributions(mse_data, to_plot):
    
    names = mse_data.index.get_level_values('name').unique()
    keys = {}
    for k in to_plot:
        presence = [(k in n) for n in names]
        if any(presence):
            if sum(presence)==1:
                name = names[presence].item()
            else:
                name = names[[n == k for n in names]].item()
            keys[name] = k
    
    selection = [name in keys.keys() for name in mse_data.index.get_level_values('name')]
    selected_data = mse_data.loc[selection]
    lrs_data = {}
    for name, k in keys.items():
        lrs_data[k] = compute_loss_contributions(selected_data, name)
    return lrs_data


def plot_losses_quick_ccnet_partial(metrics_list=None, ax=None, stop=-1, contrast=False,plot_metric=None,
                         mode='mean', lims=(1e-8,1e-1), scheduler_label=True,colorbar=False,
                         alpha=0.8, title=None, median_w=0, gaussian_w=0, step=1, lw=1.6,
                         inset=False,stop_inset = 2000, loc=None,fontsize=12,labels_sort=False,
                         metrics=None, inset_specs=None, labels=None, colors=None, grid=True,
                         bbox_to_anchor=[1.05,0.9], plot_legend=True, force_legend=False, axins_ylims=(1e-7,1e-2), capture=None, replay=None):
    from matplotlib.lines import Line2D
    from collections.abc import Iterable

    assert (step >=1 and isinstance(step, int)), "Step must be a positive integer"

    if replay is not None:
        assert mode == 'mean', 'only mode="mean" is captured/replayed'
        assert not contrast, 'contrast overlay not supported in replay mode (not captured)'
        assert not colorbar, 'colorbar path not supported in replay mode (not captured)'
        assert not inset, 'inset overlay not supported in replay mode (not captured)'
        assert not force_legend, 'force_legend not supported in replay mode (not captured)'
        assert colors is not None, 'colors=None (alpha-from-value) path not supported in replay mode (not captured)'

        metrics = ['mse', 'cortex_error', 'nocb_error'] if not metrics else metrics
        styles = {
            'mse': 'solid',
            'cortex_error': (0, (7, 5)),
            'nocb_error': (0, (8, 5))}
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

        legend_elements = [Line2D([0], [0], linestyle=linestyles[metric], color='k', label=label, lw=lw)
                            for metric, label in metrics_labels.items() if metric in metrics]

        element_order = [entry['name'] for entry in next(iter(replay.values()))]
        entries_by_metric_name = {metric: {e['name']: e for e in entries} for metric, entries in replay.items()}

        if title:
            ax.set_title(title, fontsize=1.3*fontsize)

        for name in element_order:
            for metric in sorted(replay.keys()):
                entry = entries_by_metric_name[metric].get(name)
                if entry is None:
                    continue
                zorder = 2 if metric == 'nocb_error' else 3
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
            ax.legend(handles=legend_elements, loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        return ax

    if labels_sort:
        labels = labels
        elements = labels.keys()
    else:
        elements = metrics_list.index.get_level_values('name').unique()
        labels = {element: element for element in elements} if not labels else labels
    colors = {element: f'C{i}' for i, element in enumerate(elements)} if not colors else colors
    inset_specs = [0.6,0.5,0.36,0.45] if inset_specs is None else inset_specs
    metrics = ['mse', 'cortex_error', 'nocb_error'] if not metrics else metrics
    styles = {
        'mse': 'solid', 
    'cortex_error': (0, (7,5)), 
    'nocb_error': (0, (8,5))}
    
    metrics_labels = {
        'mse': 'full model',
        'cortex_error': 'controller + feedback',
        'nocb_error': 'controller memory',
    }
    nocb_ls = (0, (8, 5))
    
    if plot_metric is not None:
        linestyles = {metric: 'solid' for metric in metrics}
    else:
        linestyles = {metric: styles[metric] for metric in metrics}
    metrics = linestyles.keys()
    N = metrics_list.shape[1]
    if stop == -1:
        stop = N
    x = np.linspace(0, stop-1, stop)[::step]
    legend_elements = [Line2D([0], [0], linestyle=linestyles[metric], color='k', label=label, lw=lw) for metric, label in metrics_labels.items() if metric in metrics] if not force_legend else []
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
            metric = metric[0] if isinstance(metric, List) else metric
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
            
            if (metric == 'mse' or force_legend):
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
                if not colors:
                    t_alpha = alpha if mode == 'mean' else max(alpha,0.9)
                else:
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
            if mode in ['all','both']:
                t_alpha = alpha if mode == 'all' else 0.2
                t_lw = lw if mode == 'all' else 0.8*lw
                for k in range(data.shape[0]):
                    series = data.values[k, :stop]
                    if median_w>0:
                        series = moving_median(series, median_w)
                    if gaussian_w>0:
                        series = apply_gaussian_smoothing(series, gaussian_w//10)
                    ax.plot(x,series[::step], linestyle=linestyles[metric], color=color, lw=t_lw, alpha=t_alpha, zorder=zorder-1)
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
                if mode in ['all','both']:
                    for k in range(data.shape[0]):
                        series = data.values[k, :stop_inset+1]
                        if median_w>0:
                            series = moving_median(series, median_w//10)
                        if gaussian_w>0:
                            series = apply_gaussian_smoothing(series, gaussian_w//10)
                        axins.plot(x_inset, series, linestyle=linestyles[metric],color=color, lw=0.9*lw, alpha=0.2, zorder=zorder-1)
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

# %% [markdown]
# ## data processing

# %%
_dllocal_data_loaded = None

def get_dllocal_data():
    global _dllocal_data_loaded
    if _dllocal_data_loaded is None:
        mse_data = load_metrics(
            load_folder,
            norm=True,
            norm_mode='normal',
            update=UPDATE_METRICS,
            debug=False,
        )

        names = mse_data.index.get_level_values('name')

        lr_data = mse_data.loc[
            np.array([
                ('gain' not in name) for name in names
            ])
        ].copy()
        print(lr_data.shape)

        #removing duplicated configuration (n. 5)
        job_ids = lr_data.index.get_level_values('order')
        lr_data = lr_data.loc[
            np.array([
                id !=5 for id in job_ids
            ])
        ].copy()
        print(lr_data.shape)

        gain_data = mse_data.loc[
            np.array([
                ('gain' in name) for name in names
            ])
        ].copy()



        # Raw-free path: reuse cached effective gains (derived from the
        # pretraining cortex checkpoints) so a from-metrics rebuild does not need
        # the raw weights. Falls back to computing them (identical values) and
        # caches the result — plus floor(W_fb) for get_mean_fb_weights_nuc_norm.
        meta_path = _dllocal_fb_meta_path()
        if not UPDATE_METRICS and os.path.exists(meta_path):
            effective_gains = load_var(meta_path)['effective_gains']
        else:
            params, config_list = load_train_params(config_folder)
            W_fb = []
            for order in range(5):
                cortex_sd_path = os.path.join(WORKDIR, 'pretrainings', 'cortex', str(order))
                reaching = ReachingDataset(params['dataset_params'])
                std = reaching.generate_unperturbed_dataset('cart')[1].std().item()
                in_features = reaching.input_size[2]

                rnn_hidden_size = params['rnn_hidden_size']

                params['net_params']['cereb_specs']['sizes'] = [1024,1024]
                params['net_params']['device'] = 'cpu'
                model = CerebelloCorticalNet(in_features, rnn_hidden_size, **params['net_params'])
                model_state = params['model_state']
                model.cortex.load_state_dict(load_var(cortex_sd_path))
                # model_state['plant'] = True
                model.eval(**model_state)
                loss = nn.MSELoss()

                W_fb.append(torch.linalg.matrix_norm(model.cortex.rnn.input2h_2.weight,ord='nuc'))

            W_fb = np.mean(W_fb)
            gains = {name: float(name.split('.')[1]) for name in gain_data.index.get_level_values('name').unique()}
            effective_gains = {name: gain*W_fb for name,gain in gains.items()}
            effective_gains = {k: f'{v:.1f}' if v<1 else f'{int(v)}' for k,v in effective_gains.items()}

            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            save_var(
                {'effective_gains': effective_gains, 'fb_nuc_norm_floor': float(np.floor(W_fb))},
                meta_path,
            )

        _dllocal_data_loaded = {
            'lr_data': lr_data,
            'gain_data': gain_data,
            'effective_gains': effective_gains,
        }
    return _dllocal_data_loaded

# %% [markdown]
# ## Losses: S11a-b

# %%
def figure_S11a(lims=[3e-8, 5e-2], figsize=(10, 4), median_w=50, step=10,
                metrics=['mse', 'nocb_error'], stop=30000, fontsize=18, lw=1.4, dpi=200,
                axins_ylims=[5e-7, 8e-3], inset=False, inset_specs=[0.6, 0.65, 0.4, 0.35],
                loc=(0.2, 0.495), bbox_to_anchor=[1.03, -0.2], yticks=[1e-2, 1e-4, 1e-6]):
    def prepare():
        lr_data = get_dllocal_data()['lr_data']
        mse_list = lr_data.copy()
        names = mse_list.index.get_level_values('name').unique()
        lrs = np.argsort([float(name.split('.')[1]) for name in names])
        names = [names[i] for i in lrs]
        labels = {name: f"local lr = {name.split('.')[1]}" for name in names}
        colors = {name: Greens(0.9 - 0.5 * i / (len(names) - 1)) for i, name in enumerate(names)}
        return {
            'figuretype': 'losses_lr',
            'meta': {'lims': lims, 'stop': stop, 'median_w': median_w, 'step': step,
                      'fontsize': fontsize, 'metrics': metrics},
            'panels': [{'metric': m, 'series': []} for m in metrics],
            'mse_list': mse_list, 'labels': labels, 'colors': colors,
        }

    def render(spec):
        raw_data = spec.get('mse_list')
        if raw_data is not None:
            for panel in spec['panels']:
                panel['series'] = []
        panel_by_metric = {panel['metric']: panel['series'] for panel in spec['panels']}
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        temp_labels = deepcopy(spec['labels'])
        if raw_data is not None:
            ax = plot_losses_quick_ccnet_partial(
                raw_data, ax, stop=stop, mode='mean', lims=lims, metrics=metrics, gaussian_w=0,
                alpha=0.95, title='', median_w=median_w, step=step, lw=lw, labels=temp_labels,
                colors=spec['colors'], axins_ylims=axins_ylims, inset=inset, stop_inset=1000,
                loc=loc, inset_specs=inset_specs, fontsize=fontsize, labels_sort=True,
                bbox_to_anchor=bbox_to_anchor, capture=panel_by_metric)
            del spec['mse_list']
        else:
            ax = plot_losses_quick_ccnet_partial(
                ax=ax, stop=stop, mode='mean', lims=lims, metrics=metrics,
                title='', median_w=median_w, step=step, lw=lw,
                colors=spec['colors'], axins_ylims=axins_ylims, inset=inset, stop_inset=1000,
                loc=loc, inset_specs=inset_specs, fontsize=fontsize,
                bbox_to_anchor=bbox_to_anchor, replay=panel_by_metric)
        ax.set_ylim(lims)
        ax.set_yticks(yticks)
        plt.tight_layout()
        return fig, ax

    spec = load_or_build_spec("S11a", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
fig, spec = figure_S11a()
save_plot(fig, "S11a", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=200, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %%
def figure_S11c(lims=[3e-8, 5e-2], figsize=(10, 4), median_w=50,
                step=10, metrics=['mse', 'nocb_error'], stop=30000, fontsize=18, lw=1.4, dpi=200,
                axins_ylims=[5e-7, 8e-3], inset=False, inset_specs=[0.6, 0.65, 0.4, 0.35],
                loc=(0.2, 0.495), yticks=[1e-2, 1e-4, 1e-6], cbar_fontsize=20, cbar_ticklabelsize=22):
    def prepare():
        data = get_dllocal_data()
        gain_data = data['gain_data']
        effective_gains = data['effective_gains']
        mse_list = gain_data.copy()
        names = mse_list.index.get_level_values('name').unique()
        lrs = np.argsort([float(name.split('.')[1]) for name in names])[::-1]
        names = [names[i] for i in lrs]
        labels = {name: f"local lr = {name.split('.')[1]}" for name in names}
        colors = {name: Greens(0.9 - 0.5 * i / (len(names) - 1)) for i, name in enumerate(names)}
        e_gains = {name: effective_gains[name] for name in names}
        return {
            'figuretype': 'losses_gain',
            'meta': {'lims': lims, 'stop': stop, 'median_w': median_w, 'step': step,
                      'fontsize': fontsize, 'metrics': metrics},
            'panels': [{'metric': m, 'series': []} for m in metrics],
            'mse_list': mse_list, 'labels': labels, 'colors': colors, 'e_gains': e_gains,
        }

    def render(spec):
        raw_data = spec.get('mse_list')
        if raw_data is not None:
            for panel in spec['panels']:
                panel['series'] = []
        panel_by_metric = {panel['metric']: panel['series'] for panel in spec['panels']}
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        temp_labels = deepcopy(spec['labels'])
        colors = spec['colors']
        if raw_data is not None:
            ax = plot_losses_quick_ccnet_partial(
                raw_data, ax, stop=stop, mode='mean', lims=lims, metrics=metrics, gaussian_w=0,
                alpha=0.95, title='', median_w=median_w, step=step, lw=lw, labels=temp_labels,
                colors=colors, axins_ylims=axins_ylims, inset=inset, loc=loc,
                inset_specs=inset_specs, fontsize=fontsize, plot_legend=False, capture=panel_by_metric)
            del spec['mse_list']
        else:
            ax = plot_losses_quick_ccnet_partial(
                ax=ax, stop=stop, mode='mean', lims=lims, metrics=metrics,
                title='', median_w=median_w, step=step, lw=lw,
                colors=colors, axins_ylims=axins_ylims, inset=inset, loc=loc,
                inset_specs=inset_specs, fontsize=fontsize, plot_legend=False, replay=panel_by_metric)
        ax.set_ylim(lims)
        ax.set_yticks(yticks)

        cmap = ListedColormap(colors.values())
        n = len(colors)
        bounds = np.linspace(0, n, n + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.3, pad=0.07, shrink=0.75)
        cbar.set_label('feedback strength', fontsize=cbar_fontsize)
        cbar.set_ticks(np.arange(n).astype('int') + 0.5)
        cbar.set_ticklabels(spec['e_gains'].values())
        cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        plt.tight_layout()
        return fig, ax

    spec = load_or_build_spec("S11c", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
fig, spec = figure_S11c()
save_plot(fig, "S11c", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=200, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %% [markdown]
# ## OUTPUT  

# %%
def figure_7b(load_folder, selected_run='14.gain.10', iteration='1', figsize=(16, 4), dpi=400,
                   plot_specs={'epochs_list': [1, 2, 5, 10, 20, 60], 'steps': 500, 'color_range': [0.1, 1]},
                   lims=[[-0.12, 0.12], [0.22, 0.44]], fontsize=13, targets_lw=6, targets_alpha=0.1,
                   excluded=[], trial_alpha=0.55, trial_lw=1.25, debug=False, gain=None,
                   selected_task='all', markersize=180, edge=2, contrast=1.1, plot_axes=False,
                   colorbar='each', title='', legend_position=(1.4, 0.7), plot_fb=True):
    def prepare():
        path = os.path.join(load_folder, selected_run)
        return {
            'figuretype': 'reaching_evolution',
            'meta': {
                'path': path, 'iteration': iteration, 'plot_specs': plot_specs, 'lims': lims,
                'fontsize': fontsize, 'targets_lw': targets_lw, 'targets_alpha': targets_alpha,
                'trial_alpha': trial_alpha, 'trial_lw': trial_lw, 'selected_task': selected_task,
                'gain': gain, 'plot_fb': plot_fb,
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
            legend_position=legend_position, plot_fb=meta['plot_fb'])
        if series:
            result = plot_reachings_evolution(meta['path'], meta['iteration'], replay=series, **kwargs)
        else:
            result = plot_reachings_evolution(meta['path'], meta['iteration'], capture=series, **kwargs)
        fig = result[0]
        return fig, fig.axes

    spec = load_or_build_spec("7b", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
fig, spec = figure_7b(load_folder)
save_plot(fig, "7b", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=400, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %% [markdown]
# ## Contributions

# %%
def figure_contributions(data_key, to_plot:  list, name, fontsize=12,
                                 lw=1.6, dpi=400, contribution=True, stop=2000, norm=True, start=1,
                                 selection=None, clip_values=False, split=False, figsize=(6, 2),
                                 ylims=[-0.1, 1.1], plot_fb=False, gain_values=False, cbar_title=None,
                                 legend_loc=[0.1, 0.1], wspace=0.1, invert_colors=False
    ):
    def prepare():
        data = get_dllocal_data()[data_key]
        loss_data = compute_lr_loss_contributions(data, to_plot)
        if gain_values:
            loss_data = {f"{float(k.split('.')[1]):.1E}": v for k, v in loss_data.items()}
        return {
            'figuretype': 'loss_contributions',
            'meta': {'to_plot': to_plot, 'stop': stop, 'start': start, 'norm': norm,
                      'contribution': contribution, 'split': split, 'plot_fb': plot_fb,
                      'fontsize': fontsize},
            'panels': [{'metric': 'contribution', 'series': []}],
            'loss_data': loss_data,
        }

    def render(spec):
        raw_data = spec.get('loss_data')
        if raw_data is not None:
            spec['panels'][0]['series'] = []
        series = spec['panels'][0]['series']
        if raw_data is not None:
            fig, ax = plot_loss_contributions(
                raw_data, fontsize=fontsize, lw=lw, dpi=dpi, contribution=contribution,
                stop=stop, norm=norm, start=start, selection=selection, clip_values=clip_values,
                split=split, figsize=figsize, ylims=ylims, plot_fb=plot_fb,invert_colors=invert_colors,
                legend_loc=legend_loc, wspace=wspace, cb_label=cbar_title, gain_values=gain_values,
                capture=series)
            del spec['loss_data']
        else:
            fig, ax = plot_loss_contributions(
                fontsize=fontsize, lw=lw, dpi=dpi, figsize=figsize, ylims=ylims, split=split,
                legend_loc=legend_loc, cb_label=cbar_title, gain_values=gain_values,
                replay=series)
        return fig, ax

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
fig, spec = figure_contributions(
    'gain_data', np.array(['gain.1e-1', 'gain.1', 'gain.2', 'gain.5']), name="7c",
    cbar_title= "feedback strength",
    stop=1000,
    figsize=(5,2),
    gain_values=True,
    plot_fb=True)

save_plot(fig, "7c", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=400, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %%
fig, spec = figure_contributions('lr_data', ['5e-4', '1e-3', '2e-3', '5e-3', '1e-2'], name="S11b", cbar_title= "consolidation rate", gain_values=False, invert_colors=True)
save_plot(fig, "S11b", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=400, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %%
fig, spec = figure_contributions('gain_data', ['gain.1e-1', 'gain.1', 'gain.2', 'gain.5', 'gain.10', 'gain.20'], name="S11d", cbar_title= "feedback strength", gain_values=True, plot_fb=True, stop=1000)

save_plot(fig, "S11d", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
          savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
          dpi=400, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)


# %%
