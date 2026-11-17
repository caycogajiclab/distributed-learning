# %%
"""Figure generation for the savings experiments."""
MACHINE = 'cerbero' # 'cerbero / frontex/ node41'

# %load_ext autoreload
# %autoreload 2


############################################
# SETTINGS
############################################

import os
from ccdl.paths import DATA_DIR, SOURCE_DATA_DIR, FIGURES_DIR, PROJECT_ROOT


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

load_folder = os.path.join(DATA_DIR, 'savings')
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

import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import Reds, Greens
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
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
# ## Methods

# %%


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


def get_continual_task_info(sim_folder):
    params = load_train_params(sim_folder, single_training=True)
    continual_task = params["continual_task"]
    epochs = [phase["epochs"] for phase in continual_task]
    angles = [phase["perturbation_params"]["angle"] for phase in continual_task]
    return epochs, angles


def load_savings_plot_data(
    data_folder,
    selected_names=None,
    metrics=('mse', 'nocb_error', 'cortex_error', 'changes'),
    normalize_losses=True,
    normalize_changes=False,
):
    data = {}
    epochs = None
    angles = None

    folder_names = sorted(os.listdir(data_folder))
    for folder_name in folder_names:
        sim_folder = os.path.join(data_folder, folder_name)
        if not os.path.isdir(sim_folder):
            continue
        if folder_name in ['configs', 'snapshot', 'submitted_jobs', 'scripts']:
            continue
        if folder_name.endswith('.json') or folder_name.endswith('.lock'):
            continue

        parts = folder_name.split('.')
        if not parts[0].isdigit():
            continue

        sim_name = '.'.join(parts[1:])
        if selected_names is not None and sim_name not in selected_names:
            continue

        try:
            params = load_train_params(sim_folder, single_training=True)
        except Exception:
            continue

        if epochs is None or angles is None:
            epochs, angles = get_continual_task_info(sim_folder)

        dataset_params = params['dataset_params']
        dataset_params['device'] = 'cpu'
        reaching = ReachingDataset(dataset_params)
        std = reaching.generate_unperturbed_dataset('cart')[1].std().item() if normalize_losses else 1.0

        iterations = {}
        for run_name in sorted(os.listdir(sim_folder), key=lambda x: int(x) if x.isdigit() else 10**9):
            if not run_name.isdigit():
                continue

            exp = Experiment2(sim_folder, run_name, load=True)

            continual_task = params["continual_task"]
            n_phases = len(continual_task)
            extended_metrics = {k: [] for k in metrics if k != 'changes'}
            if 'changes' in metrics:
                extended_metrics['changes'] = []

            if normalize_changes:
                pretrain_sd = load_var(os.path.join(sim_folder, 'pretrainings', run_name))
                W0 = pretrain_sd['rnn.h2h.weight'].abs().mean().item()
            else:
                W0 = 1.0

            for j in range(n_phases):
                train_output_j = getattr(exp, f'train_outputs_{j}')
                phase_metrics = train_output_j['metrics']

                if 'changes' in metrics:
                    changes = np.array(train_output_j['changes']['cortex.rnn.h2h.weight']) / W0
                    extended_metrics['changes'].append(changes)

                for metric_name in ['mse', 'nocb_error', 'cortex_error']:
                    if metric_name not in metrics or metric_name not in phase_metrics:
                        continue
                    vals = phase_metrics[metric_name]
                    if normalize_losses:
                        vals = [v / std for v in vals]
                    vals = vals if j == 0 else vals[1:]
                    extended_metrics[metric_name].append(vals)

            extended_metrics = {
                k: np.concatenate(v) if len(v) > 0 else np.array([])
                for k, v in extended_metrics.items()
            }
            iterations[run_name] = extended_metrics
            del exp

        data[sim_name] = iterations

    data['epochs'] = epochs
    data['angles'] = angles
    return data


def make_style_spec(cxbp_names, rnn_name, reds, title, full_model_label):
    spec = {'title': title}
    for idx, name in enumerate(cxbp_names):
        spec[name] = {
            'color': reds[4 - idx],
            'linestyle': 'solid',
            'zorder': 3,
            'marker': None,
            'label': full_model_label if '1e-06' in name else None,
        }
    spec[rnn_name] = {
        'color': 'black',
        'linestyle': 'dashed',
        'zorder': 2,
        'marker': None,
        'label': 'plastic controller',
    }
    return spec


def build_style_specs(variant_data, lr_colors, rnn_data, variant, full_model_label='full model'):
    style_specs = []
    lr_keys = sorted(variant_data.keys(), key=lambda x: float(x), reverse=True)
    main_lr = '1e-06' if '1e-06' in lr_keys else (lr_keys[-1] if lr_keys else None)
    label = full_model_label if variant == 'normal' else f'{full_model_label} (strong feedback)'

    for rnn_key in [k for k in rnn_data.keys() if k not in ('epochs', 'angles')]:
        spec = {'title': f'{variant}.{rnn_key}', 'rnn_key': rnn_key, 'variant': variant}
        for lr in lr_keys:
            spec[lr] = {
                'color': lr_colors[lr],
                'linestyle': 'solid',
                'zorder': 3,
                'marker': None,
                'label': label if lr == main_lr else None,
            }
        spec[rnn_key] = {
            'color': 'black',
            'linestyle': 'dashed',
            'zorder': 2,
            'marker': None,
            'label': 'plastic controller',
        }
        style_specs.append(spec)
    return style_specs


def plot_metrics_final(ax, data=None, epochs=None, style_dict=None, lw=1, metric='mse', fontsize=12,
                       start=0, stop=800, ms=2, markevery=100, logscale=True,
                       ylims=[1e-8, 1e-1], yticks=[1e-3, 1e-5, 1e-7], mean=True,
                       alpha=0.8, median_w=0, marker_on_nocb=False, errorbars=True,
                       capture=None, replay=None):

    if replay is not None:
        assert not errorbars, 'errorbar replay not supported (not captured; unused in this script, where errorbars=False throughout)'
        for entry in replay:
            x = np.array(entry['x'])
            y = np.array(entry['y'])
            if metric == 'nocb_error':
                ax.plot(x, y, c=entry['color'], linestyle=entry['linestyle'],
                        label=entry['label'], lw=lw, alpha=alpha, zorder=entry['zorder'])
            else:
                ax.plot(x, y, c=entry['color'], label=entry['label'], marker=entry['marker'],
                        markersize=ms, markevery=entry['markevery'],
                        markeredgewidth=entry['markeredgewidth'], markeredgecolor=entry['markeredgecolor'],
                        linestyle=entry['linestyle'], lw=lw, alpha=alpha, zorder=entry['zorder'])
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(*ylims)
            if yticks is not None:
                ax.set_yticks(yticks)
            if metric == 'changes':
                ax.set_ylim(1e-5, 2)
        by_label = {'model': Line2D([0], [0], color='gray', lw=lw, label='model'),
                 'plastic controller': Line2D([0], [0], color='gray', lw=lw, linestyle='dashed', label='plastic controller')}
        ax.legend(by_label.values(), by_label.keys(), fontsize=0.8*fontsize)
        if metric=='changes':
            ax.set_ylabel('total changes', fontsize=fontsize)
        else:
            ax.set_ylabel(f'training {metric}', fontsize=fontsize)
        return

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default matplotlib colors
    color_index = 0  # To cycle through default colors if style_dict is None
    
    for i, (key, val) in enumerate(data.items()):
        if style_dict and key not in style_dict:
            continue
        if key in ['epochs', 'angles']:
            continue
        
        if style_dict:
            style = style_dict[key]
            color = style['color']
            label = style['label']
            marker = style['marker']
            zorder = style['zorder']
            ls = style['linestyle']
        else:
            color = color_cycle[color_index % len(color_cycle)]
            label = key
            marker = None
            zorder = 0
            color_index += 1
            ls = 'solid'
        
        me =((i%6)*markevery//6+100,markevery) if marker else None
        mew = 0.2 if marker else None
        mec = 'k' if marker else None
        
        if metric in ['mse','changes']:
            i_data = np.array([val[k][metric] for k in val.keys()])
        else:
            if 'classical' not in key.lower() and 'RNN' not in key:
                i_data = np.array([val[k][metric] for k in val.keys()])
            else:
                i_data = np.array([val[k]['mse'] for k in val.keys()])
        
        if metric == 'changes':
            i_data = np.cumsum(i_data, axis=1)
            if mean:
                values = i_data.mean(axis=0)[start:stop]
            else:
                values = i_data[:,start:stop].T
        else:
            if median_w>0:
                i_data = moving_median(np.squeeze(metric), median_w)
            if mean:
                values = i_data.mean(axis=0)
            else:
                values = i_data.T
        
        if 'RNN' in key:
            # nocb_ls = (0, (8, 5))
            nocb_ls = 'dashed'
        else:
            nocb_ls = 'solid'
        start_epoch = 0
        offset = i *15
        idx = np.arange(offset, stop, markevery)
        for i in range(len(epochs)):
            x = np.arange(start_epoch, start_epoch+ epochs[i])  # Generate x for the current epoch range
            x = x[x>=start]
            if stop > 0:
                x = x[x<=stop]
            start_epoch += epochs[i]  # Update the starting epoch for the next line
            i_zorder = zorder+7-i
            if metric=='mse':
                ax.plot(x, 
                            values[x[0]:(x[-1]+1)], 
                            c=color, 
                            label=label, 
                            marker=marker, 
                            markersize=ms, 
                            markevery=me,
                            markeredgewidth=mew,
                            markeredgecolor=mec,
                            linestyle = ls,
                            lw=lw,
                            alpha=alpha,
                            zorder=i_zorder
                        )
                if capture is not None:
                    capture.append({
                        'name': f'{key}_epoch{i}', 'label': label, 'metric': metric,
                        'x': x.tolist(), 'y': values[x[0]:(x[-1]+1)].tolist(),
                        'color': color, 'marker': marker, 'linestyle': ls, 'zorder': i_zorder,
                        'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                    })
                if errorbars:
                    x = np.intersect1d(x, idx)
                    means = values[x]
                    mins = i_data.min(axis=0)[x]
                    maxs = i_data.max(axis=0)[x]

                    yerr = np.vstack((means - mins, maxs - means))
                    ax.errorbar(x, means, yerr=yerr, label=label,fmt='o',
                                marker='o', markersize=ms, lw=0.7*lw, linestyle=ls,
                                elinewidth=0.9, color=color,
                                markerfacecolor=color, markeredgewidth=0.9,
                                capsize=3, alpha=0.8, zorder=i_zorder+1)

            elif metric=='nocb_error':
                ax.plot(
                    x, 
                    values[x[0]:(x[-1]+1)], 
                    c=color, 
                    linestyle=nocb_ls, 
                    # marker=marker, 
                    # markersize=ms, 
                    # markevery=me,
                    # markeredgewidth=mew,
                    # markeredgecolor=mec,
                    label='controller memory' if i == 0 else None, 
                    lw=lw, 
                    alpha=alpha, 
                    zorder=i_zorder
                )
                if capture is not None:
                    capture.append({
                        'name': f'{key}_epoch{i}', 'label': label, 'metric': metric,
                        'x': x.tolist(), 'y': values[x[0]:(x[-1]+1)].tolist(),
                        'color': color, 'linestyle': nocb_ls, 'zorder': i_zorder,
                    })
                if errorbars:
                    x = np.intersect1d(x, idx)
                    means = values[x]
                    mins = i_data.min(axis=0)[x]
                    maxs = i_data.max(axis=0)[x]

                    yerr = np.vstack((means - mins, maxs - means))
                    ax.errorbar(x, means, yerr=yerr, label=label,fmt='o',
                                marker='o', markersize=ms, lw=lw, linestyle=ls,
                                elinewidth=0.9, color=color,
                                markerfacecolor=color, markeredgewidth=0.9,
                                capsize=3, alpha=1, zorder=i_zorder+1)
        if metric=='changes':
            x = np.arange(len(values))[start:stop]
            
            try:
                ax.plot(
                    x, 
                    values, 
                    c=color, 
                    marker=marker, 
                    markersize=ms, 
                    markevery=me,
                    markeredgewidth=mew,
                    markeredgecolor=mec,
                    linestyle=ls,
                    label='cumulative changes' if i == 0 else None,
                    lw=lw,
                    alpha=alpha,
                    zorder=zorder
                )
                if capture is not None:
                    capture.append({
                        'name': f'{key}_changes', 'label': label, 'metric': metric,
                        'x': x.tolist(), 'y': values.tolist(),
                        'color': color, 'marker': marker, 'linestyle': ls, 'zorder': zorder,
                        'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                    })
            except:
                ax.plot(
                    x[1:],
                    values,
                    c=color,
                    marker=marker,
                    markersize=ms,
                    markevery=me,
                    markeredgewidth=mew,
                    markeredgecolor=mec,
                    linestyle=ls,
                    label='cumulative changes' if i == 0 else None,
                    lw=lw,
                    alpha=alpha,
                    zorder=zorder
                )
                if capture is not None:
                    capture.append({
                        'name': f'{key}_changes', 'label': label, 'metric': metric,
                        'x': x[1:].tolist(), 'y': values.tolist(),
                        'color': color, 'marker': marker, 'linestyle': ls, 'zorder': zorder,
                        'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                    })
            if errorbars:
                x = np.intersect1d(x, idx)
                means = values[x]
                mins = i_data.min(axis=0)[x]
                maxs = i_data.max(axis=0)[x]
                
                yerr = np.vstack((means - mins, maxs - means))
                ax.errorbar(x, means, yerr=yerr, label=label,fmt='o',
                                marker='o', markersize=ms, lw=lw, linestyle=ls,
                                elinewidth=0.9, color=color, 
                                markerfacecolor=color, markeredgewidth=0.9,
                                capsize=3, alpha=0.8, zorder=i_zorder+1)
    
    if logscale:
        ax.set_yscale('log')
        ax.set_ylim(*ylims)
        if yticks is not None:
            ax.set_yticks(yticks)
        
        if metric=='changes':           
            ax.set_ylim(1e-5, 2)
    
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))  # Keep only one per label
    by_label={'model': Line2D([0], [0], color='gray', lw=lw, label='model'),
             'plastic controller': Line2D([0], [0], color='gray', lw=lw, linestyle='dashed', label='plastic controller')}
    
    # if metric=='nocb_error':
    #     from matplotlib.lines import Line2D
    #     by_label['full model'] = Line2D([0], [0], color='gray', lw=lw, label='full model')
    #     by_label['controller memory'] = Line2D([0], [0], color='gray', lw=lw, linestyle=nocb_ls, label='No-Adapter')
    
    ax.legend(by_label.values(), by_label.keys(), fontsize=0.8*fontsize)
    
    if metric=='changes':
        ax.set_ylabel('total changes', fontsize=fontsize)
    else:
        ax.set_ylabel(f'training {metric}', fontsize=fontsize)
        
# Function to customize spines and ticks
def customize_axes_final(ax, start, stop, epochs, fontsize,nticks=5, x_margin=100, grid=True, 
                   grid_color = (233/255, 233/255, 233/255, 0.97), sp_lw=0.7,
                   ax_color=[0.2]*3+[1], y_ticks=None, y_ticklabels=None, default_ticks=False,
                   metric='mse', area_coloring=True):
    last_tick = (stop//200)*200
    x_ticks = np.linspace(start, start+last_tick, nticks)
    defualt_y_ticks = [1e-6,1e-4,1e-2]
    y_ticks = defualt_y_ticks if not y_ticks else y_ticks
    default_y_ticklabels = [r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$']
    y_ticklabels = default_y_ticklabels if not y_ticklabels else y_ticklabels
    if grid:
        ax.grid(True, color=grid_color, linewidth=0.64,zorder=0)
        # ax.axhline(1e-4, color=grid_color, linewidth=0.5, zorder=0)
        # ax.axhline(1e-6, color=grid_color, linewidth=0.5, zorder=0)
    # ax.spines['top'].set(linewidth=0.72, color=grid_color, zorder=-1)
    ax.spines['right'].set_visible(False)
    # ax.spines['right'].set(linewidth=0.72, color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set(linewidth=sp_lw,color=ax_color)
    ax.spines['left'].set(linewidth=sp_lw,color=ax_color)
    ax.tick_params(axis='both', which='major', width=0.6, length=4.5, labelsize=fontsize,color=ax_color)
    ax.tick_params(axis='y', which='minor', width=0.4, length=3,color=ax_color)
    ax.set_xlim(-x_margin + start, stop+x_margin)
    ax.set_xlabel('epoch', fontsize=fontsize,color=ax_color)
    ax.set_ylabel('error (adaptation)', fontsize=fontsize,color=ax_color)
    ax.set_xticks(x_ticks,color=ax_color)
    ax.set_xticklabels(x_ticks.astype('int'),color=ax_color, fontname='Arial')
    if not default_ticks:
        ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_ticklabels, fontsize=fontsize)
    # if metric=='changes':
    #     ax.set_yticks([1, 1e-2,1e-4])
    if area_coloring:
        ax.axvspan(start-1, epochs[0], alpha=0.06, color='green', lw=0, zorder=0)
        ax.axvspan(epochs[0], sum(epochs[:2]), alpha=0.06, color='r', lw=0, zorder=0)
        ax.axvspan(sum(epochs[:2]), min(sum(epochs[:3]),stop)+1, alpha=0.06, color='green', lw=0, zorder=0)

def plot_angles(axs, n_epochs, angles, grid=None, capture=None, **kwargs):
    """
    Plots perturbation angles across epochs with customizable parameters.

    Parameters:
    - axs: Matplotlib Axes object where the plot will be drawn
    - n_epochs: List of integers, the number of epochs for each angle
    - angles: List of integers, the perturbation angles corresponding to each epoch range
    - kwargs: Additional keyword arguments for customization:
        - line_lw: Line width for the plot lines (default 2)
        - spine_lw: Line width for spines (default 2)
        - positive_color: Color for positive angles (default 'green')
        - negative_color: Color for negative angles (default 'red')
        - neutral_color: Color for angle 0 (default 'k')
        - xlabel: Label for x-axis (default 'Epochs')
        - ylabel: Label for y-axis (default 'Perturbation Angle (°)')
        - title: Title for the plot (default 'Perturbation Angles Across Epochs')
        - fontsize: Font size for axis labels and title (default 12)
    """
    # Extracting custom parameters with default values
    line_lw = kwargs.get('line_lw', 1.4)
    spine_lw = kwargs.get('spine_lw', 2)
    spine_color = kwargs.get('ax_color','k')
    positive_color = kwargs.get('positive_color', 'green')
    negative_color = kwargs.get('negative_color', 'red')
    neutral_color = kwargs.get('neutral_color', 'k')
    n_xticks = kwargs.get('nxticks',5)
    yticks = kwargs.get('yticks',None)
    n_yticks = kwargs.get('nxticks',7)
    xmargin = kwargs.get('xmargin',200)
    xlabel = kwargs.get('xlabel', 'epoch')
    ylim = kwargs.get('ylim',[-60,60])
    ylabel = kwargs.get('ylabel', 'rotation angle (°)')
    fill = kwargs.get('fill',True)
    title = kwargs.get('title', 'Perturbation Angles Across Epochs')
    fontsize = kwargs.get('fontsize', 12)
    stop = kwargs.get('stop', -1)
    start = kwargs.get('start',0)

    # Initialize the starting epoch
    start_epoch = start
    y_old=0
    for i in range(len(n_epochs)):
        if stop >0:
            if start_epoch > stop:
                break
            x = np.arange(start_epoch, min(start_epoch + n_epochs[i],stop))  # Generate x for the current epoch range
        else:
            x = np.arange(start_epoch, start_epoch + n_epochs[i])
        y = np.full_like(x, angles[i])  # Create a flat line at the current angle
        # Determine the color based on the angle value
        if angles[i] > 0:
            color = positive_color
        elif angles[i] < 0:
            color = negative_color
        else:
            color = neutral_color
        
        # Fill between the line and x-axis
        if fill:
            axs.fill_between(x, 0, y, color=color, alpha=0.1, edgecolor=None, zorder=2)  # Filling
        
        # Plot the line
        axs.plot(x, y, color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        if capture is not None:
            capture.append({'name': f'phase{i}', 'label': f'angle {angles[i]}', 'x': x.tolist(), 'y': y.tolist()})
        # axs.plot([x[0]-1,x[0]], [y_old,y[0]], color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        # y_old=y[-1]
        start_epoch += n_epochs[i]  # Update the starting epoch for the next line
    axs.plot([-xmargin,-1],[0,0], color='k', lw=line_lw, zorder=1)
    if capture is not None:
        capture.append({'name': 'baseline', 'label': 'baseline', 'x': [-xmargin, -1], 'y': [0, 0]})
    
    # Customize the plot style
    axs.set_xlabel(xlabel, fontsize=fontsize, color=spine_color)
    axs.set_ylabel(ylabel, fontsize=fontsize, color=spine_color)
    axs.set_title(title, fontsize=fontsize, color=spine_color)    

    # Set spines
    axs.spines['left'].set_color(spine_color)
    axs.spines['left'].set_linewidth(spine_lw)
    # axs.spines['bottom'].set_position(('data', 0))
    axs.spines['bottom'].set_color(spine_color)
    axs.spines['bottom'].set_linewidth(spine_lw)

    # Add arrow for the x-axis (DOES NOT WORK)
    # axs.annotate('', xy=(sum(n_epochs) + 1000, 0), xytext=(0, 0),
    #             arrowprops=dict(arrowstyle="->", lw=2, color='black', mutation_scale=20))

    # Set x-ticks (equally spaced)
    axs.set_xticks(np.linspace(0, sum(n_epochs), n_xticks))
    if not yticks:
        axs.set_yticks(np.linspace(-max_angle - 10, max_angle + 10, n_yticks))
        axs.set_yticklabels(np.linspace(-max_angle - 10, max_angle + 10, n_yticks),color=spine_color, fontsize=fontsize)
        axs.set_ylim([-max_angle - 10, max_angle + 10])
    else:
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticks,color=spine_color, fontsize=fontsize)
        axs.set_ylim([min(ylim),max(ylim)])
    axs.tick_params(axis='both', which='major', width=spine_lw, length=4.5, labelsize=fontsize, color=spine_color)
    axs.set_xticklabels([f'{int(i)}' for i in np.linspace(0, sum(n_epochs)-1, n_xticks)], color=spine_color, fontsize=fontsize)

    # Customize y-ticks and make y-limits symmetric
    max_angle = max(abs(a) for a in angles)  # Find max absolute angle value  # Adjust y-ticks
      # Symmetric y-limits

    # Clean up ticks and labels
    axs.tick_params(axis='both', which='major', labelsize=fontsize, color=spine_color)
    if grid:
        # axs.spines['right'].set_color(grid)
        # axs.spines['right'].set_linewidth(0.5*spine_lw)
        # axs.spines['top'].set_color(grid)
        # axs.spines['top'].set_linewidth(0.5*spine_lw)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        # axs.grid(True, color=grid, linewidth=0.64,zorder=0)



def setup_plot_final(add_changes, figsize=None, dpi=400, ratios=[20, 1], add_angles=True, add_nocb=True):
    rows = 1
    height_ratios = []
    if add_angles:
        rows +=1
        height_ratios.append(0.45)
    height_ratios.append(1)
    if add_nocb:
        rows +=1
        height_ratios.append(1)
    if add_changes:
        rows +=1
        height_ratios.append(1)
    figsize = (6.5, sum(height_ratios)*2.4) if figsize is None else figsize
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = GridSpec(rows, 2, width_ratios=ratios, height_ratios=height_ratios, figure=fig, hspace=0.1)
    axs = []
    index = 0
    if add_angles:
        axs.append(fig.add_subplot(gs[index, 0]))
        index +=1
    axs.append([fig.add_subplot(gs[index, 0]),fig.add_subplot(gs[index, 1])])
    index +=1
    if add_nocb:
        axs.append([fig.add_subplot(gs[index, 0]),fig.add_subplot(gs[index, 1])])
        index += 1
    if add_changes:
        axs.append([fig.add_subplot(gs[index, 0]),fig.add_subplot(gs[index, 1])])
    return fig, axs




# %%
# variant: normal = weak feedback, h-gain = strong feedback
# rnn_selection: which RNN style_spec (by rnn_key) to draw; the original always used 'plastic-out'
def figure_smithtask(panel_name, model_name, variant, rnn_key, plots_mode,
                      full_model_label,
                      fontsize=14, lw=1, ms=2, median_w=0, markevery=50, logscale=True,
                      ylims=[4e-7, 5e-2], mean=True, alpha=1, errorbars=False,
                      start=0, stop=800, x_margin=50, nticks=5, grid=False,
                      grid_color=(233/255, 233/255, 233/255, 0.97), ax_color='k', sp_lw=0.6,
                      add_angles=True, h_ratio=[20, 1], figsize=None, dpi=400):
    def prepare():
        savings_data = get_savings_data()
        epochs = savings_data['epochs']
        angles = savings_data['angles']
        dd_full = savings_data['mse_data']
        rnn_data = savings_data['rnn_data']
        if model_name == 'CxBP':
            variant_data = savings_data['cxbp_data'][variant]
            lr_colors = savings_data['lr_colors_cxbp']
        elif model_name == 'CxLocal':
            variant_data = savings_data['cxlocal_data'][variant]
            lr_colors = savings_data['lr_colors_cxlocal']

        style_specs = build_style_specs(variant_data, lr_colors, rnn_data, variant, full_model_label)
        style_specs = [spec for spec in style_specs if rnn_key == spec['rnn_key']]
        style_spec = style_specs[0]
        title = style_spec['title']

        plot_style = {}
        for key, style in style_spec.items():
            if key in ('title', 'rnn_key', 'variant'):
                continue
            if key == rnn_key:
                plot_style[f'RNN.{rnn_key}'] = style
            elif model_name == 'CxBP':
                if variant == 'normal':
                    plot_style[f'CxBP.cx_lr.{key}'] = style
                else:
                    plot_style[f'CxBP.h-gain.cx_lr.{key}'] = style
            elif model_name == 'CxLocal':
                if variant == 'normal':
                    plot_style[f'CxLocal.local_lr.{key}'] = style
                else:
                    plot_style[f'CxLocal.h-gain.local_lr.{key}'] = style

        used_start = start if plots_mode == 'compact' else 0
        used_x_margin = x_margin if plots_mode == 'compact' else 40
        ticks_start = start if plots_mode == 'compact' else 0

        if plots_mode == 'full':
            plot_changes = True
            add_nocb = True
        else:
            plot_changes = False
            add_nocb = False

        metrics = ['mse'] + (['nocb_error'] if add_nocb else [])

        ordered_lrs = sorted(lr_colors.keys(), key=lambda x: float(x), reverse=True)
        colors = {lr: lr_colors[lr] for lr in ordered_lrs}

        panels = []
        if add_angles:
            panels.append({'metric': 'angles', 'series': []})
        for metric in metrics:
            panels.append({'metric': metric, 'series': []})
        if plot_changes:
            panels.append({'metric': 'changes', 'series': []})

        return {
            'figuretype': 'smithtask',
            'meta': {
                'title': title, 'model_name': model_name, 'variant': variant, 'rnn_key': rnn_key,
                'plots_mode': plots_mode, 'used_start': used_start, 'used_x_margin': used_x_margin,
                'ticks_start': ticks_start, 'plot_changes': plot_changes, 'metrics': metrics,
                'fontsize': fontsize, 'lw': lw, 'ms': ms, 'median_w': median_w, 'markevery': markevery,
                'logscale': logscale, 'ylims': ylims, 'mean': mean, 'alpha': alpha, 'errorbars': errorbars,
                'stop': stop, 'nticks': nticks, 'grid': grid, 'grid_color': grid_color, 'ax_color': ax_color,
                'sp_lw': sp_lw, 'add_angles': add_angles, 'h_ratio': h_ratio, 'figsize': figsize, 'dpi': dpi,
                'ordered_lrs': ordered_lrs,
            },
            'plot_style': plot_style, 'dd_full': dd_full, 'epochs': epochs, 'angles': angles, 'colors': colors,
            'panels': panels,
        }

    def render(spec):
        meta = spec['meta']
        epochs = spec['epochs']
        angles = spec['angles']
        colors = spec['colors']
        dd_full = spec.get('dd_full')
        plot_style = spec.get('plot_style')
        building = dd_full is not None
        for panel in spec['panels']:
            if building or panel['metric'] == 'angles':
                panel['series'] = []
        panel_by_metric = {panel['metric']: panel['series'] for panel in spec['panels']}

        fig, axs = setup_plot_final(
            meta['plot_changes'],
            figsize=meta['figsize'],
            dpi=meta['dpi'],
            add_angles=meta['add_angles'],
            ratios=meta['h_ratio'],
            add_nocb='nocb_error' in meta['metrics'],
        )

        gray = (50/255, 50/255, 50/255, 0.9)
        index = 0

        if meta['add_angles']:
            ax = axs[index]
            plot_angles(
                ax, epochs, angles,
                fontsize=meta['fontsize'], yticks=[-40, 0, 40], ylim=[-60, 60],
                title='', grid=meta['grid_color'], spine_lw=meta['sp_lw'], xmargin=40,
                positive_color='C2', negative_color='C3', neutral_color=gray,
                fill=True, ax_color=meta['ax_color'], stop=meta['stop'],
                capture=panel_by_metric['angles']
            )
            x_ticks = np.linspace(meta['ticks_start'], 800, meta['nticks']).astype(int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
            ax.set_xlim(meta['used_start'] - meta['used_x_margin'], meta['stop'] + meta['used_x_margin'])
            index += 1

        cbar_axs = []
        for metric in meta['metrics']:
            ax = axs[index][0]
            ax.set_title('controller memory' if metric == 'nocb_error' else 'full model', fontsize=meta['fontsize'])

            plot_metrics_final(
                ax, dd_full, epochs, plot_style, meta['lw'], metric, meta['fontsize'],
                meta['used_start'], meta['stop'], ms=meta['ms'], median_w=meta['median_w'],
                markevery=meta['markevery'], logscale=meta['logscale'], ylims=meta['ylims'], yticks=None,
                mean=meta['mean'], alpha=meta['alpha'], errorbars=meta['errorbars'],
                capture=panel_by_metric[metric] if building else None,
                replay=None if building else panel_by_metric[metric],
            )
            customize_axes_final(
                ax, meta['used_start'], meta['stop'], epochs, meta['fontsize'],
                x_margin=2 * meta['used_x_margin'], nticks=meta['nticks'],
                grid=meta['grid'], grid_color=meta['grid_color'], ax_color=meta['ax_color'],
                area_coloring=True, y_ticks=None, y_ticklabels=None, sp_lw=meta['sp_lw']
            )
            ax.set_xlim(meta['used_start'] - meta['used_x_margin'], meta['stop'] + meta['used_x_margin'])
            x_ticks = np.linspace(meta['ticks_start'], 800, meta['nticks']).astype(int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
            cbar_axs.append(axs[index][1])
            index += 1

        if meta['plot_changes']:
            ax = axs[index][0]
            ax.set_title('recurrent layer changes', fontsize=meta['fontsize'])
            plot_metrics_final(
                ax, dd_full, epochs, plot_style, meta['lw'],
                'changes', meta['fontsize'], 0, meta['stop'], ms=meta['ms'], median_w=meta['median_w'],
                markevery=meta['markevery'], logscale=False, ylims=meta['ylims'], yticks=None,
                mean=meta['mean'], alpha=meta['alpha'], errorbars=meta['errorbars'],
                capture=panel_by_metric['changes'] if building else None,
                replay=None if building else panel_by_metric['changes'],
            )
            customize_axes_final(
                ax, meta['used_start'], meta['stop'], epochs, meta['fontsize'],
                x_margin=2 * meta['used_x_margin'], nticks=meta['nticks'],
                grid=meta['grid'], grid_color=meta['grid_color'], ax_color=meta['ax_color'],
                area_coloring=True, y_ticks=None, y_ticklabels=None,
                sp_lw=meta['sp_lw'], default_ticks=True
            )
            x_ticks = np.linspace(meta['ticks_start'], 800, meta['nticks']).astype(int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
            ax.set_xlim(meta['used_start'] - meta['used_x_margin'], meta['stop'] + meta['used_x_margin'])
            ax.set_yticks([0,0.02,0.04, 0.06])
            ax.set_yticklabels(['0', '2', '4', '6'])
            ax.set_ylabel('cumul. % changes')
            cbar_axs.append(axs[index][1])

        for cax in cbar_axs:
            add_cbar(cax, meta['ordered_lrs'], colors, fontsize=meta['fontsize'])

        if building:
            del spec['dd_full']
            del spec['plot_style']

        return fig, axs

    spec = load_or_build_spec(panel_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %% [markdown]
# ## Data Loading

# %%
_savings_data_loaded = None

def get_savings_data():
    global _savings_data_loaded
    if _savings_data_loaded is None:
        data_folder = os.path.join(DATA_DIR, "savings")

        mse_data = load_continual_metrics(
            data_folder,
            UPDATE_METRICS,
            pretrain_folder,
            normalize_changes=True,
            metrics=['mse', 'nocb_error', 'cortex_error','changes']
        )

        epochs = mse_data['epochs']
        angles = mse_data['angles']

        # RNN: all keys containing "RNN", simplified by stripping the "RNN." prefix
        rnn_data = {
            k.replace('RNN.', ''): v
            for k, v in mse_data.items()
            if 'RNN' in k
        }
        rnn_data['epochs'] = epochs
        rnn_data['angles'] = angles

        # CxBP: split into h-gain vs normal, keys simplified to lr only
        cxbp_data = {
            'normal': {
                k.replace('CxBP.cx_lr.', ''): v
                for k, v in mse_data.items()
                if k.startswith('CxBP.') and 'h-gain' not in k
            },
            'h-gain': {
                k.replace('CxBP.h-gain.cx_lr.', ''): v
                for k, v in mse_data.items()
                if k.startswith('CxBP.') and 'h-gain' in k
            },
        }

        # CxLocal: split into h-gain vs normal, keys simplified to lr only
        cxlocal_data = {
            'normal': {
                k.replace('CxLocal.local_lr.', ''): v
                for k, v in mse_data.items()
                if k.startswith('CxLocal.') and 'h-gain' not in k
            },
            'h-gain': {
                k.replace('CxLocal.h-gain.local_lr.', ''): v
                for k, v in mse_data.items()
                if k.startswith('CxLocal.') and 'h-gain' in k
            },
        }

        # Fixed LR -> color maps
        # darker = smaller lr, lighter = larger lr
        N = 5
        color_values = [0.9 - 0.6 * i / (N - 1) for i in range(N)]
        reds = {i: Reds(color_values[i]) for i in range(N)}
        greens = {i: Greens(color_values[i]) for i in range(N)}

        cxbp_lrs = sorted(cxbp_data['normal'].keys(), key=lambda x: float(x), reverse=True)
        cxlocal_lrs = sorted(cxlocal_data['normal'].keys(), key=lambda x: float(x), reverse=True)

        lr_colors_cxbp = {lr: reds[4 - i] for i, lr in enumerate(cxbp_lrs)}
        lr_colors_cxlocal = {lr: greens[4 - i] for i, lr in enumerate(cxlocal_lrs)}

        _savings_data_loaded = {
            'mse_data': mse_data,
            'epochs': epochs,
            'angles': angles,
            'rnn_data': rnn_data,
            'cxbp_data': cxbp_data,
            'cxlocal_data': cxlocal_data,
            'lr_colors_cxbp': lr_colors_cxbp,
            'lr_colors_cxlocal': lr_colors_cxlocal,
        }
    return _savings_data_loaded


# %% [markdown]
# ## Plot parameters

# %%
mean = True
median_w = 0
ms = 2
plot_nocb_error = False
logscale = True
alpha = 1
nticks = 5
grid = False
markevery = 50
errorbars = False

min_distance = 1000
grid_color = (233/255, 233/255, 233/255, 0.97)
ax_color = 'k'
add_angles = True
lw = 1

figsize = None
fontsize = 14
ylims = [4e-7, 5e-2]
dpi = 400
h_ratio = [20, 1]
x_margin = 50
start, stop = 0, 800
sp_lw = 0.6

modes = ['compact']
modes = ['compact', 'full']

# %% [markdown]
# ## CxBP

# %%
panels = {
    '4e': ('compact', 'h-gain'),
    'S8a': ('full', 'normal'),
    'S8b': ('full', 'h-gain'), 
}

for panel_name, (plots_mode, variant) in panels.items():
    pr_print(['bold','red'],f"panel: {panel_name} (variant: {variant})")
    fig, spec = figure_smithtask(
        panel_name=panel_name, model_name='CxBP', variant=variant, rnn_key='plastic-out', plots_mode=plots_mode,
        full_model_label='full model',
        fontsize=fontsize, lw=lw, ms=ms, median_w=median_w, markevery=markevery, logscale=logscale,
        ylims=ylims, mean=mean, alpha=alpha, errorbars=errorbars, start=start, stop=stop,
        x_margin=x_margin, nticks=nticks, grid=grid, grid_color=grid_color, ax_color=ax_color,
        sp_lw=sp_lw, add_angles=add_angles, h_ratio=h_ratio, figsize=figsize, dpi=dpi,
    )
    save_plot(fig, panel_name, plots_folder, savepdf=save_pdf, savetiff=save_tiff, savedata=save_data,
              data=spec, plotsdatafolder=plots_data_folder, dpi=dpi, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## CxLocal

# %%
panels = {
    'S12a': ('full', 'normal'),
    'S12b': ('full', 'h-gain'), 
}

for panel_name, (plots_mode, variant) in panels.items():
    pr_print(['bold','red'],f"panel: {panel_name} (variant: {variant})")
    fig, spec = figure_smithtask(
        panel_name=panel_name, model_name='CxLocal', variant=variant, rnn_key='plastic-out', plots_mode=plots_mode,
        full_model_label='full model',
        fontsize=fontsize, lw=lw, ms=ms, median_w=median_w, markevery=markevery, logscale=logscale,
        ylims=ylims, mean=mean, alpha=alpha, errorbars=errorbars, start=start, stop=stop,
        x_margin=x_margin, nticks=nticks, grid=grid, grid_color=grid_color, ax_color=ax_color,
        sp_lw=sp_lw, add_angles=add_angles, h_ratio=h_ratio, figsize=figsize, dpi=dpi,
    )
    save_plot(fig, panel_name, plots_folder, savepdf=save_pdf, savetiff=save_tiff, savedata=save_data,
              data=spec, plotsdatafolder=plots_data_folder, dpi=dpi, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %%
