# %%
# %load_ext autoreload
# %autoreload 2

"""Figure generation for the continual-learning experiments."""


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

load_folder = os.path.join(WORKDIR, 'continual')
pretrainings_folder = os.path.join(WORKDIR, "pretrainings", "cortex")
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
from plot_utils import save_plot as save_plot_shared  # a local save_plot() is defined below, shadowing the import

notebook_header(HOME, display_width=70, font_family="Arial", seed=124)

import pandas as pd
import torch
import numpy as np
from copy import deepcopy
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
# # Methods

# %%
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

def build_continual_styles(data, lr_colors):
    styles = {}
    for group, cmap in [('CxBP', Reds), ('CxBP-hgain', Reds), ('CxLocal', Greens), ('CxLocal-hgain', Greens)]:
        names = [k for k in data.keys() if group.split('-')[0] in k and ('h-gain' in k) == ('hgain' in group)]
        lrs = [k.split('cx_lr.')[-1] if 'cx_lr.' in k else k.split('local_lr.')[-1] for k in names]
        
        for name in names:
            lr = name.split('cx_lr.')[-1] if 'cx_lr.' in name else name.split('local_lr.')[-1]
            styles[name] = {'color': lr_colors.get(lr, 'k'), 'label': lr, 'linestyle': 'solid', 'marker': 'o', 'zorder': 3}
    for name in [k for k in data.keys() if 'RNN' in k]:
        styles[name] = {'color': 'k', 'label': 'plastic controller', 'linestyle': 'dashed', 'marker': 'o', 'zorder': 2}
    return styles


# Function to set up the plot
def setup_plot(plot_changes, figsize=(16, 10), dpi=400, ratios=[3,1], share_x=True):
    # Set up GridSpec for consistent legend space
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    if plot_changes:
        # 2x2 layout: 2 rows for plots, 1 column for legend
        gs = GridSpec(3, 2, width_ratios=ratios, height_ratios=[0.65, 1, 0.8], figure=fig, hspace=0.2)

        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
    else:
        # 1x2 layout: 1 row, 1 plot, 1 legend
        gs = GridSpec(2, 2, width_ratios=ratios, figure=fig)

        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]

    # Reserve the right column for the legend
    # legend_ax = fig.add_subplot(gs[1, 1])
    # legend_ax.axis("off")  # Turn off axes for the legend placeholder

    return fig, axs

# Function to customize spines and ticks
def customize_axes(axs, fontsize,nticks=6, x_margin=100, grid=True, grid_color = (233/255, 233/255, 233/255, 0.97), ax_color=[0.2]*3+[1], y_ticks=None, y_ticklabels=None, start=0,stop=10000):
    for k, ax in enumerate(axs):
        x_ticks = np.linspace(start, stop, nticks)
        default_y_ticks = [1e-7,1e-5,1e-3] if k==0 else [0,0.1,0.2]
        used_y_ticks = y_ticks if (y_ticks is not None and k==0) else default_y_ticks
        default_y_ticklabels = [r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$']if k==0 else ['0%','10%','20%']
        used_y_ticklabels = y_ticklabels if (y_ticklabels is not None and k==0) else default_y_ticklabels
        if grid:
            ax.grid(True, color=grid_color, linewidth=0.64,zorder=0)
            # ax.axhline(1e-4, color=grid_color, linewidth=0.5, zorder=0)
            # ax.axhline(1e-6, color=grid_color, linewidth=0.5, zorder=0)
        ax.spines['top'].set(linewidth=0.72, color=grid_color, zorder=-1)
        ax.spines['right'].set(linewidth=0.72, color=grid_color)
        ax.spines['bottom'].set(linewidth=0.7,color=ax_color)
        ax.spines['left'].set(linewidth=0.7,color=ax_color)
        ax.tick_params(axis='both', which='major', width=0.72, length=4.5, labelsize=fontsize,color=ax_color)
        ax.tick_params(axis='y', which='minor', width=0.5, length=2.5,color=ax_color)
        ax.set_xlim(-x_margin + start, stop+x_margin)
        ax.set_xlabel('epoch', fontsize=fontsize,color=ax_color)
        ax.set_ylabel('error (adaptation)', fontsize=fontsize,color=ax_color)
        ax.set_xticks(x_ticks,color=ax_color)
        ax.set_xticklabels(x_ticks.astype('int'),color=ax_color, fontname='Arial')
        ax.set_yticks(used_y_ticks,color=ax_color)
        ax.set_yticklabels(used_y_ticklabels,color=ax_color, fontname='Arial')


def annotate_phases(axs, angles, epochs, fontsize, start, stop, 
                    positions=None, lw=1.5, print_angles=True):
    epochs_cum = np.cumsum(epochs)
    temp_epochs = np.insert(epochs_cum,0,0)
    index = (temp_epochs>=start) & (temp_epochs<=stop)
    temp_epochs = temp_epochs[index]
    angles = [angles[i] for i in range(len(index)-1) if index[i]]
    
    for j, ax in enumerate(axs):
        counter = -20
        for angle, phase_id in zip(angles, range(len(angles))):
            pos = counter + epochs[phase_id] / 3 if not positions else positions[phase_id]
            y = 1e-2 if j == 0 else 0.9
            if pos >= start and pos<=stop and print_angles:
                ax.annotate(f'{angle}°', xy=(pos, y), fontsize=fontsize)
            if (phase_id+1)<len(temp_epochs):
                ax.axvline(temp_epochs[phase_id+1], color='k', linestyle='dashed', linewidth=0.5*lw,zorder=2, dashes=[4,4])
                # ax.annotate(f'{epochs_cum[phase_id]}', xy=(epochs_cum[phase_id], 0), fontsize=0.6*fontsize)
            counter += epochs[phase_id]


def plot_metrics(axs, data=None, style_dict=None, lw=1, plot_nocb_error=False, plot_changes=False, fontsize=12, start=0, stop=800, ms=30, markevery=100, metric_name='mse', logscale=True, ylims=[1e-8, 1e-1], yticks=[1e-3, 1e-5, 1e-7], mean=True, alpha=0.8, median_w=0, legend=False, capture_mse=None, capture_changes=None, replay_mse=None, replay_changes=None):

    nocb_ls = (0, (8, 5))

    if replay_mse is not None:
        for entry in replay_mse:
            x = np.array(entry['x'])
            y = np.array(entry['y'])
            entry_lw = 0.8 * lw if entry['metric'] == 'nocb_error' else lw
            entry_alpha = 0.75 * alpha if entry['metric'] == 'nocb_error' else alpha
            axs[0].plot(x, y, c=entry['color'], label=entry['label'], marker=entry['marker'],
                        markersize=ms, markevery=entry['markevery'],
                        markeredgewidth=entry['markeredgewidth'], markeredgecolor=entry['markeredgecolor'],
                        lw=entry_lw, ls=entry['linestyle'], alpha=entry_alpha, zorder=entry['zorder'])
        for entry in (replay_changes or []):
            x = np.array(entry['x'])
            y = np.array(entry['y'])
            axs[1].plot(x, y, c=entry['color'], marker=entry['marker'], markersize=ms,
                        markevery=entry['markevery'], markeredgewidth=entry['markeredgewidth'],
                        markeredgecolor=entry['markeredgecolor'], label=entry['label'], lw=lw,
                        ls=entry['linestyle'], alpha=alpha, zorder=entry['zorder'])

        if logscale:
            axs[0].set_yscale('log')
            axs[0].set_ylim(*ylims)
            axs[0].set_yticks(yticks)

        handles, labels = axs[0].get_legend_handles_labels()

        if plot_nocb_error:
            from matplotlib.lines import Line2D
            handles.append(Line2D([0], [0], color='gray', lw=lw, label='Full Model'))
            labels.append('Full Model')
            handles.append(Line2D([0], [0], color='gray', lw=lw, linestyle=nocb_ls, label='No-Adapter'))
            labels.append('No-Adapter')

        if legend:
            axs[0].legend(handles, labels, fontsize=fontsize)

        if plot_changes:
            axs[1].set_ylabel('Total Changes', fontsize=fontsize)
        else:
            axs[0].set_ylabel(f'Training {metric_name}', fontsize=fontsize)
        return

    for i, (key, val) in enumerate(data.items()):
        if key not in style_dict:
            continue
        
        ls = 'solid' if 'RNN' not in key else 'dashed'
        color = style_dict[key]
        label = key
        marker = None
        zorder = 2 if 'RNN' in key else 3
        
        me =((i%4)*markevery//4+100,markevery) if marker else None
        mew = 0.2 if marker else None
        mec = 'k' if marker else None
        
        metric = np.array([val[k][metric_name] for k in val.keys()])
        if mean:
            metric = metric.mean(axis=0)[start:stop]
        if median_w>0:
            metric = moving_median(np.squeeze(metric), median_w)
        
        x = np.arange(start, len(metric) + start)
        # t_lw = lw if 'RNN' not in key else lw
        axs[0].plot(x, 
                    metric, 
                    c=color, 
                    label=label, 
                    marker=marker, 
                    markersize=ms, 
                    markevery=me,
                    markeredgewidth=mew,
                    markeredgecolor=mec,
                    lw=lw, 
                    ls = ls,
                    alpha=alpha,
                    zorder=zorder+7-i
                )
        if capture_mse is not None:
            capture_mse.append({
                'name': key, 'x': x.tolist(), 'y': np.asarray(metric).tolist(),
                'label': label, 'metric': metric_name, 'color': color, 'marker': marker,
                'linestyle': ls, 'zorder': zorder + 7 - i,
                'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
            })

        if plot_nocb_error and 'Classical' not in key and 'RNN' not in key:
            nocb_error = np.array([val[k]['nocb_error'] for k in val.keys()])
            if mean:
                nocb_error = nocb_error.mean(axis=0)[start:stop]
            if nocb_error is not None:
                axs[0].plot(
                    x, 
                    nocb_error, 
                    c=color, 
                    linestyle=nocb_ls, 
                    marker=marker, 
                    markersize=ms, 
                    markevery=me,
                    markeredgewidth=mew,
                    markeredgecolor=mec,
                    label='No-Adapter Error' if i == 0 else None,
                    lw=0.8*lw,
                    alpha=0.75*alpha,
                    zorder=zorder-1
                )
                if capture_mse is not None:
                    capture_mse.append({
                        'name': f'{key}_nocb_error', 'x': x.tolist(), 'y': np.asarray(nocb_error).tolist(),
                        'label': ('No-Adapter Error' if i == 0 else None), 'metric': 'nocb_error',
                        'color': color, 'marker': marker, 'linestyle': nocb_ls, 'zorder': zorder - 1,
                        'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                    })

        if plot_changes:
            changes = np.array([val[k]['changes'] for k in val.keys()])
            if changes is not None:
                changes = np.cumsum(changes, axis=1)
                if mean:
                    changes = changes.mean(axis=0)[start:stop]
                try:
                    axs[1].plot(
                        x, 
                        changes, 
                        c=color, 
                        marker=marker, 
                        markersize=ms, 
                        markevery=me,
                        markeredgewidth=mew,
                        markeredgecolor=mec,
                        label='Cumulative Changes' if i == 0 else None,
                        lw=lw,
                        ls = ls,
                        alpha=alpha,
                        zorder=zorder
                    )
                    if capture_changes is not None:
                        capture_changes.append({
                            'name': key, 'x': x.tolist(), 'y': np.asarray(changes).tolist(),
                            'label': ('Cumulative Changes' if i == 0 else None),
                            'color': color, 'marker': marker, 'linestyle': ls, 'zorder': zorder,
                            'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                        })
                except:
                    axs[1].plot(
                        x[1:],
                        changes,
                        c=color,
                        marker=marker,
                        markersize=ms,
                        markevery=me,
                        markeredgewidth=mew,
                        markeredgecolor=mec,
                        label='Cumulative Changes' if i == 0 else None,
                        lw=lw,
                        ls = ls,
                        alpha=alpha,
                        zorder=zorder
                    )
                    if capture_changes is not None:
                        capture_changes.append({
                            'name': key, 'x': x[1:].tolist(), 'y': np.asarray(changes).tolist(),
                            'label': ('Cumulative Changes' if i == 0 else None),
                            'color': color, 'marker': marker, 'linestyle': ls, 'zorder': zorder,
                            'markevery': me, 'markeredgewidth': mew, 'markeredgecolor': mec,
                        })

    if logscale:
        axs[0].set_yscale('log')
        axs[0].set_ylim(*ylims)
        axs[0].set_yticks(yticks)
        
    handles, labels = axs[0].get_legend_handles_labels()
    
    if plot_nocb_error:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='gray', lw=lw, label='Full Model'))
        labels.append('Full Model')
        handles.append(Line2D([0], [0], color='gray', lw=lw, linestyle=nocb_ls, label='No-Adapter'))
        labels.append('No-Adapter')
    
    if legend:
        axs[0].legend(handles, labels, fontsize=fontsize)
    
    if plot_changes:
        axs[1].set_ylabel('Total Changes', fontsize=fontsize)
    else:
        axs[0].set_ylabel(f'Training {metric_name}', fontsize=fontsize)

    
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
    n_xticks = kwargs.get('nxticks',6)
    yticks = kwargs.get('yticks',None)
    n_yticks = kwargs.get('nxticks',7)
    xmargin = kwargs.get('xlims',200)
    xlabel = kwargs.get('xlabel', 'epochs')
    ylabel = kwargs.get('ylabel', 'rotation angle (°)')
    fill = kwargs.get('fill',True)
    title = kwargs.get('title', 'Perturbation Angles Across Epochs')
    start = kwargs.get('start',0)
    stop = kwargs.get('stop',np.sum(n_epochs))
    fontsize = kwargs.get('fontsize', 12)

    y_old=0
    start_epoch = 0
    for i in range(len(n_epochs)):
        if start_epoch >=stop:
            break
        x = np.arange(start_epoch, min(start_epoch + n_epochs[i],stop))  # Generate x for the current epoch range
        y = np.full_like(x, angles[i])[:len(x)]  # Create a flat line at the current angle
        
        # Determine the color based on the angle value
        if angles[i] > 0:
            color = positive_color
        elif angles[i] < 0:
            color = negative_color
        else:
            color = neutral_color
        
        # Fill between the line and x-axis
        if fill:
            axs.fill_between(x, 0, y, color=color, alpha=0.3, edgecolor=None, zorder=2)  # Filling
        
        # Plot the line
        axs.plot(x, y, color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        if capture is not None:
            capture.append({'name': f'phase{i}', 'x': x.tolist(), 'y': y.tolist()})
        # axs.plot([x[0]-1,x[0]], [y_old,y[0]], color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        # y_old=y[-1]
        start_epoch += n_epochs[i]  # Update the starting epoch for the next line

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
    axs.set_xticks(np.linspace(start, stop, n_xticks))
    if not yticks:
        axs.set_yticks(np.linspace(-max_angle - 10, max_angle + 10, n_yticks))
        axs.set_yticklabels(np.linspace(-max_angle - 10, max_angle + 10, n_yticks),color=spine_color, fontsize=fontsize)
        axs.set_ylim([-max_angle - 10, max_angle + 10])
    else:
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticks,color=spine_color, fontsize=fontsize)
        axs.set_ylim([min(yticks),max(yticks)])
    axs.tick_params(axis='both', which='major', width=spine_lw, length=4.5, labelsize=fontsize, color=spine_color)
    axs.set_xticklabels([f'{int(i)}' for i in np.linspace(start, stop, n_xticks)], color=spine_color, fontsize=fontsize)
    axs.set_xlim(-xmargin+start, stop+xmargin)

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
        axs.grid(True, color=grid, linewidth=0.64,zorder=0)
        
def plot_angles_mini(axs, n_epochs, angles, grid=None, capture=None, **kwargs):
    """
    Plots perturbation angles across epochs with customizable parameters.

    Parameters:
    - axs: Matplotlib Axes object where the plot will be drawn
    - n_epochs: List of integers, the number of epochs for each angle (cumulative)
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
    n_xticks = kwargs.get('nxticks',len(n_epochs))
    yticks = kwargs.get('yticks',None)
    n_yticks = kwargs.get('nxticks',7)
    xmargin = kwargs.get('xlims',200)
    xlabel = kwargs.get('xlabel', 'epoch')
    ylabel = kwargs.get('ylabel', 'rotation angle (°)')
    fill = kwargs.get('fill',True)
    title = kwargs.get('title', 'Perturbation Angles Across Epochs')
    start = kwargs.get('start',0)
    stop = kwargs.get('stop',n_epochs[-1])
    phasex = kwargs.get('phasex',False)
    fontsize = kwargs.get('fontsize', 12)

    y_old=0
    start_epoch = 0
    for i in range(len(n_epochs)):
        x = np.arange(start_epoch, min(n_epochs[i],stop))  # Generate x for the current epoch range
        y = np.full_like(x, angles[i])[:len(x)]  # Create a flat line at the current angle
        
        # Determine the color based on the angle value
        if angles[i] > 0:
            color = positive_color
        elif angles[i] < 0:
            color = negative_color
        else:
            color = neutral_color
        
        # Fill between the line and x-axis
        if fill:
            axs.fill_between(x, 0, y, color=color, alpha=0.3, edgecolor=None, zorder=2)  # Filling
        
        # Plot the line
        axs.plot(x, y, color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        if capture is not None:
            capture.append({'name': f'phase{i}', 'x': x.tolist(), 'y': y.tolist()})
        # axs.plot([x[0]-1,x[0]], [y_old,y[0]], color=color, lw=line_lw, zorder=3)  # Plotting the flat line
        # y_old=y[-1]
        start_epoch = n_epochs[i]  # Update the starting epoch for the next line

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
    if phasex:
        rel_pos = n_epochs[0]//2
        axs.set_xticks(np.arange(rel_pos, n_epochs[-2]+rel_pos, n_epochs[0]-1))
        axs.set_xticklabels(np.arange(1,len(n_epochs)+1))
    # else:
    #     axs.set_xticks(np.linspace(start, stop, n_xticks))
    
    if not yticks:
        axs.set_yticks(np.linspace(-max_angle - 10, max_angle + 10, n_yticks))
        axs.set_yticklabels(np.linspace(-max_angle - 10, max_angle + 10, n_yticks),color=spine_color, fontsize=fontsize)
        axs.set_ylim([-max_angle - 10, max_angle + 10])
    else:
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticks,color=spine_color, fontsize=fontsize)
        axs.set_ylim([min(yticks),max(yticks)])
    axs.tick_params(axis='both', which='major', width=spine_lw, length=4.5, labelsize=fontsize, color=spine_color)
    # axs.set_xticklabels([f'{int(i)}' for i in np.linspace(start, stop, n_xticks)], color=spine_color, fontsize=fontsize)
    axs.set_xlim(-xmargin+start, stop+xmargin)

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
        axs.grid(True, color=grid, linewidth=0.64,zorder=0)

def per_phase_loss(data,epochs, angles, mode ='same',val=None, selection=None): # either same or last (I assue all chunks have same size but last one)
    d_data = deepcopy(data)
    metrics = ['mse','nocb_error','changes']
    converted_data = {}
    assert isinstance(selection,list) or isinstance(selection,type(None)), "Wrong selection passed"  # (old version) assert(isinstance(selection,list) or isinstance(selection,type(None)),"Wrong selection passed")
    epochs_list = []
    epoch_flag = True
    for model, dd in d_data.items():
        if selection is not None:
            if model not in selection:
                continue
        model_data = {i: {metric: [] for metric in metrics} for i in range(len(epochs))}
        assert(sum(epochs)==len(dd['0']['mse']),'epochs list != training length')
        for f, f_dd in dd.items():
            epoch_count=0
            for i, epoch in enumerate(epochs):
                if i==len(epochs)-1:
                    epoch = epoch if mode=='last' else epochs[len(epochs)-2]
                index =epoch_count + epoch if val is None else epoch_count + val
                epoch_count +=epoch
                for metric in metrics:
                    if metric == 'changes':
                        epoch_metric=np.cumsum(f_dd[metric])[index-2]
                    else:
                        epoch_metric=f_dd[metric][index-1]
                    model_data[i][metric].append(epoch_metric)
                if epoch_flag:
                    epochs_list.append(epoch_count)
            epoch_flag=False
        converted_data[model] = model_data
    return converted_data, epochs_list

def custom_axes(ax,xlabel,ylabel=None,grid=True,fontsize=20,xlims=None,ylims=None,
                yticks=None, logscale=True,yticklabels=None):
    if logscale:
        ax.set_yscale('log')
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)
    # yticks = np.arange(0, 6001, 1500) if yticks is None else yticks
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis='both',which='both', labelsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if grid:
        ax.grid(True, color=(0.92, 0.92, 0.92, 1), zorder=0)

def add_cbar(ax,tick_labels,colors, fontsize=16):
    cmap = ListedColormap([colors[lr] for lr in tick_labels])
    n = len(colors)
    bounds = np.linspace(0, n, n + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical', fraction=0.3, pad=0.07, shrink=0.95)
    cbar.set_label('controller learning rate', fontsize=0.8*fontsize)
    cbar.set_ticks(np.arange(n).astype('int') + 0.5)
    cbar.set_ticklabels(tick_labels, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    


# %% [markdown]
# # Data processing/loading

# %%
_continual_data_loaded = None

def get_continual_data():
    global _continual_data_loaded
    if _continual_data_loaded is None:
        mse_data = load_continual_metrics(
            load_folder,
            UPDATE_METRICS,
            pretrainings_folder,
            normalize_changes=True,
            metrics=['mse', 'nocb_error', 'cortex_error','changes']
        )

        epochs = mse_data['epochs']
        angles = mse_data['angles']

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

        N = 5
        color_values = [0.9 - 0.6 * i / (N - 1) for i in range(N)]
        reds = {i: Reds(color_values[i]) for i in range(N)}
        greens = {i: Greens(color_values[i]) for i in range(N)}

        cxbp_lrs = sorted(cxbp_data['normal'].keys(), key=lambda x: float(x), reverse=True)
        cxlocal_lrs = sorted(cxlocal_data['normal'].keys(), key=lambda x: float(x), reverse=True)

        cxbp_data['normal']['RNN'] = mse_data['RNN']
        cxbp_data['h-gain']['RNN'] = mse_data['RNN']
        cxlocal_data['normal']['RNN'] = mse_data['RNN']
        cxlocal_data['h-gain']['RNN'] = mse_data['RNN']

        lr_colors_cxbp = {lr: reds[4 - i] for i, lr in enumerate(cxbp_lrs)}
        lr_colors_cxbp['RNN'] = 'k'
        lr_colors_cxlocal = {lr: greens[4 - i] for i, lr in enumerate(cxlocal_lrs)}
        lr_colors_cxlocal['RNN'] = 'k'

        _continual_data_loaded = {
            'mse_data': mse_data, 'epochs': epochs, 'angles': angles,
            'cxbp_data': cxbp_data, 'cxlocal_data': cxlocal_data,
            'lr_colors_cxbp': lr_colors_cxbp, 'lr_colors_cxlocal': lr_colors_cxlocal,
        }
    return _continual_data_loaded

# %% [markdown]
# # Plots

# %%
figsize=(16,8)
dpi=100
fontsize = 18

plot_nocb_error = True
plot_changes = True
plot_phase_ticks = True
logscale=True
metric_name='mse'

start = 0
stop = 4000

lw = 1.2
ms = 4.5
ylims=[5e-8, 3e-2]
yticks=[1e-3, 1e-5, 1e-7]
mean=True
alpha=1
nticks=10
grid=True
markevery=100

mean=True
min_distance=1000
x_margin=250
grid_color = (233/255, 233/255, 233/255, 0.97)
ax_color = (0.15,0.15,0.15,1.0)

# %% [markdown]
# ## DL-BP: continual losses (S10 + unused)

# %%
def figure_continual_losses(name, variant,
              figsize=(16, 8), dpi=100, fontsize=18,
              plot_nocb_error=True, plot_changes=True, logscale=True, metric_name='mse',
              start=0, stop=4000, lw=1.2, ms=4.5, ylims=[5e-8, 3e-2], yticks=[1e-3, 1e-5, 1e-7],
              mean=True, alpha=1, nticks=10, grid=True, markevery=100, x_margin=250,
              grid_color=(233/255, 233/255, 233/255, 0.97), ax_color=(0.15, 0.15, 0.15, 1.0)):
    def prepare():
        cd = get_continual_data()
        cxbp_data = cd['cxbp_data']
        epochs = cd['epochs']
        angles = cd['angles']
        lr_colors_cxbp = cd['lr_colors_cxbp']
        data = cxbp_data[variant]
        title_map = {'normal': "weak feedback (not used)", 'h-gain': "S10"}
        return {
            'figuretype': 'continual_phase_summary',
            'meta': {
                'variant': variant, 'title': title_map[variant], 'figsize': figsize, 'dpi': dpi,
                'fontsize': fontsize, 'plot_nocb_error': plot_nocb_error, 'plot_changes': plot_changes,
                'logscale': logscale, 'metric_name': metric_name, 'start': start, 'stop': stop,
                'lw': lw, 'ms': ms, 'ylims': ylims, 'yticks': yticks, 'mean': mean, 'alpha': alpha,
                'nticks': nticks, 'grid': grid, 'markevery': markevery, 'x_margin': x_margin,
                'grid_color': grid_color, 'ax_color': ax_color,
            },
            'panels': [
                {'metric': 'angles', 'series': []},
                {'metric': 'mse', 'series': []},
                {'metric': 'changes', 'series': []},
            ],
            'data': data, 'epochs': epochs, 'angles': angles, 'lr_colors_cxbp': lr_colors_cxbp,
        }

    def render(spec):
        meta = spec['meta']
        data = spec.get('data')
        building = data is not None
        spec['panels'][0]['series'] = []
        if building:
            spec['panels'][1]['series'] = []
            spec['panels'][2]['series'] = []
        fig, axs = setup_plot(meta['plot_changes'], figsize=meta['figsize'], dpi=meta['dpi'])
        fig.suptitle(meta['title'], fontsize=1.25 * meta['fontsize'])

        # Plot angles
        gray = (20/255, 20/255, 20/255, 0.6)
        plot_angles(axs[0], spec['epochs'], spec['angles'], fontsize=meta['fontsize'], yticks=[-60, 0, 60],
                    title='', grid=meta['grid_color'], spine_lw=0.7, xlims=meta['x_margin'],
                    positive_color=gray, negative_color=gray, neutral_color=gray, fill=True,
                    ax_color=meta['ax_color'], start=meta['start'], stop=meta['stop'],
                    capture=spec['panels'][0]['series'])
        # Plot metrics
        plot_metrics(
            axs[1:], data, spec.get('lr_colors_cxbp'), meta['lw'], meta['plot_nocb_error'],
            meta['plot_changes'], meta['fontsize'], meta['start'], meta['stop'], ms=meta['ms'],
            markevery=meta['markevery'], metric_name=meta['metric_name'], logscale=meta['logscale'],
            ylims=meta['ylims'], yticks=meta['yticks'], mean=meta['mean'], alpha=meta['alpha'],
            median_w=10, legend=False,
            capture_mse=spec['panels'][1]['series'] if building else None,
            capture_changes=spec['panels'][2]['series'] if building else None,
            replay_mse=None if building else spec['panels'][1]['series'],
            replay_changes=None if building else spec['panels'][2]['series'],
        )

        # Annotate phases
        annotate_phases(axs[1:], spec['angles'], spec['epochs'], meta['fontsize'], meta['start'], meta['stop'],
                         None, meta['lw'], False)

        # Customize axes
        axs[-1].set_yscale('linear')
        customize_axes(axs[1:], meta['fontsize'], x_margin=meta['x_margin'], nticks=meta['nticks'],
                        grid=meta['grid'], grid_color=meta['grid_color'], ax_color=meta['ax_color'],
                        y_ticks=None, y_ticklabels=None, start=meta['start'], stop=meta['stop'])
        axs[-1].set_ylabel(r'cumulative changes')
        plt.tight_layout()
        for ax in axs:
            ax.set_xticks(np.linspace(0, 4000, 5), color=meta['ax_color'], fontsize=meta['fontsize'])
            ax.set_xticklabels([f'{int(i)}' for i in np.linspace(0, 4000, 5)], color=meta['ax_color'],
                                fontsize=meta['fontsize'])

        if building:
            del spec['data']

        return fig, axs

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
for name, variant in zip(["S10","S10_weak_unused"],['h-gain', 'normal']):
    fig, spec = figure_continual_losses(name, variant,
                           figsize=figsize, dpi=dpi, fontsize=fontsize,
                           plot_nocb_error=plot_nocb_error, plot_changes=plot_changes,
                           logscale=logscale, metric_name=metric_name, start=start, stop=stop,
                           lw=lw, ms=ms, ylims=ylims, yticks=yticks, mean=mean, alpha=alpha,
                           nticks=nticks, grid=grid, markevery=markevery, x_margin=x_margin,
                           grid_color=grid_color, ax_color=ax_color)
    save_plot(fig, name, plots_folder, savepdf=save_pdf, savetiff=save_tiff,
                      savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
                      dpi=dpi, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## continual compact

# %%
def figure_continual_compact(name, dataset_key, variant, lr_colors_key, title, gain,
                    fontsize=17, stop=4000, xlabel='phase', ylabel='error (adaptation)',
                    metrics=['mse', 'changes'], figsize=(5, 8), ratios=[0.7, 1, 1]):
    def prepare():
        cd = get_continual_data()
        source = cd[dataset_key]
        epochs = cd['epochs']
        angles = cd['angles']
        lr_colors = cd[lr_colors_key]

        data = source[variant].copy()
        data_rnn = {'RNN': data.pop('RNN')}
        dd, epochs_list = per_phase_loss(data, epochs, angles, mode=500, val=None, selection=None)
        basics, _ = per_phase_loss(data_rnn, epochs, angles, mode=500, val=None, selection=None)

        x = np.arange(len(epochs_list))
        lrs = [lr for lr in lr_colors.keys() if lr != 'RNN']
        colors = {k: v for k, v in lr_colors.items() if k != 'RNN'}

        basic_colors = {'RNN': 'k'}
        basic_labels = {'RNN': 'plastic controller'}

        return {
            'figuretype': 'phase_wise_losses',
            'meta': {'title': title, 'gain': gain, 'fontsize': fontsize, 'stop': stop,
                     'xlabel': xlabel, 'ylabel': ylabel, 'metrics': metrics},
            'panels': [
                {'metric': 'angles', 'series': []},
                {'metric': 'mse', 'series': []},
                {'metric': 'changes', 'series': []},
            ],
            'x': x, 'lrs': lrs, 'colors': colors, 'basic_colors': basic_colors,
            'basic_labels': basic_labels, 'dd': dd, 'basics': basics,
            'epochs_list': epochs_list, 'angles': angles,
        }

    def render(spec):
        spec['panels'][0]['series'] = []
        spec['panels'][1]['series'] = []
        spec['panels'][2]['series'] = []
        meta = spec['meta']
        x = spec['x']
        colors = spec['colors']
        basic_colors = spec['basic_colors']
        basic_labels = spec['basic_labels']

        fig, axs = plt.subplots(
            3, 1, figsize=figsize, dpi=96,
            sharex=False, sharey=False,
            gridspec_kw={'height_ratios': ratios}
        )

        plot_angles_mini(
            axs[0], spec['epochs_list'], spec['angles'],
            fontsize=meta['fontsize'], yticks=[-60, 0, 60],
            title='', grid='w', spine_lw=0.7, xlims=200,
            positive_color='gray', negative_color='gray', neutral_color='gray',
            fill=True, ax_color='k', start=0, stop=meta['stop'], phasex=True, xlabel='phase',
            dtx=True, capture=spec['panels'][0]['series']
        )

        for j, (ax, metric) in enumerate(zip(axs[1:3], meta['metrics'])):
            panel = spec['panels'][1 + j]
            lw = 1.2
            linestyle = 'dashed'
            fmt = ''
            alpha = 1
            ms = 5
            for model, dd_model in spec['basics'].items():
                values = np.array([dd_model[e][metric] for e in range(len(x))]).T
                means = values.mean(axis=0)
                mins = values.min(axis=0)
                maxs = values.max(axis=0)
                yerr = np.vstack((means - mins, maxs - means))

                ax.errorbar(
                    x, means, yerr=yerr, label=basic_labels[model], fmt='',
                    marker='o', markersize=ms, linestyle=linestyle, lw=lw,
                    elinewidth=0.9, color=basic_colors[model],
                    markerfacecolor=basic_colors[model], markeredgewidth=0.9,
                    capsize=5, alpha=alpha, zorder=3
                )
                panel['series'].append({'name': f'basic_{model}', 'x': x.tolist(), 'y': means.tolist(),
                                        'xfill': x.tolist(), 'ymin': mins.tolist(), 'ymax': maxs.tolist()})
            for model, dd_model in spec['dd'].items():
                values = np.array([dd_model[e][metric] for e in range(len(x))]).T
                means = np.mean(values, axis=0)
                mins = values.min(axis=0)
                maxs = values.max(axis=0)
                yerr = np.vstack((means - mins, maxs - means))


                ax.errorbar(
                    x, means, yerr=yerr, fmt=fmt,
                    marker='o', markersize=ms, lw=lw,
                    label='full model' if model == list(spec['dd'].keys())[0] else None,
                    elinewidth=0.5, color=colors[model],
                    markerfacecolor=colors[model], markeredgewidth=0.4,
                    capsize=5, alpha=alpha, zorder=3
                )
                panel['series'].append({'name': model, 'x': x.tolist(), 'y': means.tolist(),
                                        'xfill': x.tolist(), 'ymin': mins.tolist(), 'ymax': maxs.tolist()})

            if metric == 'changes':
                custom_axes(
                    ax, meta['xlabel'],
                    ylabel=(meta['ylabel'] if j == 0 else 'cumul. % changes'),
                    grid=False, fontsize=meta['fontsize'], xlims=(-1, 8),
                    ylims=(-0.01, 0.02), yticks=[0., 0.1, 0.2], logscale=False,
                    yticklabels=['0','10','20']
                )
            else:
                custom_axes(
                    ax, meta['xlabel'],
                    ylabel=(meta['ylabel'] if j == 0 else 'cumul. % changes'),
                    grid=False, fontsize=meta['fontsize'], xlims=(-1, 8),
                    ylims=(8e-8, 1e-4), yticks=[1e-4, 1e-5, 1e-6, 1e-7]
                )

            ax.set_xticks(np.arange(len(spec['epochs_list'])))
            ax.set_xticklabels(np.arange(1, len(spec['epochs_list']) + 1))
            ax.legend(fontsize=0.8 * meta['fontsize'], loc='upper left')

        plt.tight_layout()
        return fig, axs

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec

# %% [markdown]
# ### 6

# %%
for variant, title, gain in zip(
    ['normal', 'h-gain'],
    ['weak feedback (unused)', 'strong feedback (5b)'],
    ['gain1', 'gain20']
):
    print(title)
    name = "6" if gain == 'gain20' else "6_weak_unused"
    fig, spec = figure_continual_compact(name, 'cxbp_data', variant, 'lr_colors_cxbp', title, gain,
                                         figsize=(6.25, 10), ratios=[0.7, 1, 1])
    save_plot(fig, name, plots_folder, savepdf=save_pdf, savetiff=save_tiff,
                      savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
                      dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## S14 a-b

# %%
for variant, title, gain in zip(
    ['normal', 'h-gain'],
    ['weak feedback (S14a)', 'strong feedback (S14b)'],
    ['gain1', 'gain20']
):
    print(title)
    name = "S14a" if gain == 'gain1' else "S14b"
    fig, spec = figure_continual_compact(name, 'cxlocal_data', variant, 'lr_colors_cxlocal', title, gain, figsize=(5, 8))
    save_plot(fig, name, plots_folder, savepdf=save_pdf, savetiff=save_tiff,
                      savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
                      dpi=96, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# %% [markdown]
# ## S15

# %%
def figure_S15(mode='last', val=500, fontsize=18, xlabel='phase', ylabel='error (adaptation)',
               metrics=['mse', 'changes'], stop=4000):
    def prepare():
        cd = get_continual_data()
        cxlocal_data = cd['cxlocal_data']
        cxbp_data = cd['cxbp_data']
        epochs = cd['epochs']
        angles = cd['angles']
        lr_colors_cxlocal = cd['lr_colors_cxlocal']
        lr_colors_cxbp = cd['lr_colors_cxbp']
        comparison_data = {
            'local, weak-fb':     cxlocal_data['normal']['5e-03'],
            'local, strong-fb':   cxlocal_data['h-gain']['5e-03'],
            'backprop, weak-fb':  cxbp_data['normal']['1e-05'],
            'backprop, strong-fb': cxbp_data['h-gain']['1e-05'],
        }

        baseline_data = {
            'plastic controller': cxlocal_data['normal']['RNN'],
        }

        final_losses, epochs_list = per_phase_loss(
            comparison_data, epochs, angles, mode=mode, val=val, selection=None
        )
        basics, _ = per_phase_loss(
            baseline_data, epochs, angles, mode=mode, val=val, selection=None
        )

        x = np.arange(len(epochs_list))

        colors = {
            'local, weak-fb':     lr_colors_cxlocal['5e-04'],
            'local, strong-fb':   lr_colors_cxlocal['5e-04'],
            'backprop, weak-fb':  lr_colors_cxbp['5e-07'],
            'backprop, strong-fb': lr_colors_cxbp['5e-07'],
        }

        marker = {
            'local, weak-fb':     'o',
            'local, strong-fb':   's',
            'backprop, weak-fb':  'o',
            'backprop, strong-fb': 's',
        }

        basic_colors = {'plastic controller': 'k'}

        return {
            'figuretype': 'phase_wise_comparison',
            'meta': {'fontsize': fontsize, 'xlabel': xlabel, 'ylabel': ylabel,
                     'metrics': metrics, 'stop': stop},
            'panels': [
                {'metric': 'angles', 'series': []},
                {'metric': 'mse', 'series': []},
                {'metric': 'changes', 'series': []},
            ],
            'x': x, 'colors': colors, 'marker': marker, 'basic_colors': basic_colors,
            'comparison_models': list(comparison_data.keys()), 'final_losses': final_losses,
            'basics': basics, 'epochs_list': epochs_list, 'angles': angles,
        }


    def render(spec):
        spec['panels'][0]['series'] = []
        spec['panels'][1]['series'] = []
        spec['panels'][2]['series'] = []
        meta = spec['meta']
        x = spec['x']
        colors = spec['colors']
        marker = spec['marker']
        basic_colors = spec['basic_colors']

        fig, axs = plt.subplots(
            3, 1, figsize=(10, 10), dpi=96,
            sharex=False, sharey=False,
            gridspec_kw={'height_ratios': [0.7, 1, 1]}
        )

        plot_angles_mini(
            axs[0], spec['epochs_list'], spec['angles'],
            fontsize=meta['fontsize'], yticks=[-60, 0, 60],
            title='', grid='w', spine_lw=0.7, xlims=200,
            positive_color='gray', negative_color='gray', neutral_color='gray',
            fill=True, ax_color='k', start=0, stop=meta['stop'], phasex=True, xlabel='phase',
            dtx=True, capture=spec['panels'][0]['series']
        )

        for j, (ax, metric) in enumerate(zip(axs[1:3], meta['metrics'])):
            panel = spec['panels'][1 + j]
            for model in spec['comparison_models']:
                dd_model = spec['final_losses'][model]
                values = np.array([dd_model[e][metric] for e in range(len(x))]).T
                means = np.mean(values, axis=0)
                mins = values.min(axis=0)
                maxs = values.max(axis=0)

                ax.plot(
                    x, means,
                    label=model, ms=9,
                    marker=marker[model],
                    color=colors[model], linestyle='solid',
                    alpha=0.6, markeredgecolor=colors[model],
                    markerfacecolor='none', markeredgewidth=1.4
                )
                panel['series'].append({'name': model, 'x': x.tolist(), 'y': means.tolist()})

            for model, dd_model in spec['basics'].items():
                values = np.array([dd_model[e][metric] for e in range(len(x))]).T
                means = values.mean(axis=0)

                ax.plot(
                    x, means,
                    label=model, ms=9, marker='o',
                    color=basic_colors[model], linestyle='dashed',
                    alpha=0.6, markeredgecolor=basic_colors[model],
                    markerfacecolor='none', markeredgewidth=1.4
                )
                panel['series'].append({'name': f'basic_{model}', 'x': x.tolist(), 'y': means.tolist()})

            if metric == 'changes':
                custom_axes(
                    ax, meta['xlabel'],
                    ylabel=(meta['ylabel'] if j == 0 else 'cumulative changes'),
                    grid=False, fontsize=meta['fontsize'],
                    xlims=(-1, 8), ylims=(-0.01, 0.02),
                    yticks=[0., 0.1, 0.2], logscale=False,
                    yticklabels=['0%','10%','20%']
                )
            else:
                custom_axes(
                    ax, meta['xlabel'],
                    ylabel=(meta['ylabel'] if j == 0 else 'cumulative changes'),
                    grid=False, fontsize=meta['fontsize'],
                    xlims=(-1, 8), ylims=(6e-8, 1e-5),
                    yticks=None
                )

            ax.legend(fontsize=0.8 * meta['fontsize'], loc='upper left')
            ax.set_xticks(np.arange(len(spec['epochs_list'])))
            ax.set_xticklabels(np.arange(1, len(spec['epochs_list']) + 1))
            ax.grid(True, color=(233/255, 233/255, 233/255, 0.97))

        plt.tight_layout()
        return fig, axs

    spec = load_or_build_spec("S15", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec

# %%
fig, spec = figure_S15()
save_plot(fig, "S15", plots_folder, savepdf=save_pdf, savetiff=save_tiff,
                  savedata=save_data, data=spec, plotsdatafolder=plots_data_folder,
                  dpi=96, transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

# %%
