# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: selma (3.8.10)
#     language: python
#     name: python3
# ---

# %%
"""Figure generation for the fixed-cortex experiments."""
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

load_folder = os.path.join(DATA_DIR, 'fixed_cortex')
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

notebook_header(HOME, display_width=70, font_family="Arial", seed=25)

import torch
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
from matplotlib.ticker import LogLocator
from matplotlib import cm

from ccdl.utils import *
from ccdl.reaching import ReachingDataset
from ccdl.ccnet import CerebelloCorticalNet



##############################################################################################
# OTHER SETTINGS

# Making titles always visible
import matplotlib as mpl

mpl.rcParams['axes.titley'] = None  # let matplotlib auto-place
mpl.rcParams['axes.titlepad'] = 6

# Patch title rendering to always use a bbox
import matplotlib.axes as maxes
_orig_set_title = maxes.Axes.set_title

def _patched_set_title(self, label, *args, **kwargs):
    kwargs.setdefault('bbox', dict(facecolor='white', edgecolor='none', alpha=0.65, pad=3))
    return _orig_set_title(self, label, *args, **kwargs)

maxes.Axes.set_title = _patched_set_title


device = torch.device('cpu')
dtype = 'float32'

npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)
rnn_hidden_size = 256
model_class = "CerebelloCorticalNet"




# %% [markdown]
# # Methods

# %%

def get_rescaling(dataset_params, norm_mode):
    reaching = ReachingDataset(dataset_params)
    targets = reaching.generate_unperturbed_dataset()[1]
    if norm_mode == 'normal':
        return targets.std().item()
    else:
        return targets.max().item() - targets.min().item()

def moving_average(data, window_size):
    if len(data.shape) == 2:
        for i in range(data.shape[0]):
            data[i] = np.convolve(data[i], np.ones(window_size) / window_size, mode='same')
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

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

def _get_rescaling_from_params(config, datatype, mode='normal'):
    """Helper: compute rescaling from a full config dict (mirrors BaseModelProcessor.get_rescaling)."""
    dataset_params = config['dataset_params']
    reaching = ReachingDataset(dataset_params)
    target = reaching.generate_unperturbed_dataset(datatype)[1]
    if mode.lower() == 'minmax':
        return target.max().item() - target.min().item()
    elif mode.lower() == 'normal':
        return target.std().item()
    else:
        raise ValueError('Please choose a mode between minmax and normal for rescaling!')

def plot_loss_curves(mse_data=None, params_folder=None, norm_mode='normal', ax=None, figsize = (20,6), start=0, steps=20,
                  smoothing_window=10, selection=None, alpha=0.3, ls=None, lims=(8e-7,1e-1),
                  figure=False, title=None, fill = True, alpha_fill = 0.2, lws=1, stop=None, dpi=60, fontsize=14, label_names = None, plot_iters = False, colors=None,
                  groups_size=4, plot_dataset=True, norm=False,contrast=False, error_bars=False,
                  bbox_to_anchor=None, in_yticks=None, in_ylims=None, in_xticks=None,average_mode='mean',
                  inset=False, inset_specs=None,stop_inset = 2000,legend_loc=None, legend=True,
                  smoothing='moving_average', label_q='name', markersize=5, return_fig=False,
                  minortickson=False, fill_smoothing=(5,5), debug=False, x_ticks=None, y_ticks=None, capture=None, replay=None):
    inset_specs = inset_specs if inset_specs else [0.05,0.6,0.6,0.35]
    return_ax = False

    if replay is not None:
        assert not (inset or error_bars or contrast or plot_iters), \
            "plot_loss_curves replay only supports mean-line + fill-band (inset/error_bars/contrast/plot_iters unused at this script's call site)"

        if figure:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
            if title:
                fig.suptitle(title, fontsize=1.4 * fontsize)
        elif ax:
            return_ax = True
        else:
            ax = plt.gca()
            if title:
                ax.set_title(title, fontsize=1.1 * fontsize)

        ax.set_rasterization_zorder(2.5)

        for entry in replay:
            if fill:
                ax.fill_between(entry['xfill'], entry['ymin'], entry['ymax'], alpha=alpha_fill,
                                 facecolor='none', color=entry['color'], linestyle=entry['linestyle'],
                                 hatch=None, linewidth=0.0, zorder=entry['fill_zorder'])
            ax.plot(entry['x'], entry['y'], lw=1.8 * entry['lw'], alpha=2 * alpha,
                    linestyle=entry['linestyle'], marker=None, label=entry['label'],
                    color=entry['color'], markeredgecolor=None, markeredgewidth=None,
                    zorder=entry['zorder'])

        ax.set_yscale('log')
        ax.set_ylim(*lims)
        if x_ticks:
            ax.set_xticks(x_ticks)
        if y_ticks:
            ax.set_yticks(y_ticks)
        ax.set_xticks(np.arange(0, stop + 1, stop // 4))
        ax.set_ylabel('error (adaptation)', fontsize=fontsize)
        ax.set_xlabel('epoch', fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set(linewidth=1)
        ax.spines['left'].set(linewidth=1)
        ax.tick_params(axis='x', which='major', width=1, length=3)
        ax.tick_params(axis='y', which='major', width=1, length=3)
        ax.tick_params(axis='y', which='minor', width=0.3, length=1.4)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if legend:
            ax.legend(by_label.values(), by_label.keys(), fontsize=0.7 * fontsize, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)

        if return_ax:
            return ax
        if figure and return_fig:
            return fig
        if figure:
            plt.show()
        return

    if norm:
        path = os.path.join(params_folder, 'template_config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        cart_rescaling = _get_rescaling_from_params(config, 'cart', mode=norm_mode)
        thetas_rescaling = _get_rescaling_from_params(config, 'angles', mode=norm_mode)

    smoothing = globals()[smoothing]

    label_names = {
        'r_init': "Random init., backprop",
        'ae_init': "Pseudoinv. init., backprop",
        'ae_fixed': "Pseudoinv.,  fixed"
    } if not label_names else label_names

    groups = list(label_names.keys())
    colors = {group: f'C{i}' for i, group in enumerate(groups)} if not colors else colors
    hatches = {group: None for group in groups}
    # hatches = {group: hatch for group, hatch in zip(groups, ['/', '|', '-','\\','*'][:len(groups)])}

    if figure:
        fig = plt.figure(figsize=figsize,dpi=dpi)
        ax = plt.gca()
        if title:
            fig.suptitle(title,fontsize=1.4*fontsize)
    elif ax:
        return_ax = True
    else:
        ax = plt.gca()
        if title:
            ax.set_title(title,fontsize=1.1*fontsize)

    ax.set_rasterization_zorder(2.5)
    axins=None
    if inset:
        #add inset centered, on the top part of the plot
        axins = ax.inset_axes(inset_specs)
        axins.set_yscale('log')
        axins.set_yticks([1e-2,1e-4,1e-6])
        axins.set_ylim([1e-7,1e-2])
        axins.yaxis.set_ticks_position('right')
        axins.tick_params(axis='both', which='both', labelsize=0.8*fontsize)
        # axins.yaxis.grid(True, color=(233/255, 233/255, 233/255, 97/100))
        x_inset = np.linspace(0, stop_inset, stop_inset + 1)
    
    for group in groups:
        lw =lws if isinstance(lws, float) else lws[group]
        for l,((order,name), data) in enumerate(mse_data.groupby(['order','name'])):
            if selection is not None:
                assert isinstance(selection,list), "selection must be list(str,bool)"
                if (selection[0] in name)!=selection[1]:
                    continue
            # selecting only mse metric
            if 'metric' in data.index.names:
                unique_metrics = data.index.get_level_values('metric').unique()
                # Check if 'mse' is present and there are other metrics
                if 'mse' in unique_metrics and len(unique_metrics) > 1:
                    data = data.loc[(slice(None), slice(None), slice(None), 'mse'), :]
            
            if ls == 'correction':
                hatch = ''
                if not ('correction_fb' in name):
                    linestyle = 'solid'
                    marker = None
                    label = ''
                else:
                    linestyle = 'solid'
                    marker = 'o'
                    label = ' | correction feedback'
            elif ls =='plant':
                hatch = ''
                if 'no_plant' in name:
                    linestyle = 'solid'
                    marker=None
                    cc = 'o'
                    label = ' | No plant'
                else:
                    linestyle = 'solid'
                    marker='o'
                    label = ' | Plant'
                    
            else:
                label=''
                linestyle='solid' if group != 'RNN' else 'dashed'
                if fill:
                    hatch = hatches[group]
                marker=None
            
            if label_q == 'name':
                if group not in name:
                    continue
            else:
                if group!=str(order):
                    continue
            data = data.values
            if norm:
                if 'no_plant' in name:
                    data = data / thetas_rescaling**2
                else:
                    data = data / cart_rescaling**2
            
            markers_steps = (data.shape[1]//20)//steps
            if smoothing_window>0:
                data = apply_gaussian_smoothing(smoothing(data, smoothing_window),10)
            
            stop = data.shape[1] if not stop else stop
            if average_mode=='mean':
                mean = data[:,start:stop:steps].mean(axis=0)
            else:
                mean = np.median(data[:,start:stop:steps],axis=0)
                
            mins = apply_gaussian_smoothing(moving_median(data[:,start:stop:2*steps].min(axis=0),fill_smoothing[0]),fill_smoothing[1])
            mins = np.minimum(mins, mean[::2])
            maxs = apply_gaussian_smoothing(moving_median(data[:,start:stop:2*steps].max(axis=0),fill_smoothing[0]),fill_smoothing[1])
            maxs = np.maximum(maxs, mean[::2])

            if capture is not None:
                capture.append({
                    'name': name, 'label': label_names[group]+label, 'color': colors[group],
                    'x': np.linspace(start,stop,len(mean)).tolist(), 'y': mean.tolist(),
                    'xfill': np.linspace(start,stop,len(mins))[:-1].tolist(),
                    'ymin': mins[:-1].tolist(), 'ymax': maxs[:-1].tolist(),
                    'raw_y': [data[j,start:stop:steps].tolist() for j in range(data.shape[0])] if plot_iters else None,
                    'linestyle': linestyle, 'lw': lw,
                    'zorder': 4 if group != 'RNN' else 3,
                    'fill_zorder': 2 if group != 'RNN' else 0,
                })

            if error_bars:
                x_offset=100
                error_epochs = np.linspace(0, stop-start-1, 6, dtype=int)[1:]-l*x_offset#np.arange(start,stop+1,(len(mean))//10)[:-1]
                errors = [(mean[::2]-mins)[error_epochs//2], (maxs-mean[::2])[error_epochs//2]]
                ax.errorbar(start + error_epochs+1, mean[error_epochs], yerr=errors, fmt='o', capsize=5, capthick=1.8, 
                            color='k',ecolor='k', markersize=4,markerfacecolor=colors[group], alpha=0.8,zorder=4)
            if fill:
                ax.fill_between(np.linspace(start,stop,len(mins))[:-1], mins[:-1], maxs[:-1], alpha=alpha_fill,
                                 facecolor='none', color=colors[group], linestyle=linestyle, hatch=hatch, linewidth=0.0, zorder=2 if group != 'RNN' else 0)
            if plot_iters:
                for j in range(data.shape[0]):
                    dd = data[j,start:stop:steps]
                    ax.plot(np.linspace(start,stop,len(dd)), dd, alpha=alpha, lw=0.6*lw,
                             linestyle=linestyle, marker=marker, markersize=markersize if marker else None, 
                             markevery=markers_steps if marker else None, color=colors[group], 
                             markeredgecolor='k' if marker else None, markeredgewidth=0.2*lw if marker else None, zorder=3)
                    
            if contrast:
                ax.plot(np.linspace(start,stop,len(mean)), mean, lw=1.84*lw, alpha=2*alpha,
                         linestyle=linestyle, marker=marker, markersize=markersize if marker else None, 
                         markevery=markers_steps if marker else None, color='k', 
                         markeredgecolor='k' if marker else None, markeredgewidth=0.2*lw if marker else None, zorder=4 if group != 'RNN' else 3)
                
            ax.plot(np.linspace(start,stop,len(mean)), mean, lw=1.8*lw, alpha=2*alpha,
                     linestyle=linestyle, marker=marker, markersize=markersize if marker else None, 
                     markevery=markers_steps if marker else None, color=colors[group], 
                     label=label_names[group]+label, markeredgecolor='k' if marker else None, 
                     markeredgewidth=0.1*lw if marker else None, zorder=4 if group != 'RNN' else 3)
            
            if inset:
                mean = data.mean(axis=0)[:stop_inset+1]
                axins.plot(np.linspace(start,stop_inset,len(mean)), mean, lw=1.7*lw, alpha=2*alpha,
                     linestyle=linestyle, marker=marker, markersize=markersize if marker else None, 
                     markevery=markers_steps if marker else None, color=colors[group], 
                     label=label_names[group]+label, markeredgecolor='k' if marker else None, 
                     markeredgewidth=0.1*lw if marker else None)
    if debug:
        print("marker: ",marker)
        print("markersize: ",markersize)
        print("markers_steps: ",markers_steps)
    ax.set_yscale('log')
    # if minortickson:
    #     ax.grid(which='both', color='gray', linestyle='-', linewidth=0.4)
    # else:
    #     ax.grid(linestyle='-', linewidth=0.4)
    ax.set_ylim(*lims)
    if x_ticks: 
        ax.set_xticks(x_ticks)
    if y_ticks: 
        ax.set_yticks(y_ticks)
    if axins:
        axins.set_yticks([1e-3,1e-5,1e-7]) if not in_yticks else axins.set_yticks(in_yticks)
        if in_ylims:
            axins.set_ylim(*in_ylims)
        if in_xticks:
            axins.set_xticks(in_xticks)
    ax.set_xticks(np.arange(0, stop+1, stop//4))
    grid_color = (233/255, 233/255, 233/255, 97/100)
    # ax.grid(True, color=grid_color, linewidth=0.64, zorder=0)
    ax.set_ylabel('error (adaptation)', fontsize=fontsize)
    ax.set_xlabel('epoch', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set(linewidth=1)
    ax.spines['left'].set(linewidth=1)
    ax.tick_params(axis='x', which='major', width=1, length=3)
    ax.tick_params(axis='y', which='major', width=1, length=3)
    ax.tick_params(axis='y', which='minor', width=0.3, length=1.4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if legend:
        ax.legend(by_label.values(), by_label.keys(),fontsize=0.7*fontsize, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
    
    if return_ax:
        return ax
    if figure and return_fig:
        return fig
    if figure:
        plt.show()

def plot_comparison_partial(metrics_list=None, ax=None, colors=None, stop=-1, markersize = 5,
                         mode='mean', lims=(1e-8,1e-1), labels=None,contrast=False, legend=False,
                         alpha=0.8, title=None, median_w=0, step=1, lw=1.6, fontsize =12,
                         inset=False, lr_list=None, inset_specs=None,stop_inset = 2000,
                         inset_lims=None, legend_loc='upper right', highlight=None, capture=None, replay=None):
    from matplotlib.lines import Line2D

    if replay is not None:
        assert mode == 'mean' and not inset and not contrast and not highlight, \
            "plot_comparison_partial replay only supports mode='mean' with no inset/contrast/highlight (unused at this script's call site)"

        legend_elements = []
        for entry in replay:
            if title:
                ax.set_title(title, fontsize=fontsize)
            is_rnn = 'RNN' in entry['name']
            t_color = entry['color']
            if not is_rnn:
                r, g, b, a = tuple(t_color)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                t_color = (luminance, luminance, luminance, a)
            legend_elements.append(Line2D([0], [0], linestyle=entry['linestyle'], color=t_color,
                                           label=entry['label'], lw=1.3 * lw))
            ax.plot(entry['x'], entry['y'], linestyle=entry['linestyle'], color='w', alpha=0.8,
                    lw=entry['lw'], zorder=entry['zorder'])
            ax.plot(entry['x'], entry['y'], linestyle=entry['linestyle'], color=entry['color'],
                    lw=entry['lw'], zorder=entry['zorder'])

        ax.set_yscale('log')
        ax.set_xticks(np.arange(0, stop + 1, stop // 4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set(linewidth=1.3)
        ax.spines['left'].set(linewidth=1.3)
        ax.tick_params(axis='both', which='both', labelsize=fontsize)
        ax.tick_params(axis='x', which='major', width=1.3, length=4)
        ax.tick_params(axis='y', which='major', width=1.3, length=4)
        ax.set_ylim(lims)
        ax.set_xlim(-stop // 30, 1.05 * stop)
        if legend:
            ax.legend(handles=legend_elements, fontsize=fontsize, loc=legend_loc)
        return ax, legend_elements

    assert (step >=1 and isinstance(step, int)), "Step must be a positive integer"
    inset_specs = inset_specs if inset_specs else [0.05,0.6,0.6,0.35]
    inset_lims = inset_lims if inset_lims else [1e-7,1e-1]
    lr_list = ['.lr2e-3','.lr1e-3','.lr5e-4','.lr1e-4'] if lr_list is None else lr_list
    
    linestyles = {'mse': 'solid'}#, 'cortex_error': 'dashed'}
    metrics = list(linestyles.keys())
    N = metrics_list.shape[1]
    if stop == -1:
        stop = N
    x = np.linspace(0, stop-1, stop)[::step]
    legend_elements = []
    # legend_elements = [Line2D([0], [0], linestyle='solid', color='k', label='net loss'),
    #                 Line2D([0], [0], linestyle='dashed', color='k', label='cortex loss')]
    if inset:
        #add inset cetered, on the top part of the plot
        axins = ax.inset_axes(inset_specs)
        axins.set_yscale('log')
        axins.yaxis.set_ticks_position('right') 
        axins.tick_params(axis='both', which='both', labelsize=0.8*fontsize)
        axins.set_yticks([1e-1,1e-3,1e-5,1e-7])
        axins.yaxis.grid(True, color=(233/255, 233/255, 233/255, 97/100))
        x_inset = np.linspace(0, stop_inset, stop_inset + 1)
    
    elements = labels if labels.keys() else lr_list
    for element in elements:
        selection = metrics_list.copy()
        condition = np.array([element in name for name in selection.index.get_level_values('name')])
        selection = selection.loc[condition]
        for metric, data in selection.groupby(['metric']):
            if isinstance(metric, tuple):  # pandas >= 2 yields length-1 tuples here
                metric = metric[0]
            if metric not in metrics:
                continue
            if title:
                ax.set_title(title, fontsize=fontsize)
            plt.tight_layout()
            color = colors[element]
            if labels:
                label = labels[element]
            else:
                label = f'lr: {element[2:]}'
            ls = linestyles[metric] if 'RNN' not in element else 'dashed'
            if metric == 'mse':
                t_color = color
                if 'RNN' not in element:
                    # Calculate luminance using the Rec. 709 formula
                    r, g, b, a = tuple(t_color)
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    t_color = (luminance, luminance, luminance, a)
                legend_elements.append(Line2D([0], [0], linestyle=ls, color=t_color, label=label, lw=1.3*lw))
            marker = 'o' if 'RNN' in element else None
            markers_steps = data.shape[1]//step//20 if 'RNN' in element else None
            if mode in ['mean','both']:
                t_alpha = alpha if mode == 'mean' else max(alpha,0.9)
                t_lw = lw if mode == 'mean' else 1.3*lw
                t_lw = 1.2*lw if 'RNN' in element else t_lw
                mean = data.values.mean(axis=0)[:stop]
                if median_w>0:
                    mean = moving_median(mean, median_w)
                z_order = None
                if contrast or (highlight and highlight in element):
                    z_order=3 if highlight and highlight in element else -1
                    ax.plot(x,mean[::step], linestyle=ls, color='k', lw=1.5*t_lw,zorder=z_order)
                    ax.plot(x,mean[::step], linestyle=ls, color='w', lw=t_lw,zorder=z_order)
                z_order = 3 if 'RNN' not in element else 2
                ax.plot(x,mean[::step], linestyle=ls, color='w', alpha=0.8,lw=t_lw,zorder=z_order)
                ax.plot(x,mean[::step], linestyle=ls, color=color, lw=t_lw,zorder=z_order)
                if capture is not None:
                    capture.append({
                        'name': element, 'label': label, 'color': color,
                        'x': x.tolist(), 'y': mean[::step].tolist(),
                        'linestyle': ls, 'lw': t_lw, 'zorder': z_order,
                    })
            if mode in ['all','both']:
                t_alpha = alpha if mode == 'all' else 0.2
                t_lw = lw if mode == 'all' else 0.8*lw
                for k in range(data.shape[0]):
                    series = data.values[k, :stop]
                    if median_w>0:
                        series = moving_median(series, median_w)
                    ax.plot(x,series[::step], linestyle=linestyles[metric], color=color, lw=t_lw)
                    ax.plot(np.linspace(0,stop,len(mean)), mean, lw=1.8*lw, alpha=min(2*alpha,1), marker=marker, markersize=markersize if marker else None, 
                     markevery=markers_steps if marker else None, color=color, markeredgecolor='k' if marker else None, 
                     markeredgewidth=0.1*lw if marker else None)
            if inset:
                mean = data.values.mean(axis=0)[:stop_inset+1]
                if contrast or (highlight and highlight in element):
                    axins.plot(x_inset, mean, linestyle=ls, color='k', lw=1.5*0.8*lw, zorder=z_order)
                    axins.plot(x_inset, mean, linestyle=ls, color='w', lw=0.8*lw, zorder=z_order)
                axins.plot(x_inset, mean, linestyle=ls,color=color, lw=0.8*lw, zorder=z_order)

    if inset:
        axins.set_ylim(inset_lims)
    grid_color = (233/255, 233/255, 233/255, 97/100)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(0, stop+1, stop//4))
    # ax.grid(True, color=grid_color, linewidth=1)
    # ax.spines['top'].set(linewidth=1, color=grid_color, zorder=-1)
    ax.spines['top'].set_visible(False)
    # ax.spines['right'].set(linewidth=1, color=grid_color)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set(linewidth=1.3)
    ax.spines['left'].set(linewidth=1.3)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)
    ax.tick_params(axis='x', which='major', width=1.3, length=4)
    ax.tick_params(axis='y', which='major', width=1.3, length=4)
    ax.set_ylim(lims)
    ax.set_xlim(-stop//30, 1.05*stop)
    if legend:
        ax.legend(handles=legend_elements, fontsize=fontsize,loc=legend_loc)
    return ax, legend_elements


def plot_outputs_pre(folder, iterations=False, plot_epoch=-1, plot_unpert=False,
                 ax=None, title='', alpha_targets=0.8, empty_endpoints=False,
                 colors=None, figsize=None, dpi=100, markersize=360, edge=3.2, contrast=False,
                 alpha=0.2, x_lim=(-0.120,0.175), y_lim=(None,None), start=0, lw=2,
                 stop=-1, fontsize=12, marker='o', no_axes=False, add_dots=False, capture=None, replay=None):

    if replay is not None:
        assert empty_endpoints, \
            "plot_outputs_pre replay only supports empty_endpoints=True (the only path exercised by this script's actual call site)"

        if ax is None:
            figsize = (10, 7) if figsize is None else figsize
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        if title:
            ax.set_title(title, fontsize=fontsize)

        s_unpert = markersize
        s_pert = markersize

        for entry in replay:
            color = entry['color']
            comps = {c['component']: c for c in entry['components']}

            if 'unperturbed_target' in comps:
                c = comps['unperturbed_target']
                ax.scatter(c['x'][0], c['y'][0], alpha=0.4, marker="^", s=s_unpert,
                           facecolor='none', edgecolor=color, linewidths=edge, zorder=2,
                           label=entry['unpert_label'])

            c = comps['perturbed_target']
            ax.scatter(c['x'][0], c['y'][0], color=color, alpha=alpha_targets, marker=marker,
                       s=s_pert, edgecolor=color, facecolor='none', linewidths=edge, zorder=2,
                       label=entry['pert_label'])

            output_keys = sorted(
                (k for k in comps if k.startswith('output')),
                key=lambda k: int(k[len('output'):])
            )
            for key in output_keys:
                c = comps[key]
                if c['contrast_alpha'] is not None:
                    ax.plot(c['x'], c['y'], alpha=c['contrast_alpha'], lw=1.015 * lw, color='k',
                            label=c['label'], solid_capstyle='round')
                ax.plot(c['x'], c['y'], alpha=c['line_alpha'], lw=lw, color=color,
                        label=c['label'], solid_capstyle='round')
                if c['dots_alpha'] is not None:
                    ax.scatter(c['dots_x'], c['dots_y'], color=color, alpha=c['dots_alpha'],
                               marker='o', s=add_dots, edgecolor='k', linewidths=0.35, zorder=4)

        ax.legend(fontsize=fontsize, labelspacing=1)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if no_axes:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_aspect('equal')

        if ax is None:
            return fig
        else:
            return ax

    params_path = os.path.join(folder, "params.json")

    exp = Experiment2(folder, '0', load=True)
    with open(params_path, 'r') as f:
        params = json.load(f)
    reaching = exp.ReachingDataset
    reaching.dataset_parameters['device'] = 'cpu'
    reaching.apply_perturbation(params['perturbation_params'])

    inputs, targets = reaching.generate_perturbed_dataset('cart')
    if plot_unpert:
        _, un_targets = reaching.generate_unperturbed_dataset('cart')
    else:
        un_targets = None
    in_features = reaching.input_size[2]
    n_tasks = inputs.shape[1]
    rnn_hidden_size = params['rnn_hidden_size']
    net_params = params['net_params']
    net_params['device'] = 'cpu'
    model_state = params['model_state']

    args = [in_features, rnn_hidden_size]
    model = CerebelloCorticalNet(*args, **net_params)
    model.eval(**model_state)

    if isinstance(iterations, bool) and iterations == True:
        weights = [exp.train_outputs['parameters'][plot_epoch]]
        for i in range(1, params['iterations']):
            exp_iter = Experiment2(folder, str(i), load=True)
            weights.append(exp_iter.train_outputs['parameters'][plot_epoch].copy())
            del exp_iter
    else:
        assert isinstance(iterations, int) and iterations < params['iterations']
        exp_iter = Experiment2(folder, str(iterations), load=True)
        weights = [exp_iter.train_outputs['parameters'][plot_epoch].copy()]
    del exp

    # Model evaluation
    model.eval(**model_state)

    # Generate outputs
    outputs = []
    for w in weights:
        model.load_state_dict(w)
        model.eval(**model_state)
        outputs.append(model(inputs).detach().numpy())

    # Settings
    colors = cm.rainbow(np.linspace(0, 1, inputs.shape[1])) if not colors else colors
    if ax is None:
        figsize = (10, 7) if figsize is None else figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if title:
        ax.set_title(title, fontsize=fontsize)

    if empty_endpoints:
        markeredgecolors = colors
        markerfacecolors = {i: 'none' for i in range(n_tasks)}
        s_unpert = markersize
        s_pert = markersize
    else:
        markeredgecolors = ['k' for i in range(n_tasks)]
        markerfacecolors = colors.copy()
        s_unpert = markersize
        s_pert = markersize

    # Plot
    for task in range(n_tasks - 1, -1, -1):
        if plot_unpert:
            ax.scatter(un_targets[-1, task, 0], un_targets[-1, task, 1], alpha=0.4,
                       marker="^", s=s_unpert, facecolor='none', edgecolor=markeredgecolors[task], 
                       linewidths=edge, zorder=2, label=f'Initial targets' if task == 1 else None)

        # plotting perturbed targets
        ax.scatter(targets[-1, task, 0], targets[-1, task, 1], color=colors[task], alpha=alpha_targets,
                   marker=marker, s=s_pert, edgecolor=markeredgecolors[task],
                   facecolor=markerfacecolors[task], linewidths=edge, zorder=2,
                   label=f'Perturbed targets' if task == 1 else None)
        components = []
        for j, output in enumerate(outputs):
            output_label = 'Output trajectory' if (task == 1 and j == 0) else None
            contrast_alpha_j = None
            if contrast:
                contrast_alpha_j = alpha
                ax.plot(output[start:stop, task, 0], output[start:stop, task, 1], alpha=alpha,
                        lw=1.015 * lw, color='k',
                        label=output_label,
                        solid_capstyle='round')
                alpha = 1
            line_alpha_j = alpha
            ax.plot(output[start:stop, task, 0], output[start:stop, task, 1], alpha=alpha,
                    lw=lw, color=colors[task],
                    label=output_label,
                    solid_capstyle='round')
            dots_alpha_j = None
            dots_x_j = None
            dots_y_j = None
            if add_dots:
                dots_alpha_j = alpha
                dots_x_j = float(output[-1, task, 0])
                dots_y_j = float(output[-1, task, 1])
                ax.scatter(output[-1, task, 0], output[-1, task, 1], color=colors[task], alpha=alpha,
                           marker='o', s=add_dots, edgecolor='k', linewidths=0.35, zorder=4)
            if capture is not None:
                components.append({
                    'component': f'output{j}', 'x': output[start:stop, task, 0].tolist(), 'y': output[start:stop, task, 1].tolist(),
                    'label': output_label, 'contrast_alpha': contrast_alpha_j,
                    'line_alpha': line_alpha_j, 'dots_alpha': dots_alpha_j,
                    'dots_x': dots_x_j, 'dots_y': dots_y_j,
                })
        if capture is not None:
            if plot_unpert:
                components.append({'component': 'unperturbed_target', 'x': [float(un_targets[-1, task, 0])], 'y': [float(un_targets[-1, task, 1])]})
            components.append({'component': 'perturbed_target', 'x': [float(targets[-1, task, 0])], 'y': [float(targets[-1, task, 1])]})
            task_color = colors[task]
            capture.append({
                'name': f'{title}_task{task}', 'components': components,
                'color': task_color.tolist() if hasattr(task_color, 'tolist') else task_color,
                'unpert_label': (f'Initial targets' if task == 1 else None),
                'pert_label': (f'Perturbed targets' if task == 1 else None),
            })
    ax.legend(fontsize=fontsize, labelspacing=1)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if no_axes:
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect('equal')

    if ax is None:
        return fig
    else:
        return ax
    


# %% [markdown]
# # data preprocessing

# %%
_fixed_cortex_data_loaded = None

def get_fixed_cortex_data():
    global _fixed_cortex_data_loaded
    if _fixed_cortex_data_loaded is None:
        mse_data = load_metrics(
            load_folder,
            norm=True,
            norm_mode='normal',
            update=UPDATE_METRICS,
            debug=True,
        )

        names = mse_data.index.get_level_values('name')

        mse_list_CCnet = mse_data.loc[
            np.array([
                ('FxCx.' in name) and ('cb_lr.' in name)
                for name in names
            ])
        ].copy()

        mse_list_RNN = mse_data.loc[
            np.array([
                ('RNN.faster_decay.cx_lr.' in name)
                for name in names
            ])
        ].copy()

        _fixed_cortex_data_loaded = (mse_list_CCnet, mse_list_RNN)
    return _fixed_cortex_data_loaded

# %% [markdown]
# # Figure 3c

# %%
def figure_3c(config_folder, avg_mode='mean',
              lr='1e-04', RNN_lr='1e-04', label_names=None, lws=None, colors=None,
              ls='type', figsize=(6, 3), fig_dpi=400, steps=1, smoothing_window=5,
              start=0, stop=4000, inset=False, stop_inset=500,
              inset_specs=(0.45, 0.6, 0.40, 0.38), in_yticks=(1e-6, 1e-4, 1e-2),
              in_xticks=(0, 250, 500), in_ylims=(4e-7, 1.5e-2), groups_size=6,
              lims=(4e-8, 1e-1), fill=True, error_bars=False, smoothing='moving_median',
              contrast=False, plot_losses_dpi=100, return_fig=True, debug=False,
              plot_iters=False, markersize=7, fontsize=16, alpha=0.42, alpha_fill=0.2,
              fill_smoothing=(10, 10), x_ticks=(0, 1000, 2000, 3000, 4000),
              y_ticks=(1e-7, 1e-5, 1e-3, 1e-1), bbox_to_anchor=(0.88, 0.92)):
    def prepare():
        mse_list_CCnet, mse_list_RNN = get_fixed_cortex_data()
        condition = np.array([lr in name for name in mse_list_CCnet.index.get_level_values('name')])
        list_CCnet = mse_list_CCnet.loc[condition]

        condition = np.array([RNN_lr in name for name in mse_list_RNN.index.get_level_values('name')])
        list_RNN = mse_list_RNN.loc[condition]
        condition = np.array([metric == 'mse' for metric in list_RNN.index.get_level_values('metric')])
        list_RNN = list_RNN.loc[condition]

        metrics_list = pd.concat([list_CCnet, list_RNN])

        spec = {
            'figuretype': 'plot_loss_curves',
            'meta': {
                'avg_mode': avg_mode, 'lr': lr, 'RNN_lr': RNN_lr, 'lims': lims,
                'fontsize': fontsize, 'fig_dpi': fig_dpi,
            },
            'metrics_list': metrics_list,
            'config_folder': config_folder,
            'plot_kwargs': dict(
                steps=steps, smoothing_window=smoothing_window, start=start, stop=stop,
                inset=inset, stop_inset=stop_inset, inset_specs=list(inset_specs),
                in_yticks=list(in_yticks), in_xticks=list(in_xticks), in_ylims=list(in_ylims),
                groups_size=groups_size, lims=lims, label_names=label_names, colors=colors,
                ls=ls, title=None, fill=fill, error_bars=error_bars, average_mode=avg_mode,
                lws=lws, figure=False, legend=True, smoothing=smoothing, contrast=contrast,
                dpi=plot_losses_dpi, return_fig=return_fig, debug=debug, plot_iters=plot_iters,
                markersize=markersize, fontsize=fontsize, alpha=alpha, alpha_fill=alpha_fill,
                fill_smoothing=fill_smoothing, x_ticks=list(x_ticks), y_ticks=list(y_ticks),
                bbox_to_anchor=list(bbox_to_anchor),
            ),
            'panels': [{'metric': 'mse', 'series': []}],
        }
        return spec

    def render(spec):
        metrics_list = spec.get('metrics_list')
        building = metrics_list is not None
        if building:
            spec['panels'][0]['series'] = []
        fig = plt.figure(figsize=figsize, dpi=fig_dpi)
        if building:
            plot_loss_curves(metrics_list, spec['config_folder'], **spec['plot_kwargs'],
                          capture=spec['panels'][0]['series'])
        else:
            plot_loss_curves(**spec['plot_kwargs'], replay=spec['panels'][0]['series'])
        plt.tight_layout()
        if building:
            del spec['metrics_list']
        return fig, plt.gca()

    spec = load_or_build_spec("3c", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
label_names = {
    'x.CbFb.c':'',
    'x.CbRo.c': '',
    # 'x.CbRo.CbFb.l': 'Distributed Learning model',
    'x.CbRo.CbFb.PrGrad.c': '',
    'RNN': 'plastic controller'
}
lws = {
    'x.CbRo.c': 0.9,
    'x.CbFb.c': 0.8,
    'x.CbRo.CbFb.c': 1,
    'x.CbRo.CbFb.PrGrad.c': 0.9,
    'RNN': 0.9
}
colors = {
    'x.CbRo.c': '#3582e0ff',
    'x.CbFb.c': '#be9a15ff',
    'x.CbRo.CbFb.PrGrad.c': '#972538ff',
    'x.CbRo.CbFb.c': 'C2',
    'RNN': 'k'
}

fig, spec = figure_3c(
    config_folder=config_folder,
    avg_mode="mean", 
    lr='1e-04', 
    RNN_lr='1e-04',
    label_names=label_names, 
    lws=lws, 
    colors=colors, 
    ls='type'
)
save_plot(
    fig, 
    "3c", 
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


# %% [markdown]
# # Comparison fixing the RNN model and plotting each configuration separately, at different learning rates

# %% [markdown]
# ### S6a

# %%
def figure_systematic_lr(RNN_lr='1e-04', figsize=(10, 6),
                          lims=(1e-8, 0.1), dpi=800, stop=1000, median_w=60,
                          inset_specs=(0.1, 0.6, 0.4, 0.38), step=1, stop_inset=250, lw=1.6,
                          alpha=0.8, mode='mean', fontsize=16, base_colors=None,
                          label_names=None, highlight=False,
                          settings=('x.CbRo.c', 'x.CbFb.c', 'x.CbRo.CbFb.c', 'x.CbRo.CbFb.PrGrad.c')):
    def prepare():
        mse_list_CCnet, mse_list_RNN = get_fixed_cortex_data()
        condition = np.array([RNN_lr in name for name in mse_list_RNN.index.get_level_values('name')])
        list_RNN = mse_list_RNN.loc[condition]
        condition = np.array([metric == 'mse' for metric in list_RNN.index.get_level_values('metric')])
        list_RNN = list_RNN.loc[condition]

        rnn_name = str(list_RNN.index.get_level_values('name').unique()[0])

        panels = []
        for setting in settings:
            title = label_names[setting]

            selection = mse_list_CCnet.copy()
            condition = np.array([setting in name for name in selection.index.get_level_values('name')])
            selection = selection.loc[condition]
            names = selection.index.get_level_values('name').unique()
            lr_list = [name.split('.')[-1] for name in names]
            index_vars = np.argsort([float(lr) for lr in lr_list])[::-1]
            names = names[index_vars]
            labels = {name: f"lr = {name.split('.')[-1]}" for name in names}
            labels[rnn_name] = 'RNN'

            base_color = base_colors[setting]
            colors = {name: to_rgba(base_color, alpha=[1, 0.60, 0.4, 0.25][i]) for i, name in enumerate(names)}
            colors[rnn_name] = to_rgba(base_colors['RNN'])
            panel_selection = pd.concat([selection, list_RNN])

            panels.append({
                'metric': setting, 'title': title, 'selection': panel_selection,
                'labels': labels, 'colors': colors, 'lr_list': lr_list, 'series': [],
            })

        spec = {
            'figuretype': 'systematic_lr',
            'meta': {
                'RNN_lr': RNN_lr, 'lims': lims, 'stop': stop, 'median_w': median_w, 'mode': mode,
                'fontsize': fontsize, 'highlight': highlight,
            },
            'rnn_name': rnn_name,
            'settings': list(settings),
            'panels': panels,
        }
        return spec

    def render(spec):
        building = any('selection' in panel for panel in spec['panels'])
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3, width_ratios=[1, 1, 0.2], height_ratios=[1, 1])

        axes = [fig.add_subplot(gs[0, 0])]
        handles = None
        for j, panel in enumerate(spec['panels']):
            if j != 0:
                axes.append(fig.add_subplot(gs[j // 2, j % 2], sharex=axes[0], sharey=axes[0]))
            ax = axes[j]

            if building:
                panel['series'] = []
                ax, handles = plot_comparison_partial(
                    panel['selection'], ax, stop=spec['meta']['stop'], mode=spec['meta']['mode'],
                    lims=spec['meta']['lims'], colors=panel['colors'], contrast=False,
                    labels=panel['labels'], alpha=alpha, title=panel['title'], median_w=spec['meta']['median_w'],
                    fontsize=spec['meta']['fontsize'], lw=lw, step=step, inset=False, lr_list=panel['lr_list'],
                    inset_specs=list(inset_specs), stop_inset=stop_inset, legend_loc='upper right',
                    highlight='lr.1e-04' if spec['meta']['highlight'] else None,
                    legend=False, inset_lims=(2e-7, 5e-2), capture=panel['series'],
                )
            else:
                ax, handles = plot_comparison_partial(
                    ax=ax, stop=spec['meta']['stop'], mode=spec['meta']['mode'],
                    lims=spec['meta']['lims'], contrast=False,
                    title=panel['title'], fontsize=spec['meta']['fontsize'], lw=lw, step=step, inset=False,
                    inset_specs=list(inset_specs), stop_inset=stop_inset, legend_loc='upper right',
                    highlight='lr.1e-04' if spec['meta']['highlight'] else None,
                    legend=False, inset_lims=(2e-7, 5e-2), replay=panel['series'],
                )

            ax.set_xlabel('epoch', fontsize=spec['meta']['fontsize'])
            ax.set_ylabel('error (adaptation)', fontsize=spec['meta']['fontsize'])
            if j < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel('')
            if (j + 1) % 2 == 0:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_ylabel('')

        legend_ax = fig.add_subplot(gs[:, 2])  # this spans both rows in the last column
        legend_ax.axis('off')  # hide the axis box
        legend_ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.504, 0.54),
                          fontsize=0.9 * spec['meta']['fontsize'])

        if building:
            for panel in spec['panels']:
                panel.pop('selection', None)
                panel.pop('labels', None)
                panel.pop('colors', None)
                panel.pop('lr_list', None)

        return fig, axes

    spec = load_or_build_spec("S6a", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
base_colors = {
    'x.CbRo.c': '#3582e0ff',
    'x.CbFb.c': '#be9a15ff',
    'x.CbRo.CbFb.PrGrad.c': '#972538ff',
    'x.CbRo.CbFb.c': 'C2',
    'RNN': 'k'
}
label_names = {
    'x.CbFb.c':'feedback',
    'x.CbRo.c': 'parallel',
    'x.CbRo.CbFb.PrGrad.c': 'hybrid (private)',
    'x.CbRo.CbFb.c': 'hybrid',
    'RNN': 'plastic controller'
}

fig, spec = figure_systematic_lr(
    base_colors=base_colors,
    label_names=label_names, 
    highlight=False
)
save_plot(
    fig, 
    "S6a", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder, 
    dpi=800,
    transparent=True, 
    bbox_inches='tight'
)
plt.show()
plt.close(fig)


# %% [markdown]
# # Error bars plot

# %%
def plot_model_errorbars(ax, mse_data=None, models=None, epoch=None, base_colors=None, label_names=None,
                          fontsize=14,elinewidth=0.5,xlim=None, ylims=None, yticks=None,
                          marker_size=8, line_width=2, capsize=5, capthick=4,
                          minorytickon = True,
                          mode='mean', bar_mode='minmax',
                          jitter_points=False,
                          jitter_scale=0.04,
                          points_alpha=0.4,
                          points_size=None,
                          show_summary_line=True,
                          capture=None, replay=None):
    """
    Generates an error bars plot summarizing the MSE for multiple models grouped by learning rate.
    For each learning rate (from models) the mean MSE is computed and error bars extend to the 
    minimum and maximum values. Each model is represented by a line.
    
    Args:
        mse_data (pd.DataFrame): DataFrame containing MSE values, indexed by model name and learning rates.
        models (list): List of model settings to plot.
        epoch (int): Epoch index for which to plot the MSE.
        base_colors (dict): Dictionary mapping model settings to base colors (e.g. "#1f77b4").
        label_names (dict): Dictionary mapping model settings to human-readable labels.
        figsize (tuple): Figure size.
        fontsize (int): Font size for all text in the figure.
        dpi (int): DPI for the figure.
        title (str, optional): Title of the plot.
        ylims (tuple, optional): Limits for the y-axis.
        yticks (list, optional): Custom y-axis tick labels.
        marker_size (int, optional): Size of the markers.
        line_width (int, optional): Width of the connecting lines.
        capsize (int, optional): Size of the error bar caps.
    
    Returns:
        fig: The matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl

    # Force math text (used in tick labels) to use Arial.
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Arial'
    mpl.rcParams['mathtext.it'] = 'Arial:italic'
    mpl.rcParams['mathtext.bf'] = 'Arial:bold'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['mathtext.default'] = 'regular'  # force math text to use the regular (non-TeX) text engine

    if points_size is None:
        points_size = marker_size**2

    if replay is not None:
        assert mode in ('points+mean', 'points+median') and show_summary_line and not jitter_points, \
            "plot_model_errorbars replay only supports mode='points+mean'/'points+median', show_summary_line=True, jitter_points=False (the only path exercised by this script's actual call sites)"

        for entry in replay:
            is_rnn = entry['name'] == 'RNN'
            for x, vals in zip(entry['x'], entry['raw_y']):
                vals = np.asarray(vals)
                ax.scatter(np.full(len(vals), x), vals, s=points_size, color=entry['color'],
                           alpha=points_alpha, edgecolors='none', zorder=2, label=None)
            ax.plot(
                entry['x'], entry['y'], '-o', color=entry['color'], markersize=marker_size,
                linewidth=line_width, linestyle='solid' if not is_rnn else 'dashed', zorder=3,
                label=entry['label'] if is_rnn else None,
            )

        sorted_lr = sorted(set().union(*[set(e['x']) for e in replay]), key=float) if replay else []

        ax.set_xscale('log')
        ax.set_yscale('log')

        xtick_positions = []
        xtick_labels = []
        for lr in sorted_lr:
            try:
                xtick_positions.append(float(lr))
                xtick_labels.append(float(lr))
            except ValueError:
                continue

        ax.set_xticks(xtick_positions)
        xtick_labels = [f'{xtick:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+') for xtick in xtick_labels]
        ax.set_xticklabels(xtick_labels, fontname='Arial', fontsize=fontsize)

        ax.set_ylabel(f'error after {epoch} epochs', fontname='Arial', fontsize=fontsize)
        ax.set_xlabel('learning rate', fontname='Arial', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if not minorytickon:
            ax.yaxis.set_tick_params(which='minor', left=False, right=False)

        if yticks is not None:
            yticks_labels = [f'{ytick:.0e}' for ytick in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks_labels, fontname='Arial', fontsize=fontsize)
        elif ylims:
            ax.set_ylim(ylims)
            lower_exp = int(np.ceil(np.log10(ylims[0])))
            upper_exp = int(np.floor(np.log10(ylims[1])))
            ticks = [10**i for i in range(lower_exp, upper_exp + 1)]
            ax.set_yticks(ticks)
            tick_labels = [rf'$10^{{{i}}}$' for i in range(lower_exp, upper_exp + 1)]
            ax.set_yticklabels(tick_labels, fontname='Arial', fontsize=fontsize)

        for label in ax.get_yticklabels():
            label.set_fontname('Arial')

        if xlim:
            ax.set_xlim(*xlim)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.3)
        ax.spines['left'].set_linewidth(1.3)
        ax.tick_params(axis='x', which='major', width=1.3, length=4)
        ax.tick_params(axis='y', which='major', width=1.3, length=4)

        leg = ax.legend(fontsize=fontsize, loc='best')
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontname('Arial')

        plt.tight_layout()
        return

    def maybe_jitter_log_x(x, n, scale=0.04):
        if not jitter_points or n <= 1:
            return np.full(n, x)
        offsets = np.linspace(-scale, scale, n)
        return x * (10 ** offsets)

    # Group data by learning rate and model.
    data_by_lr_and_model = {}
    for setting in models:
        model_data = mse_data.loc[
            [setting in name for name in mse_data.index.get_level_values('name')]
        ]

        if 'metric' in model_data.index.names:
            unique_metrics = model_data.index.get_level_values('metric').unique()
            if 'mse' in unique_metrics and len(unique_metrics) > 1:
                model_data = model_data.loc[(slice(None), slice(None), slice(None), 'mse'), :]

        unique_names = model_data.index.get_level_values('name').unique()
        lr_list_for_model = sorted(
            {name.split('.')[-1] for name in unique_names},
            key=lambda x: float(x)
        )

        for lr in lr_list_for_model:
            lr_rows = model_data.loc[
                [lr in name for name in model_data.index.get_level_values('name')]
            ]
            epoch_vals = lr_rows.values[:, epoch]

            center_val = np.mean(epoch_vals) if 'mean' in mode else np.median(epoch_vals)
            data_by_lr_and_model.setdefault(lr, {})[setting] = (center_val, epoch_vals)

    sorted_lr = sorted(data_by_lr_and_model.keys(), key=lambda x: float(x))

    for setting in models:
        x_vals, centers, lower_err, upper_err = [], [], [], []
        raw_epoch_vals = []

        for lr in sorted_lr:
            group = data_by_lr_and_model.get(lr, {})
            if setting not in group:
                continue

            center_val, epoch_vals = group[setting]
            try:
                x = float(lr)
            except ValueError:
                continue

            if mode in ['points', 'points+mean', 'points+median']:
                x_pts = maybe_jitter_log_x(x, len(epoch_vals), jitter_scale)
                ax.scatter(
                    x_pts,
                    epoch_vals,
                    s=points_size,
                    color=base_colors[setting],
                    alpha=points_alpha,
                    edgecolors='none',
                    zorder=2,
                    label=label_names[setting] if (lr == sorted_lr[0] and not show_summary_line) else None
                )

            x_vals.append(x)
            centers.append(center_val)
            raw_epoch_vals.append(epoch_vals)
            if bar_mode == 'std':
                lower_err.append(np.std(epoch_vals))
                upper_err.append(np.std(epoch_vals))
            else:
                lower_err.append(center_val - np.min(epoch_vals))
                upper_err.append(np.max(epoch_vals) - center_val)

        if not x_vals:
            continue

        if capture is not None:
            capture.append({
                'name': setting, 'label': label_names.get(setting), 'color': base_colors[setting],
                'x': list(x_vals), 'y': list(centers),
                'raw_y': [v.tolist() for v in raw_epoch_vals],
            })

        if mode in ['mean', 'median']:
            ax.errorbar(
                x_vals, centers,
                yerr=[lower_err, upper_err],
                fmt='-o',
                color=base_colors[setting],
                markersize=marker_size,
                linewidth=line_width,
                capsize=capsize,
                capthick=capthick,
                elinewidth=elinewidth,
                linestyle='solid' if setting != 'RNN' else 'dashed',
                label=label_names[setting] if setting == 'RNN' else None
            )

        elif mode in ['points+mean', 'points+median'] and show_summary_line:
            ax.plot(
                x_vals, centers,
                '-o',
                color=base_colors[setting],
                markersize=marker_size,
                linewidth=line_width,
                linestyle='solid' if setting != 'RNN' else 'dashed',
                zorder=3,
                label=label_names[setting] if setting == 'RNN' else None
            )

    ax.set_xscale('log')
    ax.set_yscale('log')

    xtick_positions = []
    xtick_labels = []
    for lr in sorted_lr:
        try:
            xtick_positions.append(float(lr))
            xtick_labels.append(float(lr))
        except ValueError:
            continue

    ax.set_xticks(xtick_positions)
    xtick_labels = [f'{xtick:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+') for xtick in xtick_labels]
    ax.set_xticklabels(xtick_labels, fontname='Arial', fontsize=fontsize)

    ax.set_ylabel(f'error after {epoch} epochs', fontname='Arial', fontsize=fontsize)
    ax.set_xlabel('learning rate', fontname='Arial', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if not minorytickon:
        ax.yaxis.set_tick_params(which='minor', left=False, right=False)

    if yticks is not None:
        yticks_labels = [f'{ytick:.0e}' for ytick in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels, fontname='Arial', fontsize=fontsize)
    elif ylims:
        ax.set_ylim(ylims)
        lower_exp = int(np.ceil(np.log10(ylims[0])))
        upper_exp = int(np.floor(np.log10(ylims[1])))
        ticks = [10**i for i in range(lower_exp, upper_exp + 1)]
        ax.set_yticks(ticks)
        tick_labels = [rf'$10^{{{i}}}$' for i in range(lower_exp, upper_exp + 1)]
        ax.set_yticklabels(tick_labels, fontname='Arial', fontsize=fontsize)

    for label in ax.get_yticklabels():
        label.set_fontname('Arial')

    if xlim:
        ax.set_xlim(*xlim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.tick_params(axis='x', which='major', width=1.3, length=4)
    ax.tick_params(axis='y', which='major', width=1.3, length=4)

    leg = ax.legend(fontsize=fontsize, loc='best')
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontname('Arial')

    plt.tight_layout()
    
def figure_comparison_lrs(models, label_names, colors, mode='points+median',
                           bar_mode='minmax', epochs=(200, 1000), ylims=None, xlim=(3e-5, 0.0015),
                           title=True, figsize=(12, 5), dpi=800, fontsize=22, elinewidth=1.2,
                           marker_size=6, line_width=1.8, capsize=5, capthick=1,minorytickon=True,
                           spec_name=None):
    def prepare():
        mse_list_CCnet, mse_list_RNN = get_fixed_cortex_data()
        condition = np.array(['decay' in name for name in mse_list_RNN.index.get_level_values('name')])
        list_RNN = mse_list_RNN.loc[condition]
        condition = np.array([metric=='mse' for metric in list_RNN.index.get_level_values('metric')])
        list_RNN = list_RNN.loc[condition]
        mse_df = pd.concat([mse_list_CCnet, list_RNN])

        spec = {
            'figuretype': 'plot_model_errorbars',
            'meta': {
                'mode': mode, 'bar_mode': bar_mode, 'epochs': list(epochs), 'ylims': ylims,
                'xlim': xlim, 'fontsize': fontsize,
            },
            'mse_df': mse_df,
            'models': models,
            'label_names': label_names,
            'colors': colors,
            'panels': [{'metric': str(epoch), 'series': []} for epoch in epochs],
        }
        return spec

    def render(spec):
        mse_df = spec.get('mse_df')
        building = mse_df is not None
        fig, axs = plt.subplots(1, len(epochs), figsize=figsize, dpi=dpi)
        axs = axs if len(epochs)>1 else [axs]
        for i, epoch in enumerate(spec['meta']['epochs']):
            if title:
                axs[i].set_title(f'error after {epoch} epochs', fontsize=fontsize * 1.2)
            if building:
                spec['panels'][i]['series'] = []
                plot_model_errorbars(
                    axs[i], mse_df, spec['models'], epoch, spec['colors'], spec['label_names'],
                    fontsize=fontsize, elinewidth=elinewidth, xlim=spec['meta']['xlim'],minorytickon=minorytickon,
                    bar_mode=spec['meta']['bar_mode'], ylims=spec['meta']['ylims'][epoch], yticks=None,
                    marker_size=marker_size, line_width=line_width, capsize=capsize, capthick=capthick,
                    mode=spec['meta']['mode'], capture=spec['panels'][i]['series'],
                )
            else:
                plot_model_errorbars(
                    axs[i], epoch=epoch,
                    fontsize=fontsize, elinewidth=elinewidth, xlim=spec['meta']['xlim'],minorytickon=minorytickon,
                    bar_mode=spec['meta']['bar_mode'], ylims=spec['meta']['ylims'][epoch], yticks=None,
                    marker_size=marker_size, line_width=line_width, capsize=capsize, capthick=capthick,
                    mode=spec['meta']['mode'], replay=spec['panels'][i]['series'],
                )
        if building:
            del spec['mse_df']
        return fig, axs

    spec = load_or_build_spec(spec_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


label_names = {
    'x.CbFb.c':'',
    'x.CbRo.c': '',
    'x.CbRo.CbFb.l': 'Distributed Learning model',
    'x.CbRo.CbFb.PrGrad.c': '',
    'RNN': 'plastic controller'
}
colors = {
    'x.CbRo.c': '#3582e0ff',
    'x.CbFb.c': '#be9a15ff',
    'x.CbRo.CbFb.PrGrad.c': '#972538ff',
    'x.CbRo.CbFb.c': 'C2',
    'RNN': 'k'
}

# %% [markdown]
# ### 3d

# %%

ylims={
    200: (8e-7, 5e-3),
    1000: (1e-8, 1e-4)
}

fig, spec = figure_comparison_lrs(
    models=['x.CbFb.c', 'x.CbRo.c', 'x.CbRo.CbFb.PrGrad.c', 'RNN'],
    label_names=label_names,
    colors=colors,
    epochs=[200],
    figsize=(5.8,5),
    mode='points+median',
    ylims=ylims,
    spec_name="3d",
)
save_plot(
    fig, 
    "3d", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder, 
    dpi=800,
    transparent=True, 
    bbox_inches='tight'
)
plt.show()
plt.close(fig)


# %% [markdown]
# ### S6b

# %%
ylims={
    200: (1e-8, 8e-3),
    1000: (1e-8, 8e-3)
}

fig, spec = figure_comparison_lrs(
    models=['x.CbFb.c', 'x.CbRo.c', 'x.CbRo.CbFb.c', 'x.CbRo.CbFb.PrGrad.c', 'RNN'],
    label_names=label_names,
    colors=colors,
    epochs=(200, 1000),
    figsize=(12,5),
    mode='points+median',
    ylims=ylims,
    minorytickon=False,
    spec_name="S6b",
)
save_plot(
    fig, 
    "S6b", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder, 
    dpi=800,
    transparent=True, 
    bbox_inches='tight'
)
plt.show()
plt.close(fig)


# %% [markdown]
# # Final outputs

# %%
def figure_outputs(data_folder, labels, plot_epoch, skip_folders=('2.FxCx.CbRo.CbFb',), figsize=(24, 8),
                    dpi=500, iterations=1, plot_unpert=True, alpha_targets=1, empty_endpoints=True,
                    marker='s', colors=None, inner_dpi=100, markersize=300, edge=2.7, contrast=True,
                    alpha=0.9, x_lim=(-0.120, 0.175), y_lim=(None, None), start=0, lw=3, add_dots=120,
                    stop=-1, fontsize=18, no_axes=True):
    def prepare():
        folders = []
        for folder in os.listdir(data_folder):
            digit = folder.split('.')[0]
            if folder in skip_folders:
                continue
            if digit.isdigit():
                folders.append(folder)

        spec = {
            'figuretype': 'plot_outputs_pre',
            'meta': {'plot_epoch': plot_epoch, 'labels': labels, 'epoch': plot_epoch * 10},
            'data_folder': data_folder,
            'folders': folders,
            'panels': [{'metric': 'trajectory', 'series': []}],
        }
        return spec

    def render(spec):
        building = len(spec['panels'][0]['series']) == 0
        fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        if building:
            spec['panels'][0]['series'] = []
        for i, (folder, name) in enumerate(spec['meta']['labels'].items()):
            if building:
                sim_folder = os.path.join(spec['data_folder'], folder)
                ax = plot_outputs_pre(sim_folder, iterations=iterations, plot_epoch=plot_epoch,
                                        plot_unpert=plot_unpert, ax=axs[i], title=name,
                                        alpha_targets=alpha_targets, empty_endpoints=empty_endpoints,
                                        marker=marker, colors=colors, figsize=None, dpi=inner_dpi,
                                        markersize=markersize, edge=edge, contrast=contrast, alpha=alpha,
                                        x_lim=x_lim, y_lim=y_lim, start=start, lw=lw, add_dots=add_dots,
                                        stop=stop, fontsize=fontsize, no_axes=no_axes,
                                        capture=spec['panels'][0]['series'])
            else:
                folder_entries = [e for e in spec['panels'][0]['series'] if e['name'].startswith(f'{name}_task')]
                ax = plot_outputs_pre(None, plot_epoch=plot_epoch,
                                        plot_unpert=plot_unpert, ax=axs[i], title=name,
                                        alpha_targets=alpha_targets, empty_endpoints=empty_endpoints,
                                        marker=marker, dpi=inner_dpi,
                                        markersize=markersize, edge=edge, contrast=contrast, alpha=alpha,
                                        x_lim=x_lim, y_lim=y_lim, start=start, lw=lw, add_dots=add_dots,
                                        stop=stop, fontsize=fontsize, no_axes=no_axes,
                                        replay=folder_entries)
            ax.set_title(name)
        return fig, axs

    spec = load_or_build_spec("3e", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec


# %%
data_folder = os.path.join(WORKDIR, 'fixed_cortex_outputs')

labels = {
    "0.FxCx.CbRo": "parallel",
    "1.FxCx.CbFb": "feedback",
    "3.FxCx.CbRo.CbFb.PrGrad": "hybrid",
}

fig, spec = figure_outputs(
    data_folder=data_folder, 
    labels=labels,
    add_dots=90,
    plot_epoch=20 # (this is the saving index, models were saved every 10 epochs)
)

save_plot(
    fig, 
    f"3e", 
    plots_folder, 
    savepdf=save_pdf, 
    savetiff=save_tiff,
    savedata=save_data, 
    data=spec, 
    plotsdatafolder=plots_data_folder, 
    dpi=500,
    transparent=True, 
    bbox_inches='tight'
)

plt.show()
plt.close(fig)


# %%
