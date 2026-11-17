# %%
"""Figure generation for the damage experiments."""
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

WORKDIR = str(DATA_DIR)
load_folder = os.path.join(WORKDIR, 'damage')

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
from matplotlib.gridspec import GridSpec

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

# %%
def load_damage_data(load_folder, update=False):
    cache_path = os.path.join(DATA_DIR, 'metrics', 'damage', 'data.pkl')
    if not update and os.path.exists(cache_path):
        return load_var(cache_path)

    skip_folders = {
            'completed_jobs', 'scripts',
            'submitted_jobs', 'snapshot', 'configs', 'trainings_metrics',
        }

    sim_folders = [
            file for file in sorted(os.listdir(load_folder))
            if file not in skip_folders
            and os.path.isdir(os.path.join(load_folder, file))
        ]

    data = {}
    path = os.path.join(load_folder, "snapshot", "template_config.json")
    with open(path, 'r') as f:
        params = json.load(f)

    iterations = params['iterations']
    damage_epochs_list = params['damage_epochs_list']
    dataset_params = params["dataset_params"]
    reaching = ReachingDataset(dataset_params)
    std = reaching.generate_unperturbed_dataset('cart')[1].std().item()

    for sim in sim_folders:
        parsed = parse_model_name(sim)
        data[parsed] = {}
        damage_epochs = damage_epochs_list if parsed != "control" else [400]
        for damage_epoch in damage_epochs:
            data[parsed][damage_epoch] = {'mse': [], 'cortex_error': [], 'nocb_error': []}
            for it in range(iterations):
                it_folder = os.path.join(load_folder, sim, str(damage_epoch))
                exp = Experiment2(it_folder, str(it), load=True)
                normal = exp.train_outputs["normal"]
                damage_start = 1
                if not normal:
                    normal = {'metrics': {}}
                    damage_start = 0
                else:
                    normal = normal['metrics']
                damage = exp.train_outputs["damage"]["metrics"]
                data[parsed][damage_epoch]['mse'].append(np.concatenate([normal.get('mse',[]), damage.get('mse',[])[damage_start:]])/std)
                data[parsed][damage_epoch]['cortex_error'].append(np.concatenate([normal.get('cortex_error',[]), damage.get('cortex_error',[])[damage_start:]])/std)
                data[parsed][damage_epoch]['nocb_error'].append(np.concatenate([normal.get('nocb_error',[]), damage.get('nocb_error',[])[damage_start:]])/std)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_var(data, cache_path)
    return data

def extract_metric(data_entry, metric):
    """
    data_entry[metric] is a list of N arrays each of shape (T,).
    Returns ndarray of shape (N, T).
    """
    return np.stack(data_entry[metric], axis=0)

def performance(x, minimum, maximum, log=True):
    if log:
        x = np.log(x)
        maximum = np.log(maximum) if maximum else x.max()
        minimum = np.log(minimum) if minimum else x.min()
    return 1 - (x - minimum) / (maximum - minimum)

def plot_ablation(data=None, axs=None, stop=500, mode='mean', start=0, plot_key='mse', colors=None,
                  skip_nocb_loss=False, lw=1.35, nocb_lw=1.35, grid=True, eps=1e-4,
                  y_lims=(-0.05, 1.13), log_metric=False, plot_errors=None, median_w=0,
                  debug=False,
                  ax_color=(50/255, 50/255, 50/255, 100/100),
                  grid_color=(240/255, 240/255, 240/255, 97/100),
                  plot_all_mode='lines', alpha_fill=0.2, ablation_keys=None, capture=None, replay=None):

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerLine2D

    class HandlerVerticalLine(HandlerLine2D):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            l = Line2D([10, 10], [-1, 15], linestyle=orig_handle.get_linestyle(),
                       color=orig_handle.get_color(), lw=orig_handle.get_linewidth(), transform=trans)
            return [l]

    def create_performance_wrapper(minimum, maximum):
        def fixed_performance(x, log=True):
            return performance(x, minimum=minimum, maximum=maximum, log=log)
        return fixed_performance

    assert axs.shape == (2, 2), "Please provide a 2x2 matplotlib.Axes object as 'axs'."

    active_axs = [ax for ax in axs.flatten() if ax.get_visible()]

    colors = ['C3', 'C0', 'C4'] if not colors else colors

    metrics_to_plot = ['mse', 'nocb_error'] if plot_key == 'mse' else [plot_key]
    metrics_labels = (
        {'mse': 'full model', 'nocb_error': 'controller memory'}
        if plot_key == 'mse'
        else {plot_key: plot_key}
    )
    metrics_styles = {'mse': 'solid', 'nocb_error': 'dashed'}

    all_titles = {
        'cereb_ablation':           'adapter ablation',
        'cereb_readout_ablation':   'readout lesion',
        'cereb_plasticity_damage':  "adapter plasticity lockout",
        'cereb_feedback_ablation':  'feedback lesion',
    }

    titles = {k: all_titles[k] for k in ablation_keys} if ablation_keys is not None else all_titles

    if skip_nocb_loss and plot_key == 'mse':
        metrics_labels.pop('nocb_error')

    if replay is not None:
        assert mode == 'mean' and not plot_errors, \
            "plot_ablation replay only supports mode='mean', plot_errors=False (the only paths exercised by this script's actual figures)"

        entries_by_key = {}
        for entry in replay:
            entries_by_key.setdefault(entry['ablation_key'], []).append(entry)

        epoch_color = {}
        for entry in replay:
            if 'epoch' in entry:
                epoch_color[entry['epoch']] = entry['color']
        d_epochs = sorted(epoch_color.keys())

        legend_elements = [
            Line2D([0], [0],
                   linestyle='solid' if metric == plot_key else 'dashed',
                   color='k', label=label)
            for metric, label in metrics_labels.items() if metric in metrics_to_plot
        ]
        legend_elements.append(
            Line2D([0], [0], linestyle='solid', color='k', label='control', lw=1.5 * lw, alpha=0.6)
        )
        for epoch in d_epochs:
            legend_elements.append(
                Line2D([0], [0], linestyle='solid', color=epoch_color[epoch],
                       label=f'epoch {epoch}', lw=lw)
            )

        for i, (ablation_key, title) in enumerate(titles.items()):
            ax = active_axs[i]
            ax.set_title(title)

            for entry in entries_by_key.get(ablation_key, []):
                x = np.array(entry['x'])
                y = np.array(entry['y'])
                if 'epoch' not in entry:
                    nocb = entry['metric'] == 'nocb_error'
                    ax.plot(x, y, color='k',
                            linestyle='dashed' if nocb else 'solid',
                            lw=1.2 * nocb_lw if nocb else 1.25 * lw)
                else:
                    metric = entry['metric']
                    ax.plot(x, y, color=entry['color'], linestyle=metrics_styles[metric],
                            lw=1.2 * lw if metric == plot_key else 1.2 * nocb_lw)
                    if metric == plot_key:
                        ax.annotate(
                            '', xy=(entry['epoch'], 1.03), xytext=(entry['epoch'], 1.16),
                            arrowprops=dict(facecolor=entry['color'], edgecolor=entry['color'],
                                            shrink=0.05, width=0.8, headwidth=5, headlength=5)
                        )

            if grid:
                ax.grid(True, color=grid_color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set(linewidth=0.8, color=ax_color)
            ax.spines['left'].set(linewidth=0.8, color=ax_color)
            ax.set_ylim(*y_lims)
            ax.set_yticks([0.0, 0.5, 1, 0])
            ax.tick_params(axis='both', which='major', width=0.7, length=4.5, colors=ax_color)
            ax.set_ylabel('adaptation', fontsize=12, color=ax_color, labelpad=2)
            ax.set_xlim(-20, stop + 20)
            if i == 3:
                ax.legend(handles=legend_elements, loc='lower right', labelcolor=ax_color,
                          handler_map={Line2D: HandlerVerticalLine()}, bbox_to_anchor=[-0., 0.84])
            ax.set_xlabel('epoch', fontsize=12, color=ax_color, labelpad=2)

        plt.tight_layout()
        return axs

    # --- Damage epoch keys: sorted integers across all ablation types ---
    all_ablation_keys = [k for k in data.keys() if k != 'control']   # ← renamed
    d_epochs = sorted({epoch for k in all_ablation_keys for epoch in data[k].keys()})
    colors_map = {epoch: colors[i] for i, epoch in enumerate(d_epochs)}

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0],
               linestyle='solid' if metric == plot_key else 'dashed',
               color='k', label=label)
        for metric, label in metrics_labels.items() if metric in metrics_to_plot
    ]
    legend_elements.append(
        Line2D([0], [0], linestyle='solid', color='k', label='control', lw=1.5 * lw, alpha=0.6)
    )
    for epoch in d_epochs:
        legend_elements.append(
            Line2D([0], [0], linestyle='solid', color=colors_map[epoch],
                   label=f'epoch {epoch}', lw=lw)
        )

    # --- Control arrays: shape (N, T), sliced to [start:stop] ---
    control_epoch_key = list(data['control'].keys())[0]
    control_data = data['control'][control_epoch_key]
    control_mse    = extract_metric(control_data, 'mse')[:, start:stop]          # (N, stop-start)
    control_metric = control_mse.copy()
    if not skip_nocb_loss:
        control_nocb_error = extract_metric(control_data, 'nocb_error')[:, start:stop]

    maximum = control_mse.max() + eps
    minimum = control_mse.min() - eps
    if debug:
        print(f"max={maximum:.6f}, min={minimum:.6f}")
    fixed_performance = create_performance_wrapper(minimum, maximum)

    x_control = np.arange(start, stop)

    for i, (ablation_key, title) in enumerate(titles.items()):
        ax = active_axs[i]
        ax.set_title(title)

        # --- Control ---
        control_values = fixed_performance(control_metric, log=log_metric)        # (N, T)
        if not skip_nocb_loss:
            control_nocb_values = fixed_performance(control_nocb_error, log=log_metric)

        if mode in ['mean', 'both']:
            control_mse_mean = control_values.mean(axis=0)
            ax.plot(x_control, control_mse_mean, color='k', lw=1.25 * lw)
            if capture is not None:
                capture.append({
                    'name': f'{ablation_key}_control_mse',
                    'label': f'{title} control mse',
                    'ablation_key': ablation_key,
                    'metric': 'mse',
                    'x': x_control.tolist(),
                    'y': control_mse_mean.tolist(),
                })
            if not skip_nocb_loss:
                control_nocb_mean = control_nocb_values.mean(axis=0)
                ax.plot(x_control, control_nocb_mean,
                        color='k', linestyle='dashed', lw=1.2 * nocb_lw)
                if capture is not None:
                    capture.append({
                        'name': f'{ablation_key}_control_nocb_error',
                        'label': f'{title} control nocb_error',
                        'ablation_key': ablation_key,
                        'metric': 'nocb_error',
                        'x': x_control.tolist(),
                        'y': control_nocb_mean.tolist(),
                    })

        if mode in ['all', 'both']:
            if plot_all_mode == 'lines':
                for j in range(control_values.shape[0]):
                    ax.plot(x_control, control_values[j], color='k', lw=0.6 * 1.25 * lw)
                    if not skip_nocb_loss:
                        ax.plot(x_control, control_nocb_values[j],
                                color='k', linestyle='dashed', lw=0.6 * 1.2 * nocb_lw)
            else:
                means = control_values.mean(axis=0)
                maxs  = np.maximum(control_values.max(axis=0), means)
                mins  = np.minimum(control_values.min(axis=0), means)
                ax.fill_between(x_control, mins, maxs,
                                alpha=alpha_fill, facecolor='none', lw=0, color='k', zorder=0)
                if not skip_nocb_loss:
                    means = control_nocb_values.mean(axis=0)
                    maxs  = np.maximum(control_nocb_values.max(axis=0), means)
                    mins  = np.minimum(control_nocb_values.min(axis=0), means)
                    ax.fill_between(x_control, mins, maxs,
                                    alpha=alpha_fill, facecolor='none', lw=0, color='k', zorder=0)

        # --- Ablation curves ---
        ablation_data = data.get(ablation_key, {})

        for epoch in d_epochs:
            if epoch not in ablation_data:
                continue

            epoch_entry = ablation_data[epoch]
            color = colors_map[epoch]
            # timeseries runs from epoch onward; in the array it starts at index (epoch + start)
            values_start = epoch + start
            x_damage = np.arange(epoch + 1, stop)                # x axis (epoch+1 … stop-1)

            for metric, style in metrics_styles.items():
                if skip_nocb_loss and metric == 'nocb_error':
                    continue
                if metric not in epoch_entry:
                    continue

                # (N, stop - values_start)
                values = extract_metric(epoch_entry, metric)[:, values_start:stop]

                if median_w > 0:
                    values = moving_median(values, median_w)

                values = fixed_performance(values, log=log_metric)
                mean   = values.mean(axis=0)

                if plot_errors:
                    assert isinstance(plot_errors, int)
                    mins = np.minimum(values.min(axis=0), mean)
                    maxs = np.maximum(values.max(axis=0), mean)
                    error_epochs = np.arange(epoch + 1, stop, plot_errors, dtype=int)
                    idx = error_epochs - (epoch + 1)
                    ax.errorbar(error_epochs, mean[idx],
                                yerr=[(mean - mins)[idx], (maxs - mean)[idx]],
                                fmt='o', capsize=2, capthick=1, elinewidth=1,
                                color=color, ecolor=color, markersize=1,
                                markerfacecolor=color, alpha=0.8)

                if mode in ['mean', 'both']:
                    ax.plot(x_damage, mean, color=color, linestyle=style,
                            lw=1.2 * lw if metric == plot_key else 1.2 * nocb_lw)
                    if capture is not None:
                        capture.append({
                            'name': f'{ablation_key}_epoch{epoch}_{metric}',
                            'label': f'{title} epoch {epoch} {metric}',
                            'ablation_key': ablation_key,
                            'epoch': epoch,
                            'metric': metric,
                            'x': x_damage.tolist(),
                            'y': mean.tolist(),
                            'color': color,
                        })

                if mode in ['all', 'both']:
                    if plot_all_mode == 'lines':
                        for j in range(values.shape[0]):
                            ax.plot(x_damage, values[j], color=color,
                                    linestyle=style, lw=0.6 * lw)
                    else:
                        means = values.mean(axis=0)
                        maxs  = np.maximum(values.max(axis=0), means)
                        mins  = np.minimum(values.min(axis=0), means)
                        ax.fill_between(x_damage, mins, maxs,
                                        alpha=alpha_fill, facecolor='none', color=color,
                                        linestyle=style, linewidth=0, zorder=0)

                # Arrow at damage onset (only for the primary metric)
                if metric == plot_key:
                    ax.annotate(
                        '',
                        xy=(epoch, 1.03),
                        xytext=(epoch, 1.16),
                        arrowprops=dict(facecolor=color, edgecolor=color,
                                        shrink=0.05, width=0.8, headwidth=5, headlength=5)
                    )

        if grid:
            ax.grid(True, color=grid_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set(linewidth=0.8, color=ax_color)
        ax.spines['left'].set(linewidth=0.8, color=ax_color)
        ax.set_ylim(*y_lims)
        ax.set_yticks([0.0, 0.5, 1, 0])
        ax.tick_params(axis='both', which='major', width=0.7, length=4.5, colors=ax_color)
        ax.set_ylabel('adaptation', fontsize=12, color=ax_color, labelpad=2)
        ax.set_xlim(-20, stop + 20)
        if i == 3:
            ax.legend(handles=legend_elements, loc='lower right', labelcolor=ax_color,
                      handler_map={Line2D: HandlerVerticalLine()},bbox_to_anchor=[-0.,0.84])
        ax.set_xlabel('epoch', fontsize=12, color=ax_color, labelpad=2)

    plt.tight_layout()
    return axs


# %%
_damage_data_loaded = None

def get_damage_data():
    global _damage_data_loaded
    if _damage_data_loaded is None:
        _damage_data_loaded = load_damage_data(load_folder, update=UPDATE_METRICS)
    return _damage_data_loaded

# %% [markdown]
# ## Figure 7

# %%
def figure_Figure_8(epoch_filter={40}, colors=None,
                     ablation_keys=('cereb_ablation', 'cereb_readout_ablation'),
                     plotting_metric='mse', suptitle='mid-only',
                     figsize=(7.2 / (2 + 0.75), 5.5), dpi=400, stop=300, lw=0.8, start=1,
                     alpha_fill=0.15, y_lims=(-0.1, 1.2), log_metric=False, median_w=0,
                     eps=1e-5, plot_all_mode='area',
                     ax_color=(50/255, 50/255, 50/255, 100/100), ticklabelsize=12):
    def prepare():
        data = get_damage_data()
        resolved_colors = [Reds(0.6)] if colors is None else colors
        plot_data = {
            k: ({ep: v for ep, v in epochs.items() if ep in epoch_filter}
                if k != 'control' else epochs)
            for k, epochs in data.items()
        }
        return {
            'figuretype': 'ablation_grid',
            'meta': {
                'plotting_metric': plotting_metric, 'suptitle': suptitle, 'stop': stop,
                'lw': lw, 'start': start, 'alpha_fill': alpha_fill, 'y_lims': y_lims,
                'log_metric': log_metric, 'median_w': median_w, 'eps': eps,
                'plot_all_mode': plot_all_mode, 'ax_color': ax_color, 'colors': resolved_colors,
                'ablation_keys': list(ablation_keys) if ablation_keys is not None else None,
                'ticklabelsize': ticklabelsize,
            },
            'panels': [{'metric': plotting_metric, 'series': []}],
            'plot_data': plot_data,
        }

    def render(spec):
        meta = spec['meta']
        plot_data = spec.get('plot_data')
        building = plot_data is not None
        if building:
            spec['panels'][0]['series'] = []
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig, hspace=0.95)
        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0])
        ax01 = fig.add_axes([0, 0, 0, 0], visible=False)
        ax11 = fig.add_axes([0, 0, 0, 0], visible=False)
        axs = np.array([[ax00, ax01], [ax10, ax11]])

        plt.subplots_adjust(wspace=120, hspace=0.3)
        fig.suptitle(meta['suptitle'], fontsize=16)

        axs = plot_ablation(
            plot_data, axs, stop=meta['stop'], mode='mean', plot_errors=False,
            skip_nocb_loss=False, lw=meta['lw'], nocb_lw=1.5 * meta['lw'], start=meta['start'],
            plot_key=meta['plotting_metric'], alpha_fill=meta['alpha_fill'],
            y_lims=meta['y_lims'], log_metric=meta['log_metric'], median_w=meta['median_w'],
            eps=meta['eps'], colors=meta['colors'], grid=False, ax_color=meta['ax_color'],
            plot_all_mode=meta['plot_all_mode'], ablation_keys=meta['ablation_keys'],
            capture=spec['panels'][0]['series'] if building else None,
            replay=None if building else spec['panels'][0]['series'],
        )
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', labelsize=meta['ticklabelsize'])

        if building:
            del spec['plot_data']

        return fig, axs

    spec = load_or_build_spec("8", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
fig, spec = figure_Figure_8()
save_plot(
    fig,
    "8",
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
# ## Figure S16

# %%
def figure_Figure_14(colors=None, ablation_keys=None, plotting_metric='mse',
                      suptitle='full', figsize=(7.2, 5.5), dpi=400, stop=300, lw=0.8,
                      start=1, alpha_fill=0.15, y_lims=(-0.1, 1.2), log_metric=False,
                      median_w=0, eps=1e-5, plot_all_mode='area',
                      ax_color=(50/255, 50/255, 50/255, 100/100), ticklabelsize=12):
    def prepare():
        data = get_damage_data()
        resolved_colors = [Reds(0.3 + 0.3 * i) for i in range(3)] if colors is None else colors
        return {
            'figuretype': 'ablation_grid',
            'meta': {
                'plotting_metric': plotting_metric, 'suptitle': suptitle, 'stop': stop,
                'lw': lw, 'start': start, 'alpha_fill': alpha_fill, 'y_lims': y_lims,
                'log_metric': log_metric, 'median_w': median_w, 'eps': eps,
                'plot_all_mode': plot_all_mode, 'ax_color': ax_color, 'colors': resolved_colors,
                'ablation_keys': ablation_keys, 'ticklabelsize': ticklabelsize,
            },
            'panels': [{'metric': plotting_metric, 'series': []}],
            'plot_data': data,
        }

    def render(spec):
        meta = spec['meta']
        plot_data = spec.get('plot_data')
        building = plot_data is not None
        if building:
            spec['panels'][0]['series'] = []
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                      figure=fig, hspace=0.95, wspace=0.75)
        axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])

        plt.subplots_adjust(wspace=120, hspace=0.3)
        fig.suptitle(meta['suptitle'], fontsize=16)

        axs = plot_ablation(
            plot_data, axs, stop=meta['stop'], mode='mean', plot_errors=False,
            skip_nocb_loss=False, lw=meta['lw'], nocb_lw=1.5 * meta['lw'], start=meta['start'],
            plot_key=meta['plotting_metric'], alpha_fill=meta['alpha_fill'],
            y_lims=meta['y_lims'], log_metric=meta['log_metric'], median_w=meta['median_w'],
            eps=meta['eps'], colors=meta['colors'], grid=False, ax_color=meta['ax_color'],
            plot_all_mode=meta['plot_all_mode'], ablation_keys=meta['ablation_keys'],
            capture=spec['panels'][0]['series'] if building else None,
            replay=None if building else spec['panels'][0]['series'],
        )
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', labelsize=meta['ticklabelsize'])

        if building:
            del spec['plot_data']

        return fig, axs

    spec = load_or_build_spec("S16", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %%
fig, spec = figure_Figure_14()
save_plot(
    fig,
    "S16",
    plots_folder,
    savepdf=save_pdf,
    savetiff=save_tiff,
    savedata=save_data,
    data=spec,
    plotsdatafolder=plots_data_folder,
    dpi=400,
    transparent=True,
    bbox_inches='tight',
)
plt.show()
plt.close(fig)

# %%
