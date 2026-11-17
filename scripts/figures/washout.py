# %%
# %load_ext autoreload
# %autoreload 2

"""Figure generation for the washout experiments."""


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
load_folder   = os.path.join(DATA_DIR, 'washout')

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

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Reds, Greens
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
def check_data(sim_folders, data_keys):
    missing = []
    for folder_name in sim_folders:
        model_name = parse_model_name(folder_name)
        if model_name not in data_keys:
            missing.append(folder_name)
    return missing


def load_washout_data(data_folder, update=False, metrics=('mse', 'nocb_error'), skip_name=None):

    skip = {'snapshot', 'configs', 'trainings_metrics', 'trainings_metrics_continual'}
    data = {}

    data_path = os.path.join(DATA_DIR, 'metrics', os.path.basename(data_folder), 'data.pkl')
    # Raw-free fast path: only when the raw sim folder is absent (a from-source
    # rebuild). When the folder IS present, fall through so the cache is
    # validated against it and any missing sims are filled in.
    if os.path.exists(data_path) and not update and not os.path.isdir(data_folder):
        data = load_var(data_path)
        print(f"{data_folder}: {len(data)} elements loaded.")
        return data

    template_config = os.path.join(data_folder,"snapshot","template_config.json")
    with open(template_config) as f:
        configs = json.load(f)
        washout_windows = sorted(configs.get("pert_windows",configs.get("washout_windows",None)))
        iterations = configs["iterations"]
        dataset_params = configs["dataset_params"]
        reaching = ReachingDataset(dataset_params)
        std = reaching.generate_unperturbed_dataset('cart')[1].std().item()

    sim_folders = []
    for folder_name in sorted(os.listdir(data_folder)):
        full = os.path.join(data_folder, folder_name)
        if not os.path.isdir(full):
            continue
        if folder_name in skip or folder_name.endswith('.json') or folder_name.endswith('.lock'):
            continue
        if folder_name.startswith('.'):
            continue
        sim_folders.append(folder_name)

    if os.path.exists(data_path):
        if not update:
            data = load_var(data_path)
            missing_data = check_data(sim_folders, data.keys())
            if missing_data == []:
                print(f"{data_folder}: {len(data)} elements loaded.")
                return data
            else:
                print(f"some simulations where missing from saved data, updating (total = {len(missing_data)}).")
                sim_folders = missing_data
        else:
            os.remove(data_path)

    for i,folder_name in enumerate(sim_folders):
        model_folder = os.path.join(data_folder, folder_name)
        if skip_name and skip_name in folder_name:
            print(f"skipping folder {folder_name} ({i+1}/{len(sim_folders)})")
            continue
        model_name = parse_model_name(folder_name)
        print(f"loading folder {model_name} ({i+1}/{len(sim_folders)})",end="\r")
        model_data = {'washout_epochs': washout_windows}

        for washout_len in washout_windows:
            washout_path = os.path.join(model_folder, str(washout_len))

            iteration_data = {}
            for it in range(iterations):
                if not os.path.exists(os.path.join(washout_path, str(it))):
                    print(f"{model_folder}/{washout_len}/{it} not found, skipping.")
                    continue
                exp = Experiment2(washout_path, str(it), load=True)
                if "train_outputs" not in exp.variables:
                    print(f"{model_folder}/{washout_len}/{it} not completed, skipping.")
                    continue
                metrics_j = {}
                for k in metrics:
                    values = exp.train_outputs['washout']['metrics'][k]
                    metrics_j[k] = [v / std if k in ['mse', 'nocb_error',"cortex_error"] else v for v in values]

                iteration_data[it] = metrics_j
                del exp

            if iteration_data:
                model_data[washout_len] = iteration_data
        if len(model_data) > 1:
            data[model_name] = model_data
            print(f"{model_name} loaded."+" "*30)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    save_var(data, data_path)
    print(f"{data_folder}: total {len(data)} elements loaded.")
    return data


def compute_convergence(data, threshold, metrics):
    convergence = {}
    for model, m_data in data.items():
        convergence[model] = {}
        washout_epochs = m_data.get('washout_epochs', [])
        for p_length in washout_epochs:
            convergence[model][p_length] = {metric: [] for metric in metrics}
            for it_data in m_data[p_length].values():
                for metric in ['mse', 'nocb_error']:
                    metric_data = np.array(it_data[metric])
                    try:
                        epoch = max(np.where(metric_data <= threshold)[0][0], 1)
                    except Exception:
                        epoch = max(len(metric_data),1)
                    convergence[model][p_length][metric].append(epoch)
    return convergence


def model_group(name):
    if name.startswith('RNN'):
        return 'RNN'
    if 'CxLocal' in name:
        return 'CxLocal-hgain' if 'h-gain' in name else 'CxLocal'
    if 'CxBP' in name:
        return 'CxBP-hgain' if 'h-gain' in name else 'CxBP'
    return 'other'


def extract_lr(name):
    if 'cx_lr.' in name:
        return name.split('cx_lr.')[-1]
    if 'local_lr.' in name:
        return name.split('local_lr.')[-1]
    return None


def select_washout_subset(conv_data, sim_names, family, variant='normal', rnn_names=None):
    rnn_names = list(rnn_names or [])
    family_names = []
    for n in sim_names:
        if family not in n:
            continue
        if variant == 'normal' and 'h-gain' in n:
            continue
        if variant == 'h-gain' and 'h-gain' not in n:
            continue
        family_names.append(n)
    family_names = sorted(family_names, key=lambda n: float(extract_lr(n)))
    selected_names = family_names + [r for r in rnn_names if r in sim_names or r in conv_data]
    selected_conv = {k: v for k, v in conv_data.items() if k in selected_names}
    lrs = [extract_lr(n) for n in family_names]
    return {'conv_data': selected_conv, 'sim_names': selected_names, 'family_names': family_names, 'lrs': lrs, 'rnn_names': [r for r in rnn_names if r in selected_names]}


def make_washout_styles(sim_names, family, variant='normal', rnn_names=None):
    rnn_names = list(rnn_names or [])
    cmap = Reds if family == 'CxBP' else Greens
    family_names = sorted([n for n in sim_names if family in n and (('h-gain' in n) == (variant == 'h-gain'))], key=lambda n: float(extract_lr(n)))
    lrs = [extract_lr(n) for n in family_names]
    vals = [0.9 - 0.6 * i / max(len(lrs) - 1, 1) for i in range(len(lrs))] if lrs else []
    family_colors = {n: cmap(v) for n, v in zip(family_names, vals)}
    styles = {}
    for n in family_names:
        styles[n] = {'color': family_colors[n], 'marker': 'o', 'linestyle': 'solid', 'zorder': 3, 'label': extract_lr(n)}
    for k, r in enumerate(rnn_names):
        alpha = np.linspace(0.3,1,len(rnn_names))[k] if len(rnn_names) > 1 else 1
        styles[r] = {'color': 'k', 'alpha': alpha, 'marker': 'o', 'linestyle': 'dashed', 'zorder': 2, 'label': r}
    return {'styles': styles, 'family_names': family_names, 'lrs': lrs, 'family_colors': family_colors, 'cmap': cmap}


def _mean_curve_and_err(conv_data, model, metric, error_method='minmax'):
    epochs = sorted(conv_data[model].keys(), key=lambda x: float(x))
    x_vals = np.array([float(e) for e in epochs])
    values = np.array([conv_data[model][e][metric] for e in epochs]).T
    means = values.mean(axis=0)
    if error_method == 'minmax':
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        yerr = np.vstack((means - mins, maxs - means))
    else:
        std = values.std(axis=0)
        yerr = np.vstack((means - std, means + std))
    return x_vals, means, yerr


def plot_washout_main(ax, conv_data=None, styles=None, metric='mse', fontsize=20,
                      xlabel='perturbation length', ylabel='convergence time',
                      xlim=(20, 10000), log=True, ylim=None, rnn_names=None,
                      bbox_to_anchor=None, error_method='minmax', capture=None, replay=None):
    legend_flag = True
    legend_elements = {}
    if replay is not None:
        for entry in replay:
            model = entry['name']
            x_vals = np.array(entry['x'])
            means = np.array(entry['y'])
            yerr = np.vstack((means - np.array(entry['ymin']), np.array(entry['ymax']) - means))
            line = ax.errorbar(
                x_vals, means, yerr=yerr,
                marker=entry['marker'], markersize=4, lw=1.2,
                elinewidth=0.5, color=entry['color'], alpha=entry.get('alpha', 1), linestyle=entry['linestyle'],
                markerfacecolor=entry['color'], markeredgewidth=0.4,
                capsize=3, zorder=entry['zorder']
            )
            if rnn_names is not None and model in rnn_names:
                legend_elements[model] = line
                line.set_label(model)
            else:
                if legend_flag:
                    legend_elements['model'] = line
                    line.set_label('model')
                    legend_flag = False
    else:
        for model, st in styles.items():
            if model not in conv_data:
                continue
            x_vals, means, yerr = _mean_curve_and_err(conv_data, model, metric, error_method=error_method)
            line = ax.errorbar(
                x_vals, means, yerr=yerr,
                marker=st['marker'], markersize=4, lw=1.2,
                elinewidth=0.5, color=st['color'], alpha = st.get('alpha',1), linestyle=st['linestyle'],
                markerfacecolor=st['color'], markeredgewidth=0.4,
                capsize=3, zorder=st['zorder']
            )
            if capture is not None:
                capture.append({
                    'name': model,
                    'label': st.get('label'),
                    'metric': metric,
                    'x': x_vals.tolist(),
                    'y': means.tolist(),
                    'xfill': x_vals.tolist(),
                    'ymin': (means - yerr[0]).tolist(),
                    'ymax': (means + yerr[1]).tolist(),
                    'color': st['color'],
                    'marker': st['marker'],
                    'linestyle': st['linestyle'],
                    'zorder': st['zorder'],
                    'alpha': st.get('alpha', 1),
                })
            if rnn_names is not None and model in rnn_names:
                legend_elements[model] = line
                line.set_label(model)
            else:
                if legend_flag:
                    legend_elements['model'] = line
                    line.set_label('model')
                    legend_flag = False
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    if log:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if metric == 'mse':
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set(linewidth=1.4)
    handles = []
    # if 'RNN' in legend_elements:
    #     handles.append(legend_elements['RNN'])
    # if 'model' in legend_elements:
    #     handles.append(legend_elements['model'])
    ax.legend(handles=legend_elements.values(), fontsize=0.8 * fontsize, bbox_to_anchor=bbox_to_anchor)


def add_washout_cbar(ax, lrs, family, fontsize=20):
    cmap = Reds if family == 'CxBP' else Greens
    n = len(lrs)
    colors = [cmap(0.9 - 0.6 * i / max(n - 1, 1)) for i in range(n)]
    cmap = ListedColormap(colors)
    bounds = np.linspace(0, n, n + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_label('learning rate', fontsize=0.9 * fontsize)
    cbar.set_ticks(np.arange(n).astype(int) + 0.5)
    cbar.set_ticklabels(lrs, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    
    
def figure_washout_convergence(name, conv_data_key, family, variant, rnn_names,
                                show_nocb=True, figsize=(12, 3.5), dpi=100,bbox_to_anchor=None,
                                ylim_mse=None, ylim_nocb=None, fontsize=20, log=True, error_method='minmax'):
    def prepare():
        conv_data = get_washout_data()[conv_data_key]
        sim_names = conv_data
        subset = select_washout_subset(conv_data, sim_names, family=family, variant=variant, rnn_names=rnn_names)
        style_info = make_washout_styles(subset['sim_names'], family=family, variant=variant, rnn_names=subset['rnn_names'])
        metrics = ['mse', 'nocb_error'] if show_nocb else ['mse']
        return {
            'figuretype': 'washout_convergence',
            'meta': {
                'family': family, 'variant': variant, 'rnn_names': rnn_names, 'show_nocb': show_nocb,
                'figsize': figsize, 'dpi': dpi, 'ylim_mse': ylim_mse, 'ylim_nocb': ylim_nocb,
                'fontsize': fontsize, 'log': log,'bbox_to_anchor': bbox_to_anchor, 'error_method': error_method
            },
            'subset': subset, 'style_info': style_info, 'lrs': subset['lrs'],
            'panels': [{'metric': m, 'series': []} for m in metrics],
        }

    def render(spec):
        meta = spec['meta']
        subset = spec.get('subset')
        style_info = spec.get('style_info')
        if subset is not None:
            for panel in spec['panels']:
                panel['series'] = []
        if meta['show_nocb']:
            fig, axs = plt.subplots(1, 3, figsize=meta['figsize'], dpi=meta['dpi'],
                               sharex=False, gridspec_kw={'width_ratios': [20, 20, 1]})
        else:
            fig, axs = plt.subplots(1, 2, figsize=meta['figsize'], dpi=meta['dpi'],
                               sharex=False, gridspec_kw={'width_ratios': [20, 1]})

        if subset is not None:
            plot_washout_main(
                axs[0],
                subset['conv_data'],
                style_info['styles'],
                metric='mse',
                fontsize=meta['fontsize'],
                log=meta['log'],
                ylim=meta['ylim_mse'],
                rnn_names=rnn_names,
                bbox_to_anchor=bbox_to_anchor,
                error_method=meta['error_method'],
                capture=spec['panels'][0]['series']
            )
        else:
            plot_washout_main(
                axs[0],
                metric='mse',
                fontsize=meta['fontsize'],
                log=meta['log'],
                ylim=meta['ylim_mse'],
                rnn_names=rnn_names,
                bbox_to_anchor=bbox_to_anchor,
                error_method=meta['error_method'],
                replay=spec['panels'][0]['series']
            )
        j = 1
        if meta['show_nocb']:
            if subset is not None:
                plot_washout_main(
                    axs[1],
                    subset['conv_data'],
                    style_info['styles'],
                    metric='nocb_error',
                    fontsize=meta['fontsize'],
                    log=meta['log'],
                    ylim=meta['ylim_nocb'],
                    rnn_names=rnn_names,
                    error_method=meta['error_method'],
                    capture=spec['panels'][1]['series']
                )
            else:
                plot_washout_main(
                    axs[1],
                    metric='nocb_error',
                    fontsize=meta['fontsize'],
                    log=meta['log'],
                    ylim=meta['ylim_nocb'],
                    rnn_names=rnn_names,
                    error_method=meta['error_method'],
                    replay=spec['panels'][1]['series']
                )
            j = 2
        lrs = subset['lrs'] if subset is not None else spec['lrs']
        add_washout_cbar(axs[j], lrs, family=meta['family'], fontsize=meta['fontsize'])

        plt.tight_layout()
        if subset is not None:
            del spec['subset']
            del spec['style_info']
        return fig, axs

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


def figure_washout_convergence_multi(
    name, datasets_keys, variants, rnn_names_list, data_origins, family,
    sim_names_key='conv_data',
    figsize=(12, 3.5), dpi=100, bbox_to_anchor=None,
    ylim_mse=None, ylim_old=None, fontsize=20, log=True, error_method='minmax',
    titles=None, legend_panel=None, sharey=False,
    wspace=0.06, legend_ncol=None, legend_top=1.12,
):
    n = len(datasets_keys)
    assert len(variants) == n and len(rnn_names_list) == n == len(data_origins)

    def prepare():
        cd = {**get_washout_data(), **get_washout_old_data()}
        datasets = [cd[k] for k in datasets_keys]
        sim_names_list = [list(cd[sim_names_key].keys()) for _ in range(n)]
        panels, all_lrs = [], []
        for conv_data, sim_names, variant, rnn_names, origin in zip(
                datasets, sim_names_list, variants, rnn_names_list, data_origins):
            if origin == 'new':
                subset = select_washout_subset(
                    conv_data, sim_names, family=family, variant=variant, rnn_names=rnn_names)
                style_info = make_washout_styles(
                    subset['sim_names'], family=family, variant=variant,
                    rnn_names=subset['rnn_names'])
                panels.append({'subset': subset, 'style_info': style_info,
                               'variant': variant, 'rnn_names': rnn_names, 'series': []})
                all_lrs.append(subset['lrs'])
            else:
                panels.append({'subset': conv_data, 'style_info': None,
                               'variant': variant, 'rnn_names': None, 'series': []})
        shared_lrs = sorted(set().union(*[set(lrs) for lrs in all_lrs])) if all_lrs else []
        return {
            'figuretype': 'washout_convergence_multi',
            'meta': {
                'family': family, 'variants': variants, 'rnn_names_list': rnn_names_list,
                'figsize': figsize, 'dpi': dpi, 'ylim_mse': ylim_mse, 'ylim_old': ylim_old,
                'fontsize': fontsize, 'log': log, 'bbox_to_anchor': bbox_to_anchor,
                'error_method': error_method, 'titles': titles,
                'legend_panel': legend_panel, 'sharey': sharey,
                'wspace': wspace, 'legend_ncol': legend_ncol, 'legend_top': legend_top,
            },
            'panels': panels, 'shared_lrs': shared_lrs,
        }

    def render(spec):
        meta = spec['meta']
        panels = spec['panels']
        n = len(panels)
        building = any('subset' in panel for panel in panels)
        if building:
            for panel in spec['panels']:
                panel['series'] = []

        # When sharing y, every data panel must use the SAME limits, otherwise the
        # one visible (leftmost) axis would mislabel the panels whose ticks we hide.
        old_ylim = meta['ylim_mse'] if meta['sharey'] else meta['ylim_old']

        width_ratios = [20] * n + [1]
        fig, axs = plt.subplots(
            1, n + 1, figsize=meta['figsize'], dpi=meta['dpi'],
            sharex=False, sharey=False,
            gridspec_kw={'width_ratios': width_ratios, 'wspace': meta['wspace']},
        )

        for i, panel in enumerate(panels):
            if data_origins[i] == 'new':
                anchor = meta['bbox_to_anchor'] if i == meta['legend_panel'] else None
                if building:
                    plot_washout_main(
                        axs[i], panel['subset']['conv_data'], panel['style_info']['styles'],
                        metric='mse', fontsize=meta['fontsize'], log=meta['log'],
                        ylim=meta['ylim_mse'], rnn_names=panel['rnn_names'],
                        bbox_to_anchor=anchor, error_method=meta['error_method'],
                        capture=panel['series'],
                    )
                else:
                    plot_washout_main(
                        axs[i],
                        metric='mse', fontsize=meta['fontsize'], log=meta['log'],
                        ylim=meta['ylim_mse'], rnn_names=panel['rnn_names'],
                        bbox_to_anchor=anchor, error_method=meta['error_method'],
                        replay=panel['series'],
                    )
            else:
                if building:
                    plot_washout_OLD(axs[i], panel['subset'], fontsize=meta['fontsize'], capture=panel['series'])
                else:
                    plot_washout_OLD(axs[i], fontsize=meta['fontsize'], replay=panel['series'])
                if old_ylim is not None:
                    axs[i].set_ylim(old_ylim)
            if meta['titles'] is not None:
                axs[i].set_title(meta['titles'][i], fontsize=meta['fontsize'])

        # ---- one shared legend, drop every per-axis legend ----
        handles, labels = [], []
        seen = set()
        for i in range(n):
            h, l = axs[i].get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in seen:
                    handles.append(hi); labels.append(li); seen.add(li)
            leg = axs[i].get_legend()
            if leg is not None:
                leg.remove()

        # ---- share y across data panels; strip inner ticks + labels ----
        if meta['sharey']:
            for i in range(1, n):
                axs[i].sharey(axs[0])
            if old_ylim is not None:
                axs[0].set_ylim(old_ylim)
        for i in range(1, n):
            axs[i].tick_params(axis='y', which='both', left=False, labelleft=False)
            axs[i].set_ylabel('')

        add_washout_cbar(axs[-1], spec['shared_lrs'], family=meta['family'],
                         fontsize=meta['fontsize'])

        fig.tight_layout()
        fig.subplots_adjust(wspace=meta['wspace'])  # tight_layout resets wspace

        # center the legend over the DATA panels only (exclude the colorbar)
        x0 = axs[0].get_position().x0
        x1 = axs[n - 1].get_position().x1
        xc = 0.5 * (x0 + x1)
        ncol = meta['legend_ncol'] or len(handles)
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(xc, meta['legend_top']),
                   ncol=ncol, fontsize=0.8 * meta['fontsize'],
                   frameon=False)

        if building:
            for panel in panels:
                panel.pop('subset', None)
                panel.pop('style_info', None)

        return fig, axs

    spec = load_or_build_spec(name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


####################################################################################
# Old data handlers
####################################################################################

def extract_data_OLD(update, data_folder):
    d_data = {}
    for l, model in enumerate(os.listdir(data_folder)):
        epoch_flag = True
        create_data_flag = False
        std_flag=True

        model_folder = os.path.join(data_folder, model) 
        data_filepath = os.path.join(model_folder,'data.pkl')
        if update:
            try:
                os.remove(data_filepath)
            except:
                pass
            create_data_flag=True
        else:
            try:
                d_data[model] = load_var(data_filepath)
                create_data_flag = False
                print(f'{model} data loaded.')
            except:
                print(f'{model} Data file not found, creating it...')
                create_data_flag = True

        if create_data_flag:
            washout_epochs = os.listdir(model_folder)
            model_data = {str(i): {} for i in washout_epochs}
            for washout_epoch in washout_epochs:
                epoch_folder = os.path.join(model_folder, washout_epoch, 'data')
                for f in os.listdir(epoch_folder):
                    if not f.isdigit():
                        continue
                    print(f'Loading model {model}/{washout_epoch}/{f}',end='\r')
                    exp = Experiment2(epoch_folder,f,load=True)
                    if std_flag:
                        dataset_params = exp.train_settings['dataset_params']
                        reaching = ReachingDataset(dataset_params)
                        std = reaching.generate_unperturbed_dataset('cart')[1].std().item()
                        std_flag = False
                    metrics = deepcopy(exp.train_outputs['metrics'])
                    metrics['changes'] = exp.train_outputs['changes']
                    for k in ['mse','nocb_error','cortex_error']:
                        metrics[k] = [val/std for val in metrics[k]]
                    metrics[k] = [val[0]/std for val in metrics['endpoint_distance']]
                
                    model_data[washout_epoch][f] = metrics
                    del exp, metrics
            model_data['washout_epochs'] = washout_epochs
            d_data[model] = model_data
            print(f'Loading model {model} done.')
            save_var(model_data, os.path.join(model_folder, 'data.pkl'))
    return d_data
            
def compute_convergence_OLD(data, threshold, nocb_epoch=-1):
    data = deepcopy(data)
    metrics = ['mse','nocb_error']
    # n_models = len(data)
    # perts = data[list(data.keys())[0]].keys()
    # n_perts = len(perts)
    # n_iter = len(data[perts[0]][list(data[perts[0]].keys())[0]])
    # convergence = np.zeros((n_models, n_perts, n_iter))
    convergence = {}
    for model, m_data in data.items():
        convergence[model] = {}
        m_data.pop('washout_epochs')
        for p_length, p_data in m_data.items():
            convergence[model][p_length] = {m: [] for m in metrics}
            for it_data in p_data.values():
                for metric in metrics:
                    metric_data = np.array(it_data[metric])
                    try:
                        epoch = np.where(metric_data <= threshold)[0][0]
                    except:
                        epoch = len(metric_data)
                    convergence[model][p_length][metric].append(epoch)
                    
    return convergence



def plot_washout_OLD(ax, conv_data=None, fontsize=20, capture=None, replay=None):

    log = True
    metric = 'mse'
    ylabel = 'convergence time'
    xlabel = 'perturbation length'

    legend_flag = True
    legend_elements = {}

    if replay is not None:
        for entry in replay:
            model = entry['name']
            x_vals = np.array(entry['x'])
            means = np.array(entry['y'])
            mins = np.array(entry['ymin'])
            maxs = np.array(entry['ymax'])
            linestyle = 'dashed' if 'RNN' in model else 'solid'
            zorder = 2 if 'RNN' in model else 3
            yerr = np.vstack((means - mins, maxs - means))
            line = ax.errorbar(x_vals, means, yerr=yerr,
                        marker=entry['marker'], markersize=4, lw=1.2,
                        elinewidth=0.5, color=entry['color'], linestyle=linestyle,
                        markerfacecolor=entry['color'], markeredgewidth=0.4,
                        capsize=3, alpha=1, zorder=zorder)
            if 'RNN' in model:
                line.set_label('plastic controller')
                legend_elements['RNN'] = line
            else:
                if legend_flag:
                    line.set_label('model')
                    legend_elements['model'] = line
                    legend_flag = False
        ax.set_xscale('log')
        ax.set_xlim(20, 10000)
        if not log:
            ax.set_ylim(-400, 5500)
            ax.set_yticks(np.arange(0, 6001, 1500))
        else:
            ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.legend(handles = [legend_elements['RNN'],legend_elements['model']],fontsize=0.8*fontsize)
        ax.tick_params(axis='both',which='both', labelsize=fontsize)
        for spine in ax.spines.values():
            spine.set(linewidth=1.4)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        return ax

    selected_RNN = 'RNN.lr1e-4.plastic_out'
    lrs = ['5e-7','1e-6','2e-6','5e-6','1e-5'][::-1]


    names = ['CxBP.40deg.CxLr'+lr for lr in lrs]
    markers = { name: 'o' for name in names}
    labels = {'CxBP.40deg.CxLr'+lr: lr for lr in lrs}

    # Learning rate values and corresponding colors

    colors = {'CxBP.40deg.CxLr'+lr: Reds(k) for lr,k in zip(lrs,[0.9 - 0.6 * i / (len(lrs)- 1) for i in range(len(lrs))][::-1])}


    # Adding RNN styling
    markers[selected_RNN] = 'o'
    labels[selected_RNN] = 'RNN'
    colors[selected_RNN] = 'k'


    # Metrics for subplots
    s = 17

    error_bars = True

    for model in markers.keys():
        epochs = sorted(conv_data[model].keys(), key=lambda x: float(x))
        x_vals = np.array([float(e) for e in epochs])
        values = np.array([conv_data[model][e][metric] for e in epochs]).T
        means = values.mean(axis=0)
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        if capture is not None:
            capture.append({
                'name': model,
                'label': labels[model],
                'metric': metric,
                'x': x_vals.tolist(),
                'y': means.tolist(),
                'xfill': x_vals.tolist(),
                'ymin': mins.tolist(),
                'ymax': maxs.tolist(),
                'color': colors[model],
                'marker': markers[model],
            })

        linestyle = 'dashed' if 'RNN' in model else 'solid'
        zorder = 2 if 'RNN' in model else 3
        if error_bars:
            yerr = np.vstack((means - mins, maxs - means))
            line = ax.errorbar(x_vals, means, yerr=yerr,
                        marker=markers[model], markersize=4, lw=1.2,
                        elinewidth=0.5, color=colors[model], linestyle=linestyle,
                        markerfacecolor=colors[model], markeredgewidth=0.4,
                        capsize=3, alpha=1, zorder=zorder)
        else:
            ax.plot(x_vals, means, marker=markers[model], markersize=2, lw=1.1,
                    color=colors[model], markerfacecolor=colors[model],
                    markeredgewidth=0.6, zorder=zorder)
            for i in range(5):
                faded_color = list(colors[model])
                faded_color[3] *= 0.5
                ax.scatter(x_vals, values[i], marker=markers[model], s=s,
                            lw=0.3, color='grey', facecolor=faded_color,
                            linewidths=0.5, zorder=zorder)
        if 'RNN' in model:
            line.set_label('plastic controller')
            legend_elements['RNN'] = line
        else:
            if legend_flag:
                line.set_label('model')
                legend_elements['model'] = line
                legend_flag=False
    ax.set_xscale('log')
    ax.set_xlim(20, 10000)
    if not log:
        ax.set_ylim(-400, 5500)
        ax.set_yticks(np.arange(0, 6001, 1500))
    else:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.legend(handles = [legend_elements['RNN'],legend_elements['model']],fontsize=0.8*fontsize)
    ax.tick_params(axis='both',which='both', labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set(linewidth=1.4)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # ax.grid(True, color=(0.92, 0.92, 0.92, 1), zorder=0)
    return ax


# %% [markdown]
#

# %% [markdown]
# ## DATA PROCESSING

# %%
_washout_data_loaded = None

def get_washout_data():
    global _washout_data_loaded
    if _washout_data_loaded is None:
        selected_rnn = 'RNN.plastic-out'

        # Two washout runs (see generate_data.py), each already spanning the full set of
        # perturbation windows in a single folder:
        #   washout        → cb_lr 5e-4, plastic-controller RNN at cx_lr 1e-4  (fast cb learning)
        #   washout-extra  → cb_lr 1e-4, plastic-controller RNN at cx_lr 5e-4  (alternative rates)
        load_folder_extra = os.path.join(WORKDIR, "washout-extra")

        data_main  = load_washout_data(load_folder,       update=UPDATE_METRICS)
        data_extra = load_washout_data(load_folder_extra, update=UPDATE_METRICS)

        # Plastic-readout RNN baseline at its two learning rates.
        rnn_lr_1e4 = data_main[selected_rnn]     # cx_lr = 1e-4 (the selected controller)
        rnn_lr_5e4 = data_extra[selected_rnn]    # cx_lr = 5e-4 (comparison)

        # cb_lr = 1e-4 dataset  → washout.normal / .h-gain / .local.*
        data_extra.pop(selected_rnn, None)
        data_extra["plastic controller"] = rnn_lr_1e4

        # cb_lr = 5e-4 dataset  → 5b / S9a / S9b / S13*
        data = data_main
        data.pop(selected_rnn, None)
        data['plastic controller, lr = 5e-4'] = rnn_lr_5e4
        data['plastic controller, lr = 1e-4'] = rnn_lr_1e4

        conv_data_extra = {}
        conv_data = {}
        convergence_threshold = 1e-6
        conv_data_extra = compute_convergence(data_extra, convergence_threshold, ['mse', 'nocb_error'])
        conv_data = compute_convergence(data, convergence_threshold, ['mse', 'nocb_error'])

        _washout_data_loaded = {
            'conv_data': conv_data,
            'conv_data_extra': conv_data_extra,
        }
    return _washout_data_loaded

_washout_old_data_loaded = None

def get_washout_old_data():
    global _washout_old_data_loaded
    if _washout_old_data_loaded is None:
        ## OLD DATA
        data_folder = os.path.join('/media/leo/T7 Shield/nvisu/', 'washout.v4')
        old_data = extract_data_OLD(False, data_folder)
        data_folder = os.path.join('/media/leo/T7 Shield/nvisu/', 'washout.v4.h-gain')
        old_data_hgain = extract_data_OLD(False, data_folder)


        old_conv_data = compute_convergence_OLD(old_data, threshold=1e-6, nocb_epoch=1000)
        old_conv_data_hgain = compute_convergence_OLD(old_data_hgain, threshold=1e-6, nocb_epoch=1000)

        _washout_old_data_loaded = {
            'old_conv_data': old_conv_data,
            'old_conv_data_hgain': old_conv_data_hgain,
        }
    return _washout_old_data_loaded

# %% [markdown]
# ## Washout BP

# %%
show_nocb = False

for variant, variant_name in zip(['normal', 'normal', 'h-gain'],['5b','S9a','S9b']):
    print(f"Variant: {variant}")
    rnns = ['plastic controller, lr = 1e-4'] if variant_name != 'S9a' else ['plastic controller, lr = 5e-4','plastic controller, lr = 1e-4']
    name = f"{variant_name}"
    fig, spec = figure_washout_convergence(
        name=name,
        conv_data_key='conv_data',
        family='CxBP',
        variant=variant,
        rnn_names=rnns,
        show_nocb=show_nocb,
        figsize=(7, 3.6),
        dpi=100,
        ylim_mse=(80, 4500),
        ylim_nocb=(20, 5000),
        fontsize=19,
        # bbox_to_anchor=(0.8, -0.2)
    )
    save_plot(
        fig,
        name,
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
# ## Answer to reviewers

# %%
variants       = ['normal', 'normal', 'normal']
rnn_names_list = [
    ['plastic controller, lr = 1e-4'],
    ['plastic controller'],
    ['plastic controller, lr = 5e-4','plastic controller, lr = 1e-4'],
]
data_origins = ['old', 'new', 'new']
titles = ['', '', '']

fig, spec = figure_washout_convergence_multi(
    name="R1",
    datasets_keys=['old_conv_data', 'conv_data_extra', 'conv_data'],
    variants=variants,
    rnn_names_list=rnn_names_list,
    data_origins=data_origins,
    family='CxBP',
    sim_names_key='conv_data',
    figsize=(16, 3.6),       # ~ n panels wide + thin colorbar
    dpi=100,
    ylim_mse=(50, 6000),
    ylim_old=(50, 6000),   # the OLD panel's own range
    fontsize=19,
    titles=titles,
    sharey=True,           # now only links the two new panels
)

save_plot(
    fig, "R1", plots_folder,
    savepdf=save_pdf, savetiff=save_tiff, savedata=save_data,
    data=spec, plotsdatafolder=plots_data_folder,
    dpi=100, transparent=True, bbox_inches='tight',
)
plt.show()
plt.close(fig)





variants       = ['normal', 'h-gain', 'h-gain']
rnn_names_list = [
    ['plastic controller, lr = 1e-4'],
    ['plastic controller'],
    ['plastic controller, lr = 5e-4','plastic controller, lr = 1e-4'],
]
data_origins = ['old', 'new', 'new']
titles = ['', '', '']

fig, spec = figure_washout_convergence_multi(
    name="R2",
    datasets_keys=['old_conv_data_hgain', 'conv_data_extra', 'conv_data'],
    variants=variants,
    rnn_names_list=rnn_names_list,
    data_origins=data_origins,
    family='CxBP',
    sim_names_key='conv_data',
    figsize=(16, 3.6),       # ~ n panels wide + thin colorbar
    dpi=100,
    ylim_mse=(50, 6000),
    ylim_old=(50, 6000),   # the OLD panel's own range
    fontsize=19,
    titles=titles,
    sharey=True,           # now only links the two new panels
    bbox_to_anchor=(0.1,1.05)
)

save_plot(
    fig, "R2", plots_folder,
    savepdf=save_pdf, savetiff=save_tiff, savedata=save_data,
    data=spec, plotsdatafolder=plots_data_folder,
    dpi=100, transparent=True, bbox_inches='tight',
)
plt.show()
plt.close(fig)



# %% [markdown]
# ## Local

# %%
for variant, variant_name in zip(['normal', 'h-gain'],['a','b']):
    print(f"Variant: {variant}")
    name = f"S13{variant_name}"
    fig, spec = figure_washout_convergence(
        name=name,
        conv_data_key='conv_data',
        family='CxLocal',
        variant=variant,
        rnn_names=['plastic controller, lr = 1e-4'],
        show_nocb=show_nocb,
        figsize=(7, 3.82),
        dpi=100,
        ylim_mse=(50, 5000),
        ylim_nocb=(20, 5000),
        fontsize=16,
    )
    save_plot(
        fig,
        name,
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
