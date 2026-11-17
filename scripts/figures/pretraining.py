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

# %% [markdown]
# # Rebased notebook

# %%
# %load_ext autoreload
# %autoreload 2

"""Figure generation for the pretraining experiments."""

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

load_folder = os.path.join(DATA_DIR, 'cx_pre')
config_folder = os.path.join(HOME, 'scripts/configs/cx_pretrain')

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

import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.linalg import pinv, subspace_angles, norm
from sklearn.decomposition import PCA
from matplotlib.cm import inferno
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from torch import nn

from ccdl.reaching import ReachingDataset
from ccdl.ccnet import CerebelloCorticalNet
from ccdl.utils import *


device = torch.device('cpu')
dtype = 'float32'
npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)
rnn_hidden_size = 256
model_class = "CerebelloCorticalNet"


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


# %% [markdown]
# # Methods

# %%
def load_model(path, return_model_state=False):
    path = path
    params = load_train_params(path, single_training=True)
    reaching = ReachingDataset(params['dataset_params'])
    in_features = reaching.input_size[2]

    rnn_hidden_size = params['rnn_hidden_size']
    params['net_params']['device'] = 'cpu'
    model = CerebelloCorticalNet(in_features, rnn_hidden_size, **params['net_params'])
    if return_model_state:
        return model, params['model_state']
    return model

def load_model_and_reaching(path, iteration):
    params = load_train_params(path, single_training=True)
    params['dataset_params']['device'] = torch.device('cpu')
    reaching = ReachingDataset(params['dataset_params'])
    std = reaching.generate_unperturbed_dataset('cart')[1].std().item()
    in_features = reaching.input_size[2]
    params['net_params']['device'] = torch.device('cpu')
    params['net_params']['dtype'] = torch.float32
    model_state = params['model_state']
    model = CerebelloCorticalNet(in_features, params['rnn_hidden_size'], **params['net_params'])
    model.cortex.load_state_dict(load_var(os.path.join(path, str(iteration), 'cortex_state_dict.pkl')))
    model.eval(**model_state)
    model.to('cpu')
    nuc_norm = fb_nuc_norm(model)
    return model, model_state, reaching, params, std, nuc_norm

def compute_single_alignment(name, source_folder, temp_folder, debug):
              
    path = os.path.join(source_folder, name)
    model, _, reaching, __builtins__, _, _ = load_model_and_reaching(path, '0')
    params = load_train_params(path,True)
    try:
        num_iterations = params['iterations']
    except:
        num_iterations = sum([f.isdigit() for f in os.listdir(path)])

    results = {
        'gradients': [],
        'alignment': [],
        'weak_alignment': [],
        'norm_weak_alignment': [],
        'subspaces_angle': [],
        'sub_angle_fb_out': [],
        'sub_angle_fb_in': [],
        'sub_angle_out_subrec': [],
        'sub_angle_fb_subrec': [],
        'sub_angle_in_subrec': [],
        'sub_angle_mov_fb': [],
        'sub_angle_mov_out': [],
        'sub_angle_mov_in': [],
        'sub_angle_rest_fb': [],
        'sub_angle_rest_out': [],
        'sub_angle_rest_in': [],
        'rec_eigs': []
    }
    
    print(f'\nComputing alignment on {name}...\n')
    for iteration in range(num_iterations):
        print(f'\nRunning iteration {iteration}')
        if debug:
            if not isinstance(debug,bool):
                print(f"Debugging mode: computing only on {debug} epochs")
        align = []
        weak_al = []
        norm_weak_al = []
        sub_angle = []
        sub_angle_fb_out = []
        sub_angle_fb_in = []
        sub_angle_out_subrec = []
        sub_angle_fb_subrec = []
        sub_angle_in_subrec = []
        sub_angle_mov_fb = []
        sub_angle_mov_out = []
        sub_angle_mov_in = []
        sub_angle_rest_fb = []
        sub_angle_rest_out = []
        sub_angle_rest_in = []
        rec_eig = {}

        iter_exp = Experiment2(path, str(iteration), load=True)
        try:
            parameters = iter_exp.pretrain_outputs['parameters']
        except:
            print(name,iteration," data could not be loaded. Skipping")
            continue
        movement_onset_time = reaching.hold_cue_time
        inputs, _ = reaching.generate_unperturbed_dataset('cart')
        h_size = model.hidden_size
        
        if iteration == 0:
            loss = iter_exp.pretrain_outputs['mse_losses']
            epochs = np.linspace(0,len(loss),len(parameters))
        results['gradients'].append(iter_exp.pretrain_outputs['gradients'])
        del iter_exp

        for j, net_weights in enumerate(parameters):
            if debug:
                print(f'running epoch {j}/{len(parameters)}',end='\r')
            model.load_state_dict(net_weights)
            w_out = model.cortex.fc.weight.detach().clone().cpu().numpy()
            w_fb = model.cortex.rnn.input2h_2.weight.detach().clone().cpu().numpy()
            w_p = pinv(w_out)
            w_rec = model.cortex.rnn.h2h.weight.detach().numpy()
            w_in = model.cortex.rnn.input2h.weight.detach().numpy()

            # Defining the parameters subspace
            eigenValues, eigenVectors = np.linalg.eig(w_rec)
            real_eig = np.real(eigenValues)
            idx = np.abs(real_eig).argsort()[::-1]
            eigenValues = eigenValues[idx]
            # selecting first two eigenvectors
            w_rec_sub = eigenVectors[:,idx[:2]]

            # defining rest activity low-dim space
            _, hidden = model(inputs,return_hidden=True)
            rest_activity = hidden[:movement_onset_time].view(-1,h_size).detach().numpy()
            pca = PCA(n_components=2)
            lowdim_activity = pca.fit_transform(rest_activity)
            rest_pc = pca.components_.T
            del pca

            # defining movement activity low-dim space
            mov_activity = hidden[movement_onset_time:].view(-1,h_size).detach().numpy()
            pca = PCA(n_components=2)
            lowdim_activity = pca.fit_transform(mov_activity)
            mov_pc = pca.components_.T
            del pca
            
            eps = 1e-6
            align.append(np.dot(w_fb.flatten() / (np.linalg.norm(w_fb.flatten())+eps),
                            w_p.flatten() / (np.linalg.norm(w_p.flatten())+eps)))
            weak_al.append(norm(w_out @ w_fb - np.eye(2)))
            norm_weak_al.append(norm(w_out / (norm(w_out)+eps) @ w_fb / (norm(w_fb)+eps) - np.eye(2)))
            
            # calculating angles between column subspaces
            sub_angle.append(np.rad2deg(subspace_angles(w_fb, w_p)))
            sub_angle_fb_out.append(np.rad2deg(subspace_angles(w_fb, w_out.T)))
            sub_angle_fb_in.append(np.rad2deg(subspace_angles(w_fb, w_in)))

            sub_angle_out_subrec.append(np.rad2deg(subspace_angles(w_out.T,w_rec_sub)))
            sub_angle_fb_subrec.append(np.rad2deg(subspace_angles(w_fb,w_rec_sub)))
            sub_angle_in_subrec.append(np.rad2deg(subspace_angles(w_in,w_rec_sub)))

            sub_angle_mov_fb.append(np.rad2deg(subspace_angles(w_fb, mov_pc)))
            sub_angle_mov_out.append(np.rad2deg(subspace_angles(w_out.T, mov_pc)))
            sub_angle_mov_in.append(np.rad2deg(subspace_angles(w_in, mov_pc)))

            sub_angle_rest_fb.append(np.rad2deg(subspace_angles(w_fb, rest_pc)))
            sub_angle_rest_out.append(np.rad2deg(subspace_angles(w_out.T, rest_pc)))
            sub_angle_rest_in.append(np.rad2deg(subspace_angles(w_in, rest_pc)))

            rec_eig[epochs[j]] = eigenValues

            if debug and not isinstance(debug,bool):
                if (j+1)%debug == 0:
                    break

        results['alignment'].append(np.array(align))
        results['weak_alignment'].append(np.array(weak_al))
        results['norm_weak_alignment'].append(np.array(norm_weak_al))
        results['subspaces_angle'].append(np.array(sub_angle))
        results['sub_angle_fb_out'].append(np.array(sub_angle_fb_out))
        results['sub_angle_fb_in'].append(np.array(sub_angle_fb_in))
        results['sub_angle_out_subrec'].append(np.array(sub_angle_out_subrec))
        results['sub_angle_fb_subrec'].append(np.array(sub_angle_fb_subrec))
        results['sub_angle_in_subrec'].append(np.array(sub_angle_in_subrec))
        results['sub_angle_mov_fb'].append(np.array(sub_angle_mov_fb))
        results['sub_angle_mov_out'].append(np.array(sub_angle_mov_out))
        results['sub_angle_mov_in'].append(np.array(sub_angle_mov_in))
        results['sub_angle_rest_fb'].append(np.array(sub_angle_rest_fb))
        results['sub_angle_rest_out'].append(np.array(sub_angle_rest_out))
        results['sub_angle_rest_in'].append(np.array(sub_angle_rest_in))
        results['rec_eigs'].append(rec_eig)

        del parameters, w_out, w_fb, w_p, align, weak_al, norm_weak_al, sub_angle
        del sub_angle_fb_out,sub_angle_out_subrec,sub_angle_fb_subrec, rec_eig
        del sub_angle_in_subrec,sub_angle_mov_fb,sub_angle_mov_out,sub_angle_fb_in
        del sub_angle_mov_in,sub_angle_rest_fb,sub_angle_rest_out,sub_angle_rest_in
        
    save_var(results, os.path.join(temp_folder, name+'.pkl'))
    del model, results
        
def compute_alignment_pool(source_folder, output_folder, debug=False, num_jobs=None, update=False):

    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    names = []
    for name in os.listdir(source_folder):
        if not name.split('.')[0].isdigit():
            continue
        if name + '.pkl' in os.listdir(output_folder) and not update:
            print(f'skipping {name}, already computed.')
            continue
        names.append(name)

    if num_jobs == 1:
        for name in names:
            compute_single_alignment(name, source_folder, output_folder, debug)
    else:
        with ThreadPoolExecutor(max_workers=num_jobs) as executor:
            futures = {
                executor.submit(compute_single_alignment, name, source_folder, output_folder, debug): name
                for name in names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    print(f"{name} alignment completed.")
                except Exception as exc:
                    print(f"{name} generated an exception: {exc}")



# %%
def figure_losses(
    figsize=(20, 6),
    start=0,
    steps=20,
    smoothing_window=10,
    selection=None,
    alpha=0.3,
    ls=None,
    lims=(8e-7, 1e-1),
    title=None,
    fill=True,
    alpha_fill=0.2,
    lw=1,
    dpi=60,
    fontsize=14,
    label_names=None,
    plot_iters=False,
    contrast=False,
    cool=False,
    smoothing='moving_average',
    label_q='name',
    markersize=5,
    minortickson=False,
    fill_smoothing=(5, 5),
    debug=False,
    spines_lw=1,
    ax=None,
):
    smoothing_fn = globals()[smoothing]

    def _prepare():
        mse_data = get_mse_data()
        local_label_names = {
            'r_init': "Random init., backprop",
            'ae_fixed': "Pseudoinv., fixed"
        } if not label_names else label_names

        groups = list(local_label_names.keys())
        colors = {group: f'C{i}' for i, group in enumerate(groups)}

        series = []
        stop = None
        last_marker = None
        last_markers_steps = None

        for group in groups:
            for (order, name), data in mse_data.groupby(['order', 'name']):
                if selection is not None:
                    assert isinstance(selection, list), "selection must be list(str,bool)"
                    if (selection[0] in name) != selection[1]:
                        continue

                if 'metric' in data.index.names:
                    unique_metrics = data.index.get_level_values('metric').unique()
                    if 'mse' in unique_metrics and len(unique_metrics) > 1:
                        data = data.loc[(slice(None), slice(None), slice(None), 'mse'), :]

                if ls == 'noise':
                    if 'rec0' in name:
                        linestyle, marker, label_suffix, hatch = 'solid', None, ' | no noise', '/'
                    elif 'rec-3' in name:
                        linestyle, marker, label_suffix, hatch = 'solid', 'o', r' | noise ($\sigma$=1e-3)', '.'
                    else:
                        linestyle, marker, label_suffix, hatch = 'dotted', None, r' | noise ($\sigma$=5e-4)', '*'
                elif ls == 'plant':
                    if 'no_plant' in name:
                        linestyle, marker, label_suffix, hatch = 'solid', None, ' | No plant', '/'
                    else:
                        linestyle, marker, label_suffix, hatch = 'solid', 'o', ' | Plant', '\\'
                else:
                    linestyle, marker, label_suffix, hatch = 'solid', None, '', ''

                if label_q == 'name':
                    if group not in name:
                        continue
                else:
                    if group != str(order):
                        continue

                arr = data.values
                markers_steps = (arr.shape[1] // 20) // steps if steps > 0 else None

                if smoothing_window > 0:
                    arr = moving_median(arr, smoothing_window)

                stop = arr.shape[1]
                mean = arr[:, start::steps].mean(axis=0)
                x_mean = np.linspace(start, stop, len(mean))

                mins = apply_gaussian_smoothing(
                    moving_median(arr[:, start::5*steps].min(axis=0), fill_smoothing[0]),
                    fill_smoothing[1]
                )
                mins = np.minimum(mins, mean[::5])

                maxs = apply_gaussian_smoothing(
                    moving_median(arr[:, start::5*steps].max(axis=0), fill_smoothing[0]),
                    fill_smoothing[1]
                )
                maxs = np.maximum(maxs, mean[::5])

                iter_curves = []
                if plot_iters:
                    for j in range(arr.shape[0]):
                        dd = arr[j, start::steps]
                        iter_curves.append({
                            "x": np.linspace(start, stop, len(dd)),
                            "y": dd,
                        })

                series.append({
                    "group": group,
                    "order": order,
                    "name": name,
                    "label": local_label_names[group] + label_suffix,
                    "color": colors[group],
                    "linestyle": linestyle,
                    "marker": marker,
                    "hatch": hatch,
                    "markers_steps": markers_steps,
                    "n_runs": arr.shape[0],
                    "x_mean": x_mean,
                    "y_mean": mean,
                    "x_fill": np.linspace(start, stop, len(mins))[:-1],
                    "y_min": mins[:-1],
                    "y_max": maxs[:-1],
                    "iter_curves": iter_curves,
                })
                series[-1]["x"] = series[-1]["x_mean"]
                series[-1]["y"] = series[-1]["y_mean"]
                series[-1]["xfill"] = series[-1]["x_fill"]
                series[-1]["ymin"] = series[-1]["y_min"]
                series[-1]["ymax"] = series[-1]["y_max"]
                series[-1]["raw_y"] = [c["y"] for c in iter_curves] if iter_curves else None

                last_marker = marker
                last_markers_steps = markers_steps

        return {
            "figure_type": "losses",
            "panels": [{
                "metric": "loss",
                "ylabel": "Training MSE",
                "yscale": "log",
                "series": series,
            }],
            "meta": {
                "start": start,
                "steps": steps,
                "smoothing": smoothing,
                "smoothing_window": smoothing_window,
                "fill_smoothing": fill_smoothing,
                "selection": selection,
                "lims": lims,
                "title": title,
                "ylabel": "Training MSE",
                "xlabel": "Epoch",
                "yscale": "log",
                "stop": stop,
                "debug_marker": last_marker,
                "debug_markers_steps": last_markers_steps,
            }
        }

    def _render(spec):
        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            local_ax = plt.gca()
        else:
            if isinstance(ax, np.ndarray):
                local_ax = ax.ravel()[0]
            elif isinstance(ax, (list, tuple)):
                local_ax = ax[0]
            else:
                local_ax = ax
            fig = local_ax.get_figure()

        if title:
            local_ax.set_title(title, fontsize=1.1 * fontsize)

        for s in spec["panels"][0]["series"]:
            if fill:
                local_ax.fill_between(
                    s["x_fill"], s["y_min"], s["y_max"],
                    alpha=alpha_fill, facecolor='none',
                    color=s["color"], linestyle=s["linestyle"],
                    hatch=s["hatch"], linewidth=0.0, zorder=1
                )

            if plot_iters:
                for curve in s["iter_curves"]:
                    local_ax.plot(
                        curve["x"], curve["y"],
                        alpha=alpha, lw=0.6 * lw,
                        linestyle=s["linestyle"], marker=s["marker"],
                        markersize=markersize if s["marker"] else None,
                        markevery=s["markers_steps"] if s["marker"] else None,
                        color=s["color"],
                        markeredgecolor='k' if s["marker"] else None,
                        markeredgewidth=0.2 * lw if s["marker"] else None
                    )

            if contrast:
                local_ax.plot(
                    s["x_mean"], s["y_mean"],
                    lw=1.84 * lw, alpha=2 * alpha,
                    linestyle=s["linestyle"], marker=s["marker"],
                    markersize=markersize if s["marker"] else None,
                    markevery=s["markers_steps"] if s["marker"] else None,
                    color='k',
                    markeredgecolor='k' if s["marker"] else None,
                    markeredgewidth=0.2 * lw if s["marker"] else None
                )

            local_ax.plot(
                s["x_mean"], s["y_mean"],
                lw=1.8 * lw, alpha=2 * alpha,
                linestyle=s["linestyle"], marker=s["marker"],
                markersize=markersize if s["marker"] else None,
                markevery=s["markers_steps"] if s["marker"] else None,
                color=s["color"], label=s["label"],
                markeredgecolor='k' if s["marker"] else None,
                markeredgewidth=0.1 * lw if s["marker"] else None
            )

        if cool:
            stop = spec["meta"]["stop"]
            local_ax.set_yscale('log')
            local_ax.set_ylim(*lims)
            local_ax.set_xlim(-0.01 * stop, stop)
            local_ax.set_ylabel('Training MSE', fontsize=fontsize)
            local_ax.set_yticks([1e1, 1e-1, 1e-3, 1e-5, 1e-7])
            local_ax.set_xticks(np.arange(0, stop + 1, stop // 5))
            local_ax.spines['top'].set_visible(False)
            local_ax.spines['right'].set_visible(False)
            local_ax.spines['bottom'].set(linewidth=1.3 * spines_lw)
            local_ax.spines['left'].set(linewidth=1.3 * spines_lw)
            local_ax.tick_params(axis='x', which='major', width=1.3 * spines_lw, length=4, labelsize=fontsize)
            local_ax.tick_params(axis='y', which='major', width=1.3 * spines_lw, length=4, labelsize=fontsize)
            local_ax.grid(False)
        else:
            stop = spec["meta"]["stop"]
            local_ax.set_yscale('log')
            local_ax.set_ylim(*lims)
            local_ax.set_xlim(-0.01 * stop, stop)

        if minortickson:
            local_ax.grid(which='both', color='gray', linestyle='-', linewidth=0.4)
            local_ax.minorticks_on()

        handles, labels = local_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        local_ax.legend(by_label.values(), by_label.keys(), fontsize=fontsize)

        if debug:
            print("marker:", spec["meta"]["debug_marker"])
            print("markersize:", markersize)
            print("markers_steps:", spec["meta"]["debug_markers_steps"])

        return fig, local_ax, created_fig

    spec = load_or_build_spec("S2a", plots_data_folder, _prepare, rebuild=REBUILD_SOURCE_DATA)
    if len(spec.get("panels", [])) == 0 or len(spec["panels"][0].get("series", [])) == 0:
        raise ValueError(
            "figure_losses: no series matched the requested filters. "
            f"Check label_names={list(label_names.keys()) if label_names else None}, "
            f"selection={selection}, label_q={label_q}."
        )

    fig, ax_out, _ = _render(spec)
    return fig, spec


def figure_alignment(
    folder,
    variables,
    labels,
    mse=None,
    label_q="name",
    smoothing=50,
    fill=False,
    plot_all_lines=False,
    gaussian_smoothing=5,
    ema_smoothing=100,
    fill_smoothing=(5, 5),
    step=None,
    figsize=(6, 3),
    dpi=200,
    fontsize=14,
    ax=None,
):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, LogLocator

    label_mapping = {
        "gradients": r"$\langle |\Delta \mathbf{W}_{rec}| \rangle_{neurons}$",
        "alignment": r"$\frac{\mathbf{W_{fb}} \cdot \mathbf{W_p}}{||\mathbf{W_{fb}}|| ||\mathbf{W_p}||}$",
        "weak_alignment": r"$||\mathbf{W_{out}} \mathbf{W_{fb}} - \mathbf{I}||$",
        "norm_weak_alignment": r"$||\frac{\mathbf{W_{out}}}{||\mathbf{W_{out}}||} \frac{\mathbf{W_{fb}}}{||\mathbf{W_{fb}}||} - \mathbf{I}||$",
        "subspaces_angle": r"$\angle(\mathbf{W_{fb}}, \mathbf{W_p})$",
        "sub_angle_fb_out": r"$\angle(\mathbf{W_{fb}}, \mathbf{W_{out}^T})$",
        "sub_angle_fb_in": r"$\angle(\mathbf{W_{fb}}, \mathbf{W_{in}})$",
        "sub_angle_out_subrec": r"$\angle(\mathbf{W_{out}^T}, \mathbf{W_{rec}^{sub}})$",
        "sub_angle_fb_subrec": r"$\angle(\mathbf{W_{fb}}, \mathbf{W_{rec}^{sub}})$",
        "sub_angle_in_subrec": r"$\angle(\mathbf{W_{in}}, \mathbf{W_{rec}^{sub}})$",
        "sub_angle_mov_fb": r"$\angle(\mathbf{W_{fb}}, \mathbf{PC_{mov}})$",
        "sub_angle_mov_out": r"$\angle(\mathbf{W_{out}^T}, \mathbf{PC_{mov}})$",
        "sub_angle_mov_in": r"$\angle(\mathbf{W_{in}}, \mathbf{PC_{mov}})$",
        "sub_angle_rest_fb": r"$\angle(\mathbf{W_{fb}}, \mathbf{PC_{rest}})$",
        "sub_angle_rest_out": r"$\angle(\mathbf{W_{out}^T}, \mathbf{PC_{rest}})$",
        "sub_angle_rest_in": r"$\angle(\mathbf{W_{in}}, \mathbf{PC_{rest}})$"
    }

    save_steps = 100 if step is None else step

    def _prepare():
        flag = True
        if isinstance(labels, list):
            warnings.warn("labels should be dict, avoiding reordering errors.")
            flag = False

        alignment_data = get_alignment_data()
        mse_list = alignment_data.copy()
        name_selection = mse_list.index.get_level_values('order').astype('str').isin(labels.keys())
        mse_list = mse_list.loc[name_selection]
        names = np.unique(['.'.join([str(order), name, 'pkl']) for order, name in zip(mse_list.index.get_level_values('order'), mse_list.index.get_level_values('name'))])

        panels = []

        if mse is not None:
            panel_series = []
            for k, name in enumerate(names):
                split_name = name.split('.')
                order = split_name[0]
                n = '.'.join(split_name[1:-1])

                label = labels[n] if (flag and label_q == 'name') else labels[order] if flag else labels[k]

                cond = np.array([n == i_name for i_name in mse.index.get_level_values('name')])
                data = mse.loc[cond].values[:, ::save_steps]
                x = np.linspace(0, data.shape[1] * save_steps, data.shape[1])
                data = apply_gaussian_smoothing(moving_median(data, smoothing), gaussian_smoothing)
                mean = data.mean(axis=0)
                mins = apply_gaussian_smoothing(moving_median(data[:, ::5].min(axis=0), fill_smoothing[0]), fill_smoothing[1])
                mins = np.minimum(mins, mean[::5])
                maxs = apply_gaussian_smoothing(moving_median(data[:, ::5].max(axis=0), fill_smoothing[0]), fill_smoothing[1])
                maxs = np.maximum(maxs, mean[::5])

                panel_series.append({
                    "name": name,
                    "label": label,
                    "color": f"C{k}",
                    "x": x,
                    "y": mean,
                    "x_fill": x[::5],
                    "y_min": mins,
                    "y_max": maxs,
                    "all_lines": data if plot_all_lines else None,
                    "metric": "mse",
                })
                panel_series[-1]["xfill"] = panel_series[-1]["x_fill"]
                panel_series[-1]["ymin"] = panel_series[-1]["y_min"]
                panel_series[-1]["ymax"] = panel_series[-1]["y_max"]
                panel_series[-1]["raw_y"] = panel_series[-1]["all_lines"]

            panels.append({
                "metric": "mse",
                "ylabel": "error (baseline)",
                "yscale": "log",
                "series": panel_series,
            })

        for metric in variables:
            panel_series = []

            for k, name in enumerate(names):
                metrics = load_var(os.path.join(folder, name))
                split_name = name.split('.')
                order = split_name[0]
                n = '.'.join(split_name[1:-1])
                label = labels[n] if (flag and label_q == 'name') else labels[order] if flag else labels[k]

                data = metrics[metric]

                if metric == "gradients":
                    data = np.stack([ema(data[j]['cortex']['rnn.h2h.weight'], ema_smoothing) for j in range(len(data))])[:, ::2*save_steps]
                    x = np.linspace(1, data.shape[1] * save_steps, data.shape[1])
                    data = apply_gaussian_smoothing(moving_median(data, smoothing), gaussian_smoothing)
                    mean = data.mean(axis=0)
                    mins = apply_gaussian_smoothing(moving_median(data[:, ::5].min(axis=0), fill_smoothing[0]), fill_smoothing[1])
                    mins = np.minimum(mins, mean[::5])
                    maxs = apply_gaussian_smoothing(moving_median(data[:, ::5].max(axis=0), fill_smoothing[0]), fill_smoothing[1])
                    maxs = np.maximum(maxs, mean[::5])

                    panel_series.append({
                        "name": name,
                        "label": label,
                        "color": f"C{k}",
                        "x": x,
                        "y": mean,
                        "x_fill": x[::5],
                        "y_min": mins,
                        "y_max": maxs,
                        "all_lines": data if plot_all_lines else None,
                        "metric": metric,
                        "components": None,
                    })
                    panel_series[-1]["xfill"] = panel_series[-1]["x_fill"]
                    panel_series[-1]["ymin"] = panel_series[-1]["y_min"]
                    panel_series[-1]["ymax"] = panel_series[-1]["y_max"]
                    panel_series[-1]["raw_y"] = panel_series[-1]["all_lines"]

                elif 'sub' not in metric:
                    x = np.linspace(0, (len(data[0]) - 1) * save_steps, len(data[0]))
                    data = apply_gaussian_smoothing(moving_median(np.stack(data), smoothing), gaussian_smoothing)
                    mean = data.mean(axis=0)
                    mins = apply_gaussian_smoothing(moving_median(data[:, ::5].min(axis=0), fill_smoothing[0]), fill_smoothing[1])
                    mins = np.minimum(mins, mean[::5])
                    maxs = apply_gaussian_smoothing(moving_median(data[:, ::5].max(axis=0), fill_smoothing[0]), fill_smoothing[1])
                    maxs = np.maximum(maxs, mean[::5])

                    panel_series.append({
                        "name": name,
                        "label": label,
                        "color": f"C{k}",
                        "x": x,
                        "y": mean,
                        "x_fill": x[::5],
                        "y_min": mins,
                        "y_max": maxs,
                        "all_lines": data if plot_all_lines else None,
                        "metric": metric,
                        "components": None,
                    })
                    panel_series[-1]["xfill"] = panel_series[-1]["x_fill"]
                    panel_series[-1]["ymin"] = panel_series[-1]["y_min"]
                    panel_series[-1]["ymax"] = panel_series[-1]["y_max"]
                    panel_series[-1]["raw_y"] = panel_series[-1]["all_lines"]

                else:
                    x = np.linspace(0, (len(data[0]) - 1) * save_steps, len(data[0]))
                    comp_data = []

                    for j in range(2):
                        j_data = np.stack(data)[:, :, j]
                        j_data = apply_gaussian_smoothing(moving_median(j_data, smoothing), gaussian_smoothing)
                        mean = j_data.mean(axis=0)
                        mins = apply_gaussian_smoothing(moving_median(j_data[:, ::5].min(axis=0), fill_smoothing[0]), fill_smoothing[1])
                        mins = np.minimum(mins, mean[::5])
                        maxs = apply_gaussian_smoothing(moving_median(j_data[:, ::5].max(axis=0), fill_smoothing[0]), fill_smoothing[1])
                        maxs = np.maximum(maxs, mean[::5])

                        comp_data.append({
                            "component": f"angle_{j+1}",
                            "linestyle": 'solid' if j == 0 else '--',
                            "x": x,
                            "y": mean,
                            "x_fill": x[::5],
                            "y_min": mins,
                            "y_max": maxs,
                        })
                        comp_data[-1]["xfill"] = comp_data[-1]["x_fill"]
                        comp_data[-1]["ymin"] = comp_data[-1]["y_min"]
                        comp_data[-1]["ymax"] = comp_data[-1]["y_max"]

                    panel_series.append({
                        "name": name,
                        "label": label,
                        "color": f"C{k}",
                        "metric": metric,
                        "components": comp_data,
                    })

            panels.append({
                "metric": metric,
                "ylabel": label_mapping[metric],
                "yscale": "log" if metric == "gradients" else "linear",
                "series": panel_series,
            })

        return {
            "figure_type": "alignment",
            "panels": panels,
            "meta": {
                "label_q": label_q,
                "smoothing": smoothing,
                "gaussian_smoothing": gaussian_smoothing,
                "ema_smoothing": ema_smoothing,
                "fill_smoothing": fill_smoothing,
                "step": step,
                "fontsize": fontsize,
            }
        }

    def _render(spec):
        n_panels = len(spec["panels"])

        if ax is None:
            fig, axes = plt.subplots(n_panels, figsize=(figsize[0], figsize[1] * n_panels), dpi=dpi)
            if n_panels == 1:
                axes = [axes]
            elif isinstance(axes, np.ndarray):
                axes = axes.ravel().tolist()
        else:
            if isinstance(ax, np.ndarray):
                axes = ax.ravel().tolist()
                fig = axes[0].get_figure()
            elif isinstance(ax, (list, tuple)):
                axes = list(ax)
                fig = axes[0].get_figure()
            else:
                axes = [ax]
                fig = ax.get_figure()

        if len(axes) < n_panels:
            raise ValueError(
                f"figure_alignment: not enough axes provided. "
                f"Need {n_panels}, got {len(axes)}."
            )

        axes = axes[:n_panels]

        for i, panel in enumerate(spec["panels"]):
            axi = axes[i]
            axi.set_ylabel(panel["ylabel"], fontsize=fontsize)

            if panel["yscale"] == "log":
                axi.set_yscale("log")
                if panel["metric"] == "gradients":
                    axi.set_ylim(1e-11, 1e-1)

            for s in panel["series"]:
                if s.get("components") is None:
                    if fill:
                        axi.fill_between(s["x_fill"], s["y_min"], s["y_max"], alpha=0.2, color=s["color"])
                    axi.plot(s["x"], s["y"], color=s["color"], lw=1.5, label=s["label"], alpha=0.7)

                    if plot_all_lines and s["all_lines"] is not None:
                        for row in s["all_lines"]:
                            axi.plot(s["x"], row, color=s["color"], alpha=0.3, lw=0.6)
                else:
                    handles = [
                        Line2D([0], [0], color='black', lw=2, label='Angle 1'),
                        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Angle 2')
                    ]

                    for comp in s["components"]:
                        if fill:
                            axi.fill_between(comp["x_fill"], comp["y_min"], comp["y_max"], alpha=0.2, color=s["color"])
                        axi.plot(
                            comp["x"], comp["y"],
                            color=s["color"],
                            linestyle=comp["linestyle"],
                            lw=1.5,
                            alpha=0.7
                        )
                    axi.legend(handles=handles, fontsize=fontsize)

            # limit the y-axis to ~4-5 ticks
            if panel["yscale"] != "log":
                axi.set_ylim(-8, 100)
                axi.set_yticks([0, 45, 90])

            # ticks at 0.8 * fontsize (x/y labels and legend use fontsize)
            axi.tick_params(axis="both", labelsize=0.9 * fontsize)
            axi.grid(True)
            axi.set_xlabel("epoch", fontsize=fontsize)
        
        plt.tight_layout()
        return fig, axes

    spec = load_or_build_spec("S2b-e", plots_data_folder, _prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axes = _render(spec)
    return fig, spec


def figure_systematic_error(name, model_folder, pert_type, values, effective_gains, figsize=(17, 5), lw=2.3,
                            markersize=8, fontsize=22, average_method='median', ylims=[1e-9, 5e-2],
                            spines_lw=1, yticks=[1e-2, 1e-4, 1e-6, 1e-8], xlims=(-100, 100),
                            ytick_order=1e-3, dpi=200, ax=None, spec_name=None):
    def prepare():
        path = os.path.join(model_folder, name)
        model, model_state, reaching, params, std, nuc_norm = load_model_and_reaching(path, 0)
        n_iterations = params['iterations']
        perturbation_params = {
            "angle": 0,
            "amplitude_factor": 1.0,
            "translation": [0.0, 0.0]
        }

        mse_array_full = np.zeros((n_iterations, len(effective_gains), len(values)))

        for iteration in range(n_iterations):
            model, model_state, reaching, params, std, nuc_norm = load_model_and_reaching(path, iteration)
            loss = nn.MSELoss()

            nuc_norm = fb_nuc_norm(model)

            for i, eff_gain in enumerate(effective_gains):
                model.gain = eff_gain / nuc_norm
                for j, val in enumerate(values):
                    perturbation_params[pert_type] = val
                    reaching.apply_perturbation(perturbation_params)
                    inputs, targets = reaching.generate_perturbed_dataset('cart')
                    inputs, theta_targets = reaching.generate_perturbed_dataset('angles')
                    outputs = model(inputs, input_2=theta_targets, input_2_type='angles',
                                    return_thetas=False)
                    mse = loss(outputs, targets).detach().item() / std
                    mse_array_full[iteration, i, j] = mse
            del model

        if average_method == 'median':
            mse_array = np.median(mse_array_full, axis=0)
        else:
            mse_array = np.mean(mse_array_full, axis=0)

        colors_map = inferno(np.linspace(0.1, 0.6, len(effective_gains)))
        colors = {eg: colors_map[i] for i, eg in enumerate(effective_gains)}

        if pert_type == 'translation':
            x = [np.linalg.norm(point) for point in values]
        else:
            x = values

        x_name = {
            'amplitude_factor': 'factor (radius)',
            'angle': 'angle',
            'translation': 'strength (translation)'
        }

        panels = []
        for panel_idx in range(2):
            panel = {
                'metric': f'systematic_error_panel_{panel_idx}',
                'ylabel': 'error (perturbation)',
                'xlabel': f"perturbation {x_name[pert_type]}",
                'yscale': 'linear' if panel_idx == 0 else 'log',
                'series': [],
            }

            for i, eff_gain in enumerate(effective_gains):
                panel['series'].append({
                    'name': f'{name}_gain_{eff_gain}_panel_{panel_idx}',
                    'label': f'effective gain = {eff_gain}',
                    'color': colors[eff_gain],
                    'metric': 'systematic_error',
                    'component': 'mean',
                    'x': np.array(x),
                    'y': np.array(mse_array[i]),
                    'marker': 'o',
                    'markersize': markersize,
                    'lw': lw,
                    'nruns': n_iterations,
                    'effective_gain': eff_gain,
                    'raw_y': mse_array_full[:, i, :].copy(),
                })
            panels.append(panel)

        return {
            'figuretype': 'systematic_error',
            'panels': panels,
            'meta': {
                'name': name,
                'pert_type': pert_type,
                'values': values,
                'effective_gains': effective_gains,
                'average_method': average_method,
                'ylims': ylims,
                'yticks': yticks,
                'xlims': xlims,
                'ytick_order': ytick_order,
                'figsize': figsize,
                'dpi': dpi,
                'fontsize': fontsize,
                'lw': lw,
                'markersize': markersize,
                'spines_lw': spines_lw,
                'n_iterations': n_iterations,
                'colors_map': colors_map,
            }
        }

    def render(spec):
        created_fig = False
        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                    gridspec_kw={"width_ratios": [1, 0.8]})
            created_fig = True
        else:
            if isinstance(ax, np.ndarray):
                axs = ax.ravel().tolist()
            elif isinstance(ax, (list, tuple)):
                axs = list(ax)
            else:
                axs = [ax]
            fig = axs[0].get_figure()

        if len(axs) != 2:
            raise ValueError(f"figure_systematic_error expects 2 axes, got {len(axs)}.")

        for panel_idx, panel in enumerate(spec['panels']):
            axi = axs[panel_idx]
            for s in panel['series']:
                axi.plot(
                    s['x'], s['y'],
                    label=s['label'],
                    marker=s.get('marker', None),
                    markersize=s.get('markersize', markersize),
                    lw=s.get('lw', lw),
                    color=s['color'],
                )

        colors_map = spec['meta']['colors_map']
        cmap = ListedColormap(colors_map)
        bounds = np.linspace(0, len(effective_gains), len(effective_gains) + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axs[0], orientation='vertical', fraction=0.3, pad=0.07, shrink=0.75)
        cbar.set_label('effective feedback strength', fontsize=fontsize)
        cbar.set_ticks(np.arange(len(effective_gains)) + 0.5)
        cbar.set_ticklabels([str(int(eg)) for eg in effective_gains])
        cbar.ax.tick_params(labelsize=22)

        axs[1].set_yscale('log')
        axs[1].set_ylim(*ylims)
        axs[1].set_yticks(yticks)

        axs[0].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda k, pos: f"{int(k/ytick_order)}{ytick_order:.0e}".replace("1e-0", "e-"))
        )

        for i in range(2):
            axs[i].set_ylabel(spec['panels'][i]['ylabel'], fontsize=fontsize)
            axs[i].set_xlabel(spec['panels'][i]['xlabel'], fontsize=fontsize)
            axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
            axs[i].set_xlim(*xlims)
            grid_color = (222/255, 222/255, 222/255, 92/100)
            axs[i].grid(True, color=grid_color)
            axs[i].spines['top'].set(linewidth=1.3 * spines_lw)
            axs[i].spines['right'].set(linewidth=1.3 * spines_lw)
            axs[i].spines['bottom'].set(linewidth=1.3 * spines_lw)
            axs[i].spines['left'].set(linewidth=1.3 * spines_lw)

        return fig, axs, created_fig

    spec = load_or_build_spec(spec_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs, created_fig = render(spec)
    return fig, spec


def compute_feedback_projection_amplitude(
    name,
    folder,
    angles,
    order=2,
    values=[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    average_method='median'
):
    perturbation_params = {
        "angle": 20,
        "amplitude_factor": 1.0,
        "translation": [0.0, 0.0]
    }

    path = os.path.join(folder, name)
    mse_array = np.zeros((5, len(angles), len(values)))

    for iteration in range(5):
        model, model_state, reaching, params, std, nuc_norm = load_model_and_reaching(path, iteration)
        loss = nn.MSELoss()

        W_fb = torch.linalg.matrix_norm(model.cortex.rnn.input2h_2.weight, ord=order)
        gains = [value / W_fb for value in values]

        for i, gain in enumerate(gains):
            model.gain = gain
            for j, angle in enumerate(angles):
                perturbation_params['angle'] = angle
                reaching.apply_perturbation(perturbation_params)
                inputs, targets = reaching.generate_perturbed_dataset('cart')
                inputs, theta_targets = reaching.generate_perturbed_dataset('angles')
                _, un_theta_targets = reaching.generate_unperturbed_dataset('angles')
                outputs = model(inputs, input_2=theta_targets, input_2_type='angles', return_thetas=False)
                mse = loss(outputs, targets).detach().item() / std
                mse_array[iteration, j, i] = mse

    if average_method == 'mean':
        avg_array = mse_array.mean(axis=0)
    else:
        avg_array = np.median(mse_array, axis=0)

    return avg_array, mse_array


def figure_feedback_projection_amplitude(
    values,
    version,
    load_folder,
    angles=[40],
    order='nuc',
    average_method='median',
    figsize=(10.5, 5),
    dpi=200,
    fontsize=26,
    lw=2,
    logy=True,
    markersize=11,
    markers=None,
    labels=None,
    spec_name=None,
):
    def prepare():
        mse_arrays = {}
        full_array = {}
        series = []

        color = (70 / 255, 70 / 255, 70 / 255, 97 / 100)
        grid_color = (233 / 255, 233 / 255, 233 / 255, 97 / 100)

        for i, name in enumerate(labels.keys()):
            mse_arrays[name], full_array[name] = compute_feedback_projection_amplitude(
                name,
                load_folder,
                angles,
                order=order,
                values=values,
                average_method=average_method
            )

            for j, angle in enumerate(angles):
                series.append({
                    'name': name,
                    'label': labels[name],
                    'angle': angle,
                    'color': color,
                    'markerfacecolor': f'C{i}',
                    'marker': markers[name],
                    'markeredgewidth': 1,
                    'lw': lw,
                    'markersize': markersize,
                    'zorder': 3,
                    'x': list(values),
                    'y': mse_arrays[name][j].tolist(),
                    'raw_y': full_array[name][:, j, :].tolist(),
                })

        spec = {
            'figuretype': 'feedback_projection_amplitude',
            'panels': [{
                'metric': 'adaptation_error_vs_effective_feedback_strength',
                'ylabel': 'error (adaptation)',
                'xlabel': r'effective feedback strength',
                'xscale': 'log',
                'yscale': 'log' if logy else 'linear',
                'series': series,
            }],
            'meta': {
                'version': version,
                'angles': list(angles),
                'order': order,
                'average_method': average_method,
                'figsize': figsize,
                'dpi': dpi,
                'fontsize': fontsize,
                'lw': lw,
                'logy': logy,
                'markersize': markersize,
                'xlim': (7e-2, 1200) if version == 'long' else (7e-2, 100),
                'ylim': (5e-5, 10) if version == 'long' else (5e-5, 0.01),
                'grid_color': grid_color,
                'spine_linewidth': 2,
                'major_tick_width': 2,
                'major_tick_length': 6,
                'minor_tick_width': 1,
                'minor_tick_length': 3.5,
            }
        }
        return spec

    def render(spec):
        fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        custom_handles = []

        panel = spec['panels'][0]
        meta = spec['meta']

        for s in panel['series']:
            axs.plot(
                s['x'],
                s['y'],
                marker=s['marker'],
                markersize=s['markersize'],
                lw=s['lw'],
                color=s['color'],
                markerfacecolor=s['markerfacecolor'],
                markeredgewidth=s['markeredgewidth'],
                zorder=s['zorder']
            )
            custom_handles.append(
                Line2D(
                    [0], [0],
                    marker=s['marker'],
                    color=s['color'],
                    markerfacecolor=s['markerfacecolor'],
                    lw=2,
                    markersize=s['markersize'],
                    markeredgewidth=s['markeredgewidth'],
                    label=s['label']
                )
            )

        if panel['yscale'] == 'log':
            axs.set_yscale('log')
        axs.set_xscale('log')

        axs.legend(handles=custom_handles, fontsize=fontsize)
        axs.set_ylabel(panel['ylabel'], fontsize=fontsize)
        axs.set_xlabel(panel['xlabel'], fontsize=fontsize)
        axs.tick_params(axis='both', which='major', labelsize=fontsize)

        axs.set_xlim(*meta['xlim'])
        axs.set_ylim(*meta['ylim'])

        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set(linewidth=meta['spine_linewidth'])
        axs.spines['left'].set(linewidth=meta['spine_linewidth'])
        axs.tick_params(axis='y', which='major', width=meta['major_tick_width'], length=meta['major_tick_length'])
        axs.tick_params(axis='x', which='major', width=meta['major_tick_width'], length=meta['major_tick_length'])
        axs.tick_params(axis='y', which='minor', width=meta['minor_tick_width'], length=meta['minor_tick_length'])
        axs.tick_params(axis='x', which='minor', width=meta['minor_tick_width'], length=meta['minor_tick_length'])

        return fig, axs

    spec = load_or_build_spec(spec_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec

def figure_outputs(
    folder,
    gain,
    pert_params,
    title='',
    linestyle='solid',
    iteration=0,
    mean_iters=False,
    plot_targets=True,
    label=None,
    add_feedback_legend=False,
    draw_main_legend=True,
    figsize=(6, 6),
    dpi=200,
    lw=4.2,
    ax=None,
    debug=False,
    spec_name=None,
):
    def prepare():
        _, _, reaching, params, _, _ = load_model_and_reaching(folder, 0)

        reaching.apply_perturbation(pert_params)
        inputs, targets = reaching.generate_perturbed_dataset('cart')
        _, un_targets = reaching.generate_unperturbed_dataset('cart')
        _, theta_targets = reaching.generate_perturbed_dataset('angles')

        if mean_iters:
            iterations = list(range(params['iterations']))
        else:
            iterations = [iteration]
        
        if debug:
            criterion = torch.nn.MSELoss()
        
        output_array = []
        for current_iteration in iterations:
            model, model_state, _, _, _, _ = load_model_and_reaching(folder, current_iteration)
            current_outputs = []

            W_fb = fb_nuc_norm(model)
            gains = [value / W_fb for value in gain]

            for current_gain in gains:
                model.gain = current_gain
                outputs = model(
                    inputs,
                    input_2=theta_targets,
                    input_2_type='angles',
                    return_thetas=False
                )
                current_outputs.append(outputs.detach().cpu().numpy())
                if debug:
                    loss = criterion(outputs, targets)
                    print(f"model: {folder}, current iteration/gain: {current_iteration}/{current_gain} (norm={W_fb}), loss: ", loss.detach().item())

            output_array.append(np.stack(current_outputs, axis=0))

        output_array = np.stack(output_array, axis=0)
        mean_outputs = output_array.mean(axis=0)

        n_tasks = mean_outputs.shape[2]
        colors = cm.rainbow(np.linspace(0, 1, n_tasks))
        alpha_values = np.linspace(0.25, 1.0, len(gain)) if len(gain)>1 else [1]

        series = []
        for task in range(n_tasks - 1, -1, -1):
            gains_series = []
            for j, value in enumerate(gain):
                current_label = None
                if task == 1:
                    if label is not None and j == len(gain) - 1:
                        current_label = label
                    elif label is None:
                        current_label = 'output (strong feedback)' if value > 1 else 'output (weak feedback)'

                gains_series.append({
                    'name': f'task{task}_gain{j}',
                    'gain': float(value),
                    'alpha': float(alpha_values[j]),
                    'traj_x': mean_outputs[j, :, task, 0].tolist(),
                    'traj_y': mean_outputs[j, :, task, 1].tolist(),
                    'label': current_label,
                    'zorder': 3 + j,
                })

            series.append({
                'name': f'task{task}',
                'task': int(task),
                'color': colors[task].tolist(),
                'unperturbed_x': float(un_targets[-1, task, 0]),
                'unperturbed_y': float(un_targets[-1, task, 1]),
                'perturbed_x': float(targets[-1, task, 0]),
                'perturbed_y': float(targets[-1, task, 1]),
                'gains': gains_series,
            })
            series[-1]['components'] = [
                {'component': g['name'].split('_')[-1], 'x': g['traj_x'], 'y': g['traj_y']}
                for g in gains_series
            ] + [
                {'component': 'unperturbed_target', 'x': [series[-1]['unperturbed_x']], 'y': [series[-1]['unperturbed_y']]},
                {'component': 'perturbed_target', 'x': [series[-1]['perturbed_x']], 'y': [series[-1]['perturbed_y']]},
            ]

        return {
            'figuretype': 'outputs',
            'panels': [{
                'metric': 'trajectory',
                'series': series,
            }],
            'meta': {
                'folder': folder,
                'gain': list(gain),
                'pert_params': pert_params,
                'title': title,
                'linestyle': linestyle,
                'iteration': iteration,
                'mean_iters': mean_iters,
                'plot_targets': plot_targets,
                'label': label,
                'add_feedback_legend': add_feedback_legend,
                'draw_main_legend': draw_main_legend,
                'figsize': figsize,
                'dpi': dpi,
                'lw': lw,
                'xlim': [-0.150, 0.2],
                'ylim': [0.2, 0.45],
                'markersize_targets': 360,
                'markersize_final': 120,
                'target_edgewidth': 3.2,
                'final_edgewidth': 1.05,
            }
        }

    def render(spec):
        meta = spec['meta']

        if ax is None:
            fig, local_ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        else:
            local_ax = ax
            fig = local_ax.get_figure()

        if meta['title']:
            local_ax.set_title(meta['title'])

        for task_series in spec['panels'][0]['series']:
            color = task_series['color']
            task = task_series['task']

            if meta['plot_targets']:
                local_ax.scatter(
                    task_series['unperturbed_x'],
                    task_series['unperturbed_y'],
                    linewidths=meta['target_edgewidth'],
                    facecolor='none',
                    alpha=0.6,
                    marker='^',
                    s=meta['markersize_targets'],
                    zorder=2 if task != 0 else 15,
                    edgecolor=color,
                    label='targets (baseline)' if task == 1 else None,
                )
                local_ax.scatter(
                    task_series['perturbed_x'],
                    task_series['perturbed_y'],
                    alpha=1.0,
                    marker='s',
                    s=meta['markersize_targets'],
                    edgecolor=color,
                    facecolor='none',
                    linewidths=meta['target_edgewidth'],
                    zorder=3 if task != 0 else 15,
                    label='target (perturbed)' if task == 1 else None,
                )

            for gain_series in task_series['gains']:
                local_ax.plot(
                    gain_series['traj_x'],
                    gain_series['traj_y'],
                    alpha=1,
                    lw=meta['lw'],
                    color='w',
                    linestyle=meta['linestyle'],
                    zorder=gain_series['zorder'],
                )
                local_ax.plot(
                    gain_series['traj_x'],
                    gain_series['traj_y'],
                    alpha=gain_series['alpha'],
                    lw=meta['lw'],
                    color=color,
                    linestyle=meta['linestyle'],
                    zorder=gain_series['zorder'],
                    label=gain_series['label'],
                )

                local_ax.scatter(
                    gain_series['traj_x'][-1],
                    gain_series['traj_y'][-1],
                    color='white',
                    alpha=1,
                    marker='o',
                    s=0.9 * meta['markersize_final'],
                    edgecolor='white',
                    linewidths=meta['final_edgewidth'],
                    zorder=4,
                )
                local_ax.scatter(
                    gain_series['traj_x'][-1],
                    gain_series['traj_y'][-1],
                    color=color,
                    alpha=gain_series['alpha'],
                    marker='o',
                    s=meta['markersize_final'],
                    edgecolor='k',
                    linewidths=meta['final_edgewidth'],
                    zorder=4,
                )

        if meta['add_feedback_legend']:
            local_ax.plot([], [], color='#aaaaaa', alpha=0.6, lw=meta['lw'], label='weak feedback')
            local_ax.plot([], [], color='#444444', alpha=1.0, lw=meta['lw'], label='strong feedback')

        if meta['draw_main_legend']:
            handles, labels_ = local_ax.get_legend_handles_labels()
            bylabel = dict(zip(labels_, handles))
            local_ax.legend(
                bylabel.values(),
                bylabel.keys(),
                fontsize=18,
                labelspacing=1,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
            )

        local_ax.set_xlim(*meta['xlim'])
        local_ax.set_ylim(*meta['ylim'])
        local_ax.set_aspect('equal')
        local_ax.set_xticks([])
        local_ax.set_yticks([])
        local_ax.spines['top'].set_visible(False)
        local_ax.spines['right'].set_visible(False)
        local_ax.spines['bottom'].set_visible(False)
        local_ax.spines['left'].set_visible(False)

        return fig, local_ax

    if spec_name is not None:
        spec = load_or_build_spec(spec_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    else:
        spec = prepare()
    fig, ax = render(spec)
    return fig, ax, spec


def get_neural_activity(folder, name, iteration=None, apply_pert=False, gain=None, perturbation_params=None):

    def load_model(params, reaching):
        in_features = reaching.input_size[2]
        rnn_hidden_size = params['rnn_hidden_size']
        params['net_params']['device'] = 'cpu'
        model = CerebelloCorticalNet(in_features, rnn_hidden_size, **params['net_params'])
        return model

    def load_weights(model, path):
        cortex_sd_path = os.path.join(path, 'cortex_state_dict.pkl')
        model.cortex.load_state_dict(load_var(cortex_sd_path))
        return model

    def extract_neural_activity(path, model, reaching, model_state):
        inputs, t_targets = reaching.generate_perturbed_dataset('angles')
        model = load_weights(model, path)
        model.eval(**model_state)
        outputs, neural_activity = model(inputs, input_2=t_targets, input_2_type='angles', return_hidden=True)
        return outputs, neural_activity, t_targets

    path = os.path.join(folder, name)
    with open(os.path.join(path, 'params.json'), 'r') as f:
        params = json.load(f)

    reaching = ReachingDataset(params['dataset_params'])
    if apply_pert and perturbation_params is not None:
        reaching.apply_perturbation(perturbation_params)

    model = load_model(params, reaching)
    model_state = params['model_state']
    if gain:
        model.gain = gain

    if iteration is not None:
        paths = [os.path.join(path, iteration)]
    else:
        paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    neural_activities = []
    outputs = []
    t_targets = None
    for path in paths:
        output, neural_activity, t_targets = extract_neural_activity(path, model, reaching, model_state)
        neural_activities.append(neural_activity)
        outputs.append(output)

    neural_activities = torch.stack(neural_activities)
    outputs = torch.stack(outputs)
    return neural_activities, outputs, t_targets


class NeuralActivityExtractor:
    def __init__(self, folder, name, device=torch.device('cpu'), dtype='float32'):
        self.folder = folder
        self.name = name
        self.device = device
        self.dtype = dtype
        self.path = os.path.join(folder, name)
        self.params = self._load_params()
        self.reaching = ReachingDataset(self.params['dataset_params'])
        self.model = self._load_model()
        self.model_state = self.params['model_state']

    def _load_params(self):
        try:
            with open(os.path.join(self.path, 'params.json'), 'r') as f:
                params = json.load(f)
        except:
            with open(os.path.join(self.path, 'config.json'), 'r') as f:
                params = json.load(f)
        params["dataset_params"]["device"] = self.device
        params["net_params"]["device"] = self.device
        params["dataset_params"]["dtype"] = self.dtype
        params["net_params"]["dtype"] = self.dtype
        return params

    def _load_model(self):
        in_features = self.reaching.input_size[2]
        rnn_hidden_size = self.params['rnn_hidden_size']
        model = CerebelloCorticalNet(in_features, rnn_hidden_size, **self.params['net_params'])
        return model

    def _load_weights(self, model, path):
        cortex_sd_path = os.path.join(path, 'cortex_state_dict.pkl')
        model.cortex.load_state_dict(load_var(cortex_sd_path))
        return model

    def _extract_neural_activity(self, path, model, reaching, model_state):
        inputs, t_targets = reaching.generate_perturbed_dataset('angles')
        self.t_targets = t_targets
        model = self._load_weights(model, path)
        model.eval(**model_state)
        outputs, neural_activity = model(inputs, input_2=t_targets, input_2_type='angles', return_hidden=True)
        return outputs, neural_activity

    def apply_perturbation(self, perturbation_params):
        self.reaching.apply_perturbation(perturbation_params)

    def set_gain(self, gain):
        self.model.gain = gain

    def get_neural_activity(self, iteration=None, gain=None, perturbation_params=None):
        if perturbation_params is not None:
            self.pert = True
            self.apply_perturbation(perturbation_params)
        else:
            self.pert = False

        if gain:
            self.set_gain(gain)

        if iteration is not None:
            paths = [os.path.join(self.path, iteration)]
        else:
            paths = [os.path.join(self.path, d) for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]

        neural_activities = []
        outputs = []
        for path in paths:
            output, neural_activity = self._extract_neural_activity(path, self.model, self.reaching, self.model_state)
            neural_activities.append(neural_activity.detach())
            outputs.append(output.detach())

        inputs, targets = self.reaching.generate_perturbed_dataset('cart')
        self.targets = targets
        criterion = nn.MSELoss()
        loss = np.mean([criterion(output, targets).item() for output in outputs])
        print(f"Gain = {gain}: loss = {loss}")
        neural_activities = torch.stack(neural_activities).squeeze(0)
        outputs = torch.stack(outputs).squeeze(0)
        return neural_activities, outputs


def compute_pca_unfolded(data, n_pcs=None):
    assert data.ndim == 3, "Data must be 3 dimensional"
    assert isinstance(n_pcs, int) or n_pcs is None, "If provided, n_pcs must be an integer"

    if n_pcs is None:
        n_pcs = data.shape[2]

    unfolded_data = np.concatenate([d for d in data])
    _, _, V = np.linalg.svd(unfolded_data, full_matrices=False)

    if n_pcs is not None:
        V = V[:n_pcs]

    proj_data = data @ V.T
    return proj_data, V


def normalize_neural_activities(neural_activities):
    return torch.stack([
        (neural_activities[:, :, n] - neural_activities[:, :, n].min()) /
        (neural_activities[:, :, n].max() - neural_activities[:, :, n].min())
        for n in range(neural_activities.shape[2])
    ], axis=2)


def project(neural_activities, V):
    return torch.stack([
        neural_activities[k] @ V.T
        for k in range(neural_activities.shape[0])
    ]).numpy()


def figure_feedback_latent_3d(
    loading_folder,
    name,
    iteration,
    perturbation_params,
    gain,
    save_name,
    title,
    start_index=30,
    add_target_traces=False,
    overlay_specs=None,
    lims=None,
    lw=3.2,
    order=[5, 4, 6, 3, 7, 0, 8, 1, 2],
    view=(17, -77),
    fontsize=22,
    cmap_name='rainbow',
    color_range=(0, 1),
    pcs=np.array([0, 1, 2]),
    alpha=1,
    lightness=1,
    label=None,
    figsize=(13, 13),
    dpi=200,
):
    def prepare():
        extractor = NeuralActivityExtractor(loading_folder, name, device=device, dtype=torchdtype)

        nofb_neural_activities, _ = extractor.get_neural_activity(
            perturbation_params=perturbation_params,
            gain=0,
            iteration=iteration
        )
        fb_neural_activities, _ = extractor.get_neural_activity(
            perturbation_params=perturbation_params,
            gain=1,
            iteration=iteration
        )
        strong_fb_neural_activities, _ = extractor.get_neural_activity(
            perturbation_params=perturbation_params,
            gain=30,
            iteration=iteration
        )

        norm_nofb_neural_activities = normalize_neural_activities(nofb_neural_activities)
        norm_fb_neural_activities = normalize_neural_activities(fb_neural_activities)
        norm_strong_fb_neural_activities = normalize_neural_activities(strong_fb_neural_activities)

        data = torch.cat(
            [norm_nofb_neural_activities, norm_fb_neural_activities, norm_strong_fb_neural_activities],
            dim=0
        ).numpy()
        _, V = compute_pca_unfolded(data, n_pcs=6)

        out_layer = extractor.model.cortex.fc
        V_custom = np.stack([out_layer.weight.numpy()[0], out_layer.weight.numpy()[1], V[0]])

        nofb_proj_data = project(nofb_neural_activities, V_custom)
        fb_proj_data = project(fb_neural_activities, V_custom)
        strong_fb_proj_data = project(strong_fb_neural_activities, V_custom)

        proj_map = {
            'nofb': nofb_proj_data,
            'fb': fb_proj_data,
            'strong_fb': strong_fb_proj_data,
        }

        colors = plt.get_cmap('rainbow')(np.linspace(0, 1, 9))
        z_min = lims[2][0] if lims is not None else float(np.min(proj_map[gain][:, :, 2]))

        if overlay_specs is None:
            local_overlay_specs = [{
                'proj_key': gain,
                'start_index': start_index,
                'highlight_time': None,
                'cmap_name': cmap_name,
                'color_range': list(color_range),
                'pcs': pcs.tolist() if hasattr(pcs, 'tolist') else list(pcs),
                'lw': lw,
                'lims': lims,
                'view': view,
                'alpha': alpha,
                'fontsize': fontsize,
                'order': list(order),
                'lightness': lightness,
                'label': label,
            }]
        else:
            local_overlay_specs = []
            for item in overlay_specs:
                local_item = dict(item)
                local_item['pcs'] = local_item['pcs'].tolist() if hasattr(local_item['pcs'], 'tolist') else list(local_item['pcs'])
                local_item['order'] = list(local_item['order'])
                if 'color_range' in local_item:
                    local_item['color_range'] = list(local_item['color_range'])
                if 'view' in local_item:
                    local_item['view'] = list(local_item['view'])
                local_overlay_specs.append(local_item)

        target_series = []
        if add_target_traces:
            for i, trial in enumerate(order):
                xs = extractor.t_targets[:, trial, 0].detach().cpu().numpy().tolist()
                ys = extractor.t_targets[:, trial, 1].detach().cpu().numpy().tolist()
                target_series.append({
                    'trial': int(trial),
                    'x': xs,
                    'y': ys,
                    'z': [float(z_min)] * len(xs),
                    'linestyle': 'dashed',
                    'color': colors[trial].tolist(),
                    'alpha': 0.4,
                    'linewidth': 0.8 * lw,
                    'label': 'Targets' if i == 0 else None,
                })

        overlay_series = []
        for overlay_index, overlay in enumerate(local_overlay_specs):
            proj_data = proj_map[overlay['proj_key']]
            latent = proj_data[overlay['start_index']:]
            cmap = plt.get_cmap(overlay['cmap_name'])
            T, N, _ = latent.shape
            tasks = np.linspace(*overlay['color_range'], N)
            line_colors = cmap(tasks)
            line_colors[:, :3] *= overlay['lightness']

            series = []
            for l, trial in enumerate(overlay['order']):
                xs = latent[:, trial, 0].tolist()
                ys = latent[:, trial, 1].tolist()
                zs = latent[:, trial, 2].tolist()

                if overlay['alpha'] < 1:
                    series.append({
                        'trial': int(trial),
                        'kind': 'trajectory_underlay',
                        'x': xs,
                        'y': ys,
                        'z': zs,
                        'linestyle': 'solid',
                        'color': 'white',
                        'alpha': overlay['alpha'],
                        'linewidth': overlay['lw'],
                        'solid_capstyle': 'round',
                        'label': None,
                        'zorder': l + 2,
                    })

                series.append({
                    'trial': int(trial),
                    'kind': 'trajectory',
                    'x': xs,
                    'y': ys,
                    'z': zs,
                    'linestyle': 'solid',
                    'color': line_colors[trial].tolist(),
                    'alpha': overlay['alpha'],
                    'linewidth': overlay['lw'],
                    'solid_capstyle': 'round',
                    'label': overlay['label'] if trial == 1 else None,
                    'zorder': l + 2,
                })

            projection_series = []
            z_projection = lims[2][0] if lims is not None else float(np.min(latent[:, :, 2]))
            for trial in overlay['order']:
                xs = latent[:, trial, 0].tolist()
                ys = latent[:, trial, 1].tolist()
                projection_series.append({
                    'trial': int(trial),
                    'kind': 'projection',
                    'x': xs,
                    'y': ys,
                    'z': [float(z_projection)] * len(xs),
                    'linestyle': 'solid',
                    'color': line_colors[trial].tolist(),
                    'alpha': 0.3 * overlay['alpha'],
                    'linewidth': 0.8 * overlay['lw'],
                    'solid_capstyle': 'round',
                    'label': None,
                    'zorder': 1,
                })

            overlay_series.append({
                'overlay_index': overlay_index,
                'proj_key': overlay['proj_key'],
                'start_index': overlay['start_index'],
                'highlight_time': overlay['highlight_time'],
                'view': overlay['view'],
                'pcs': overlay['pcs'],
                'fontsize': overlay['fontsize'],
                'series': series,
                'projection_series': projection_series,
            })

        spec = {
            'figuretype': 'feedback_latent_3d',
            'panels': [{
                'metric': 'latent_3d',
                'title': title,
                'series': overlay_series,
                'target_series': target_series,
            }],
            'meta': {
                'save_name': save_name,
                'loading_folder': loading_folder,
                'name': name,
                'iteration': iteration,
                'perturbation_params': perturbation_params,
                'lims': lims,
                'lw': lw,
                'order': list(order),
                'view': list(view),
                'fontsize': fontsize,
                'figsize': figsize,
                'dpi': dpi,
                'z_min': float(z_min),
                'axis_color': 'k',
                'spines': 2,
                'box_aspect': [1, 1, 1.17],
                'zticks': [-7, -6, -5, -4],
                'zticklabels': ['-7.0', '-6.0', '-5.0', '-4.0'],
                'yticks': [1, 1.5, 2],
                'yticklabels': ['1.0', '1.5', '2.0'],
                'legend_fontsize': fontsize,
                'legend_bbox_to_anchor': (0.3, 0.7),
                'subplots_adjust': {'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1},
                'tight_layout': True,
            }
        }
        return spec

    def render(spec):
        meta = spec['meta']
        panel = spec['panels'][0]

        fig = plt.figure(figsize=meta['figsize'], dpi=meta['dpi'])
        fig.suptitle(panel['title'], fontsize=meta['fontsize'])
        ax = fig.add_subplot(111, projection='3d')

        for overlay in panel['series']:
            for s in overlay['series']:
                ax.plot(
                    s['x'], s['y'], s['z'],
                    linestyle=s['linestyle'],
                    color=s['color'],
                    alpha=s['alpha'],
                    linewidth=s['linewidth'],
                    solid_capstyle=s['solid_capstyle'],
                    label=s['label'],
                    zorder=s['zorder']
                )

            for s in overlay['projection_series']:
                ax.plot(
                    s['x'], s['y'], s['z'],
                    linestyle=s['linestyle'],
                    color=s['color'],
                    alpha=s['alpha'],
                    linewidth=s['linewidth'],
                    solid_capstyle=s['solid_capstyle'],
                    label=s['label'],
                    zorder=s['zorder']
                )

        for s in panel['target_series']:
            ax.plot(
                s['x'], s['y'], s['z'],
                linestyle=s['linestyle'],
                color=s['color'],
                alpha=s['alpha'],
                linewidth=s['linewidth'],
                solid_capstyle='round',
                label=s['label']
            )

        if meta['lims'] is not None:
            ax.set_xlim(meta['lims'][0])
            ax.set_ylim(meta['lims'][1])
            ax.set_zlim(meta['lims'][2])

        ax.set_zticks(meta['zticks'])
        ax.set_zticklabels(meta['zticklabels'])
        ax.set_yticks(meta['yticks'])
        ax.set_yticklabels(meta['yticklabels'])

        current_view = panel['series'][-1]['view'] if len(panel['series']) > 0 else meta['view']
        pcs = panel['series'][-1]['pcs'] if len(panel['series']) > 0 else [0, 1, 2]
        current_fontsize = panel['series'][-1]['fontsize'] if len(panel['series']) > 0 else meta['fontsize']

        ax.view_init(elev=current_view[0], azim=current_view[1])

        axis_color = meta['axis_color']
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid'].update({'linewidth': 0})
            axis.line.set_linewidth(meta['spines'])
            axis.line.set_color(axis_color)
            axis._axinfo['tick']['outward_factor'] = 0
            axis._axinfo['tick']['inward_factor'] = 0.2
            axis._axinfo['tick']['size'] = 4
            axis._axinfo['axisline'].update({'linewidth': meta['spines'], 'color': axis_color})
            axis.set_tick_params(pad=13, labelsize=current_fontsize, colors=axis_color)
            axis._axinfo['axisline']['color'] = axis_color
            axis.pane.set_edgecolor('w')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.pane.fill = False
            axis.set_tick_params(pad=13, labelsize=current_fontsize, colors=axis_color)

        ax.xaxis._axinfo['grid'].update(color='none')

        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_edgecolor('w')

        ax.set_xlabel(f'Latent {pcs[0] + 1}', fontsize=current_fontsize, labelpad=20, color=axis_color)
        ax.set_ylabel(f'Latent {pcs[1] + 1}', fontsize=current_fontsize, labelpad=20, color=axis_color)
        ax.set_zlabel(f'Latent {pcs[2] + 1}', fontsize=current_fontsize, labelpad=20, color=axis_color)
        ax.tick_params(axis='both', labelsize=current_fontsize)
        ax.set_box_aspect(meta['box_aspect'])
        ax.legend(fontsize=meta['legend_fontsize'], bbox_to_anchor=meta['legend_bbox_to_anchor'])

        plt.subplots_adjust(**meta['subplots_adjust'])
        if meta['tight_layout']:
            plt.tight_layout()

        return fig, ax

    spec = load_or_build_spec("2e", plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, ax = render(spec)
    return fig, spec

def plot_metric(ax, array, x, label, color, start, stop, plot_single=False, 
                lw=1.5, fill_alpha=0.07, zorder=2, alpha=1, plot_fill=True, 
                ls='solid', smoothing_window=1):
    """
    Plots the median line with shaded min-max range for a given set of data dictionaries.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - array: List of dictionaries containing 'mse' arrays.
    - x_range: Array of x-axis values.
    - label: Label for the median line.
    - color: Color for the line and fill.
    - pert_start: Index where perturbation starts.
    - stop: Index where plotting stops.
    - lw: Line width for the median line.
    - fill_alpha: Transparency for the fill_between area.
    - zorder: Z-order for the plot elements.
    """
    
    if smoothing_window > 1:
        
        array = moving_median(array, smoothing_window)
    
    if not plot_single:
        # Compute statistics across iterations
        median = np.median(array, axis=0)[start:stop]
        min_vals = np.min(array, axis=0)[start:stop]
        max_vals = np.max(array, axis=0)[start:stop]
        
        # Plot median line
        ax.plot(x, median, lw=lw, color=color, label=label, zorder=zorder, alpha=alpha, linestyle=ls)
        
        # Fill between min and max
        if plot_fill:
            ax.fill_between(x, min_vals, max_vals, color=color, alpha=fill_alpha, zorder=zorder-1, lw=0)
    else:
        for i in range(array.shape[0]):
            ax.plot(x, array[i][start:stop], lw=lw, color=color, label=label if i==0 else None, zorder=zorder, alpha=alpha, solid_capstyle='round')

def configure_axes(
    ax,
    start,
    stop,
    xlabel, 
    ylabel, 
    fontsize, 
    spines_lw, 
    ticks_length, 
    lims, 
    minortickson, 
    grid=False, 
    yticks=None, 
    xticks=None, 
    spine_color='red', 
    log=True, 
    angles=False):
    if log:
        ax.set_yscale('log')
        if yticks:
            ax.set_yticks(yticks)
    else:
        if yticks is not None:
            ax.set_yticks(yticks)
    ax.set_ylim(*lims)
    if minortickson:
        if grid:
            ax.grid(which='both', color='gray', linestyle='-', linewidth=0.4)
        ax.minorticks_on()
    ax.set_xlim(1.2*start, 1.2*stop)
    if xticks:
        ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', width=1.3*spines_lw, length=ticks_length, labelsize=fontsize, color=spine_color, labelcolor=spine_color)
    if not angles:
        ax.tick_params(axis='y', which='minor', width=0.5*spines_lw, length=0.5*ticks_length, labelsize=fontsize, color=spine_color, labelcolor=spine_color)
    # ax.tick_params(axis='x', which='minor', width=0.4*spines_lw, length=0.35*ticks_length, labelsize=fontsize, color=spine_color, labelcolor=spine_color)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.3 * spines_lw)
        ax.spines[spine].set_color(spine_color)
    if not angles:
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

        
    ax.set_xlabel(xlabel, fontsize=fontsize, color=spine_color)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=spine_color)

def load_washout_data(output_folder, normalize=True, unperturbed_period=20, update=False):
    cache_path = os.path.join(DATA_DIR, 'metrics', os.path.basename(output_folder), 'data.pkl')
    if not update and os.path.exists(cache_path):
        return load_var(cache_path)

    def _load_config(model_name, compute_std=False):
        model_dir = os.path.join(output_folder, model_name)
        data_dir = os.path.join(model_dir, 'data')
        iterations = sorted(
            [f for f in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()],
            key=int
        )
        assert iterations, f"No iteration folders found in {data_dir}"

        std = None
        if compute_std:
            params_file = os.path.join(model_dir, 'params.json')
            with open(params_file) as f:
                params = json.load(f)
            reaching = ReachingDataset(params['dataset_params'])
            std = reaching.generate_unperturbed_dataset(params['settings']['datatype'])[1].std().item()

        lists = {}
        for it in iterations:
            experiment = Experiment2(data_dir, str(it), load=True)
            print(f"  Loading {model_name}/{it}", end='\r')

            outputs = experiment.train_outputs
            full_loss = [experiment.unperturbed_loss] * (unperturbed_period + 1) + outputs['mse']

            for key in ['mse', 'angles', 'takeoff_angles']:
                if key == 'mse':
                    lists.setdefault(key, []).append(np.array(full_loss))
                elif key in outputs:
                    lists.setdefault(key, []).append(outputs[key])

        print(f"  {model_name}: {len(iterations)} iterations loaded.      ")
        return {'std': std, **{k: np.stack(v) for k, v in lists.items()}}

    print("Loading fb.ae_fixed (fixed controller)...")
    fixed_10 = _load_config('fb.ae_fixed.15', compute_std=True)
    fixed_20 = _load_config('fb.ae_fixed.20', compute_std=True)

    print("Loading RNN (plastic controller)...")
    plastic = _load_config('RNN')

    if normalize:
        std = fixed_10['std']
        print(f"  Normalizing by std: {std:.4f}")
        for config in [fixed_10, fixed_20, plastic]:
            config['mse'] /= std

    result = (fixed_10, fixed_20, plastic)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_var(result, cache_path)
    return result

def figure_washout_metric(
    output_folder,
    metric_key,
    ylabel,
    normalize=True,
    unperturbed_period=20,
    washout_start=50,
    stop=100,
    fontsize=26,
    spines_lw=1.3,
    ticks_length=5,
    minortickson=True,
    lw=3,
    xlabel='epoch',
    spine_color='k',
    xticks=[-20, 0, 50, 100],
    fixed_net_color='gray',
    fixed_lw=None,
    plastic_net_color='k',
    plastic_lw=3.5,
    s_window=5,
    coeff=180 / np.pi,
    lims=None,
    yticks=None,
    add_suptitle=False,
    spec_name=None,
):
    if fixed_lw is None:
        fixed_lw = lw

    UP = unperturbed_period
    P0 = UP + 1
    W0 = P0 + 51
    x = np.arange(stop)
    x_no_pert = np.arange(-UP, 1)

    def prepare():
        fixed_15, fixed_20, plastic = load_washout_data(
            output_folder,
            normalize=normalize,
            unperturbed_period=unperturbed_period,
            update=UPDATE_METRICS,
        )
        fixed = fixed_20

        if metric_key == 'mse':
            series = []
            for data, color, label, zlw, zorder in [
                (fixed['mse'], 'grey', 'fixed controller', fixed_lw, 3),
                (plastic['mse'], plastic_net_color, 'plastic controller', plastic_lw, 2),
            ]:
                ls = 'dashed' if data is plastic['mse'] else 'solid'
                segment_zero = [{'x': x_no_pert.tolist(), 'start': 0, 'stop': UP + 1, 'label': None, 'fill_alpha': 0.0}] if label == 'fixed controller' else []
                series.append({
                    'data': data.tolist(),
                    'color': color,
                    'label': label,
                    'zlw': zlw,
                    'zorder': zorder,
                    'ls': ls,
                    'segments':
                        segment_zero + [
                        {'x': x[1:washout_start].tolist(), 'start': P0 + 1, 'stop': P0 + washout_start, 'label': label, 'fill_alpha': 0.2},
                        {'x': x[washout_start:stop].tolist(), 'start': W0, 'stop': W0 + (stop - washout_start), 'label': None, 'fill_alpha': 0.2},
                    ]
                })
                median_full = np.median(data, axis=0)
                series[-1]['components'] = [
                    {
                        'component': f'segment{idx}',
                        'x': seg['x'],
                        'y': median_full[seg['start']:seg['stop']].tolist(),
                        'raw_y': data[:, seg['start']:seg['stop']].tolist(),
                    }
                    for idx, seg in enumerate(series[-1]['segments'])
                ]

            spec = {
                'figuretype': 'washout_metric',
                'panels': [{
                    'metric': metric_key,
                    'ylabel': 'error (adaptation)',
                    'xlabel': xlabel,
                    'yscale': 'log',
                    'series': series,
                }],
                'meta': {
                    'output_folder': output_folder,
                    'metric_key': metric_key,
                    'ylabel': 'error (adaptation)',
                    'xlabel': xlabel,
                    'fontsize': fontsize,
                    'spines_lw': spines_lw,
                    'ticks_length': ticks_length,
                    'minortickson': minortickson,
                    'lw': lw,
                    'spine_color': spine_color,
                    'xticks': xticks,
                    'lims': lims or (5e-9, 5e-2),
                    'yticks': yticks or [1e-7, 1e-5, 1e-3],
                    'washout_start': washout_start,
                    'stop': stop,
                    'coeff': coeff,
                    'fixed_net_color': fixed_net_color,
                    'fixed_lw': fixed_lw,
                    'plastic_net_color': plastic_net_color,
                    'plastic_lw': plastic_lw,
                    's_window': s_window,
                    'unperturbed_period': unperturbed_period,
                    'normalize': normalize,
                    'add_suptitle': add_suptitle,
                    'angles': False
                }
            }
            return spec

        fixed_15, fixed_20, plastic = load_washout_data(
            output_folder,
            normalize=normalize,
            unperturbed_period=unperturbed_period,
            update=UPDATE_METRICS,
        )
        fixed = fixed_20
        panels = [{
            'metric': metric_key,
            'ylabel': ylabel,
            'xlabel': xlabel,
            'yscale': 'linear',
            'series': [
                {
                    'data': fixed[metric_key].tolist(),
                    'color': fixed_net_color,
                    'label': 'fixed controller',
                    'zlw': fixed_lw,
                    'zorder': 3,
                    'ls': 'solid',
                    'segments': [
                        {'x': np.arange(-25, 1).tolist(), 'start': 0, 'stop': 26, 'label': None, 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * fixed[metric_key][:, washout_start:washout_start+26], s_window).tolist()},
                        {'x': x[:washout_start].tolist(), 'start': 0, 'stop': washout_start, 'label': 'fixed controller', 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * fixed[metric_key][:, :washout_start], s_window).tolist()},
                        {'x': x[washout_start:stop].tolist(), 'start': 0, 'stop': stop - washout_start, 'label': None, 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * fixed[metric_key][:, washout_start:stop], s_window).tolist()},
                    ]
                },
                {
                    'data': plastic[metric_key].tolist(),
                    'color': plastic_net_color,
                    'label': 'plastic controller',
                    'zlw': plastic_lw,
                    'zorder': 2,
                    'ls': 'dashed',
                    'segments': [
                        # {'x': np.arange(-25, 1).tolist(), 'start': 0, 'stop': 26, 'label': None, 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * plastic[metric_key][:, washout_start:washout_start+26], s_window).tolist()},
                        {'x': x[:washout_start].tolist(), 'start': 0, 'stop': washout_start, 'label': 'plastic controller', 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * plastic[metric_key][:, :washout_start], s_window).tolist()},
                        {'x': x[washout_start:stop].tolist(), 'start': 0, 'stop': stop - washout_start, 'label': None, 'fill_alpha': 0.2, 'preprocessed': moving_median(coeff * plastic[metric_key][:, washout_start:stop], s_window).tolist()},
                    ]
                },
            ],
        }]

        for s in panels[0]['series']:
            s['components'] = [
                {
                    'component': f'segment{idx}',
                    'x': seg['x'],
                    'y': np.median(np.array(seg['preprocessed']), axis=0)[seg['start']:seg['stop']].tolist(),
                    'raw_y': np.array(seg['preprocessed'])[:, seg['start']:seg['stop']].tolist(),
                }
                for idx, seg in enumerate(s['segments'])
            ]

        spec = {
            'figuretype': 'washout_metric',
            'panels': panels,
            'meta': {
                'output_folder': output_folder,
                'metric_key': metric_key,
                'ylabel': ylabel,
                'xlabel': xlabel,
                'fontsize': fontsize,
                'spines_lw': spines_lw,
                'ticks_length': ticks_length,
                'minortickson': minortickson,
                'lw': lw,
                'spine_color': spine_color,
                'xticks': xticks,
                'lims': lims,
                'yticks': yticks,
                'washout_start': washout_start,
                'stop': stop,
                'coeff': coeff,
                'fixed_net_color': fixed_net_color,
                'fixed_lw': fixed_lw,
                'plastic_net_color': plastic_net_color,
                'plastic_lw': plastic_lw,
                's_window': s_window,
                'unperturbed_period': unperturbed_period,
                'normalize': normalize,
                'add_suptitle': False,
                'angles': True
            }
        }
        return spec

    def render(spec):
        meta = spec['meta']
        panel = spec['panels'][0]

        fig, axs = plt.subplots(1, 1, figsize=(10.5, 5), dpi=200)
        if meta['add_suptitle']:
            fig.suptitle(f"{panel['ylabel']}", fontsize=meta['fontsize'], color=meta['spine_color'])

        for s in panel['series']:
            building = 'data' in s
            ls = s['ls']
            if not building:
                for idx, seg in enumerate(s['segments']):
                    comp = s['components'][idx]
                    if meta['metric_key'] == 'mse' and s['label'] == 'fixed controller' and idx == 0:
                        axs.plot(comp['x'], comp['y'], lw=s['zlw'], color='k', label=seg['label'],
                                 zorder=s['zorder'], alpha=1, linestyle=ls)
                    axs.plot(comp['x'], comp['y'], lw=s['zlw'], color=s['color'], label=seg['label'],
                             zorder=s['zorder'], alpha=1, linestyle=ls)
            elif meta['metric_key'] == 'mse':
                data = np.array(s['data'])
                for idx, seg in enumerate(s['segments']):
                    if s['label'] == 'fixed controller' and idx == 0:
                        plot_metric(
                            axs, data, np.array(seg['x']), seg['label'], 'k',
                            seg['start'], seg['stop'],
                            lw=s['zlw'], fill_alpha=0., zorder=s['zorder'], alpha=1,
                            plot_fill=False, plot_single=False, ls=ls
                        )
                    plot_metric(
                        axs, data, np.array(seg['x']), seg['label'], s['color'],
                        seg['start'], seg['stop'],
                        lw=s['zlw'], fill_alpha=0.2, zorder=s['zorder'], alpha=1,
                        plot_fill=False, plot_single=False, ls=ls
                    )
                del s['data']
            else:
                data = coeff * np.array(s['data'])
                for idx, seg in enumerate(s['segments']):
                    seg_data = seg.get('preprocessed')
                    if seg_data is None:
                        if idx == 0:
                            seg_data = moving_median(data[:, washout_start:washout_start + 26], s_window)
                        elif idx == 1:
                            seg_data = moving_median(data[:, :washout_start], s_window)
                        else:
                            seg_data = moving_median(data[:, washout_start:stop], s_window)
                    seg_data = np.array(seg_data)
                    xvals = np.array(seg['x'])
                    plot_metric(
                        axs, seg_data, xvals, seg['label'], s['color'],
                        seg['start'], seg['stop'],
                        lw=s['zlw'], fill_alpha=0.2, zorder=s['zorder'], alpha=1,
                        plot_fill=False, plot_single=False, ls=ls
                    )
                del s['data']

        axs.axvspan(0.5 if meta['metric_key'] == 'mse' else 0, washout_start - 0.5, alpha=0.08, color='k', lw=0)
        if meta['metric_key'] == 'mse':
            axs.annotate("perturbation", xy=(11, 1e-8), color=meta['spine_color'], fontsize=0.9 * meta['fontsize'])
        else:
            axs.annotate("perturbation", xy=(11, meta['coeff'] * 0.25), color=meta['spine_color'], fontsize=0.9 * meta['fontsize'])
            axs.annotate("washout", xy=(65, meta['coeff'] * 0.25), color=meta['spine_color'], fontsize=0.9 * meta['fontsize'])

        axs.legend(fontsize=0.8 * meta['fontsize'], labelcolor=meta['spine_color'], bbox_to_anchor=(0.55, 0.5) if meta['metric_key'] != 'mse' else None)
        configure_axes(
            axs, 
            -20, 
            stop, 
            meta['xlabel'], 
            'error (adaptation)',
            meta['fontsize'],
            meta['spines_lw'], 
            meta['ticks_length'], 
            meta['lims'], 
            meta['minortickson'],
            yticks=meta['yticks'],
            xticks=meta['xticks'],
            spine_color=meta['spine_color'],
            log=(meta['metric_key'] == 'mse'),
            angles=meta['angles']
        )

        # configure_axes(
        #     axs,
        #     meta['xlabel'],
        #     panel['ylabel'],
        #     grid=False,
        #     fontsize=meta['fontsize'],
        #     meta['lims'],
        #     yticks=meta['yticks'],
        #     xticks=meta['xticks'],
        #     meta['ticks_length'],
        #     meta['minortickson'],
        #     spine_color=meta['spine_color'],
        #     log=(meta['metric_key'] == 'mse'),
        #     angles=(meta['metric_key'] != 'mse')
        #     meta['spines_lw'],
        # )

        plt.tight_layout()
        return fig, axs

    spec = load_or_build_spec(spec_name, plots_data_folder, prepare, rebuild=REBUILD_SOURCE_DATA)
    fig, axs = render(spec)
    return fig, spec


# %% [markdown]
# # data processing

# %%
_mse_data_loaded = None

def get_mse_data():
    global _mse_data_loaded
    if _mse_data_loaded is None:
        data = load_metrics_pretraining(load_folder, norm=True, norm_mode='normal', update=UPDATE_METRICS)
        data = data.loc[data.index.get_level_values('metric') == 'mse', :]
        mse_data = data.reset_index(level='metric', drop=True)

        pr_print(['bold','purple'], "List of pretraining models (CCnet):\n")
        multi_index_df = mse_data.index.to_frame(index=False)
        extracted_levels = multi_index_df[['order', 'name']]
        unique_combinations = extracted_levels.drop_duplicates()
        print(unique_combinations)

        from functools import reduce
        pr_print(['bold','purple'], "\n\nSelected network as pretrained models (CCnet):\n")

        # selection = reduce(np.add, [mse_data.index.get_level_values('order')==2])
        # sims = mse_data.loc[selection]
        multi_index_df = mse_data.index.to_frame(index=False)
        extracted_levels = multi_index_df[['order', 'name']]
        unique_combinations = extracted_levels.drop_duplicates()

        name = unique_combinations.iloc[0]['name']
        order = unique_combinations.iloc[0]['order']
        print('Plant, correction:\t\t',f' {name} ({order})')

        _mse_data_loaded = mse_data
    return _mse_data_loaded


output_folder = os.path.join(DATA_DIR, 'metrics', os.path.basename(load_folder), 'trainings_metrics')

_alignment_data_loaded = None

def get_alignment_data():
    global _alignment_data_loaded
    if _alignment_data_loaded is None:
        if UPDATE_METRICS:
            compute_alignment_pool(load_folder, output_folder, num_jobs=1, update=UPDATE_METRICS)
        data = load_metrics_pretraining(load_folder, norm=True, norm_mode='normal', update=UPDATE_METRICS)
        data = data.loc[data.index.get_level_values('metric') == 'mse', :]
        _alignment_data_loaded = data.reset_index(level='metric', drop=True)
    return _alignment_data_loaded


# %% [markdown]
# # pretraining loss (S2a)

# %%
fig, spec = figure_losses(
    label_names={
        '1000.fb.r_init': "trained",
        'ae_fixed': "pseudoinverse",
    },
    selection=['no_plant', False],
    steps=40,
    smoothing_window=100,
    fill_smoothing=(10, 10),
    start=0,
    lims=(4e-9, 10),
    figsize=(6, 3),
    dpi=200,
    alpha=0.5,
    alpha_fill=0.3,
    lw=0.7,
    fontsize=15,
    title=None,
    cool=True,
)

plt.tick_params(axis='both', which='both', labelsize=14)
plt.ylabel('error (baseline)', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.grid(False)

save_plot(
    fig, "S2a", plots_folder,
    savepdf=save_pdf,
    savetiff=save_tiff,
    savedata=save_data,
    data=spec,
    plotsdatafolder=plots_data_folder,
    dpi=400,
    transparent=True,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Alignment (S2b-e)

# %%
label_names = {
        "1": "Fb-Backprop, no-delay",
        "2": "fixed $W^{+}$"
    }

fig, spec = figure_alignment(
    folder=output_folder,
    variables=[
        'sub_angle_fb_in',
        'sub_angle_fb_out',
        'sub_angle_out_subrec',
        "sub_angle_fb_subrec"
    ],
    labels=label_names,
    label_q="order",
    smoothing=10,
    gaussian_smoothing=0,
    ema_smoothing=0,
    fill=True,
    fill_smoothing=(1, 1),
    step=100,
    fontsize=14,
    figsize=(6, 3),
    dpi=200
)

save_plot(
    fig, "S2b-e", plots_folder,
    savepdf=save_pdf,
    savetiff=save_tiff,
    savedata=save_data,
    data=spec,
    plotsdatafolder=plots_data_folder,
    dpi=400,
    transparent=True,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Systematic error (S3)

# %%
fig_name = "S3"

effective_gains = [0, 4, 10, 20, 50]

labels = {
    '0.fb_lr-4.1000.fb.r_init': 'trained',
    '2.fb.ae_fixed': 'pseudoinverse',
}

save_names_by_label = {
    'trained': {
        'angle': 'c',
        'translation': 'e',
        'amplitude_factor': 'g',
    },
    'pseudoinverse': {
        'angle': 'd',
        'translation': 'f',
        'amplitude_factor': 'h',
    },
}

titles_by_perturbation = {
    'angle': 'Angles',
    'translation': 'Translations',
    'amplitude_factor': 'Radius changes',
}


# ====================================
#    Angles
# ====================================

for name, label in labels.items():
    print(f"{titles_by_perturbation['angle']}:", name)

    fig, spec = figure_systematic_error(
        name=name,
        model_folder=load_folder,
        pert_type='angle',
        values=np.arange(-90, 91, 10),
        effective_gains=effective_gains,
        figsize=(17, 5),
        lw=2.3,
        markersize=8,
        fontsize=22,
        average_method='median',
        ylims=[-0.002, 0.022],
        spines_lw=1,
        yticks=[1e-2, 1e-4, 1e-6, 1e-8],
        xlims=(-100, 100),
        ytick_order=1e-3,
        spec_name=fig_name + save_names_by_label[label]['angle'],
    )

    save_plot(
        fig,
        fig_name + save_names_by_label[label]['angle'],
        plots_folder,
        savepdf=save_pdf,
        savetiff=save_tiff,
        savedata=save_data,
        data=spec,
        plotsdatafolder=plots_data_folder,
        dpi=400,
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()


# ====================================
#    Translations
# ====================================

for name, label in labels.items():
    print(f"{titles_by_perturbation['translation']}:", name)

    fig, spec = figure_systematic_error(
        name=name,
        model_folder=load_folder,
        pert_type='translation',
        values=[[x, x] for x in np.linspace(0, -0.2, 11)],
        effective_gains=effective_gains,
        figsize=(17, 5),
        lw=2.3,
        markersize=8,
        fontsize=22,
        average_method='median',
        ylims=[1e-9, 1],
        spines_lw=1,
        yticks=[1e-2, 1e-4, 1e-6, 1e-8],
        xlims=(-0.03, 0.33),
        ytick_order=1e-2,
        spec_name=fig_name + save_names_by_label[label]['translation'],
    )

    save_plot(
        fig,
        fig_name + save_names_by_label[label]['translation'],
        plots_folder,
        savepdf=save_pdf,
        savetiff=save_tiff,
        savedata=save_data,
        data=spec,
        plotsdatafolder=plots_data_folder,
        dpi=400,
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()


# ====================================
#    Radius changes
# ====================================

for name, label in labels.items():
    print(f"{titles_by_perturbation['amplitude_factor']}:", name)

    fig, spec = figure_systematic_error(
        name=name,
        model_folder=load_folder,
        pert_type='amplitude_factor',
        values=np.linspace(0.5, 1.5, 11),
        effective_gains=effective_gains,
        figsize=(17, 5),
        lw=2.3,
        markersize=8,
        fontsize=22,
        average_method='median',
        ylims=[1e-9, 0.1],
        spines_lw=1,
        yticks=[1e-2, 1e-4, 1e-6, 1e-8],
        xlims=(0.4, 1.7),
        ytick_order=1e-3,
        spec_name=fig_name + save_names_by_label[label]['amplitude_factor'],
    )

    save_plot(
        fig,
        fig_name + save_names_by_label[label]['amplitude_factor'],
        plots_folder,
        savepdf=save_pdf,
        savetiff=save_tiff,
        savedata=save_data,
        data=spec,
        plotsdatafolder=plots_data_folder,
        dpi=400,
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()

# %% [markdown]
# # Error vs gain (2f, S4c)

# %%
values_dict = {
    'short': [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60],
    'long': [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 100, 500, 1000]
}

markers = {
    '0.fb_lr-4.1000.fb.r_init': 's',
    '2.fb.ae_fixed': 'o'
}

labels = {
    '0.fb_lr-4.1000.fb.r_init': 'trained',
    '2.fb.ae_fixed': 'pseudoinverse'
}

save_name = {
    'short': "2f",
    'long': "S4c"
}


for version, values in values_dict.items():
    fig, spec = figure_feedback_projection_amplitude(
        values=values,
        version=version,
        load_folder=load_folder,
        angles=[40],
        order='nuc',
        average_method='median',
        figsize=(10.5, 5),
        dpi=200,
        fontsize=26,
        lw=2,
        logy=True,
        markersize=11,
        markers=markers,
        labels=labels,
        spec_name=save_name[version],
    )
    if version =='long':
        plt.axvline(90,c='r',lw=3,label="plotted gain")

    save_plot(
        fig,
        save_name[version],
        plots_folder,
        savepdf=save_pdf,
        savetiff=save_tiff,
        savedata=save_data,
        data=spec,
        plotsdatafolder=plots_data_folder,
        dpi=200,
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

# %% [markdown]
# # Output Trajectories

# %% [markdown]
# ### 2d

# %%
labels = {
    '0.fb_lr-4.1000.fb.r_init': 'trained',
    '2.fb.ae_fixed': 'pseudoinverse'
}


# figure 1
title = '2d'

for model in ['0.fb_lr-4.1000.fb.r_init', '2.fb.ae_fixed']:
    name = f'{title}_{labels[model]}'
    fig, ax, spec = figure_outputs(
        folder=os.path.join(load_folder, model),
        gain=[0.1, 30],
        pert_params={
            "angle": 40,
            "amplitude_factor": 1.0,
            "translation": [0.0, 0.0]
        },
        title='',
        linestyle='solid',
        iteration=0,
        mean_iters=False,
        plot_targets=True,
        label=None,
        add_feedback_legend=False,
        draw_main_legend=True,
        figsize=(10, 10),
        dpi=200,
        lw=4.2,
        ax=None,
        debug=False,
        spec_name=name,
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
        dpi=400,
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)



# %%
# figure 2
title = 'S3'
gains=[1, 50]

for deg, panel in zip([10, 60], ['a','b']):
    name = f'{title}{panel}'
    fig, ax, spec_underlay = figure_outputs(
        folder=os.path.join(load_folder, '0.fb_lr-4.1000.fb.r_init'),
        gain=gains,
        pert_params={
            "angle": deg,
            "amplitude_factor": 1.0,
            "translation": [0.0, 0.0]
        },
        title='',
        linestyle='dashed',
        iteration=0,
        mean_iters=False,
        plot_targets=True,
        label=labels['0.fb_lr-4.1000.fb.r_init'],
        add_feedback_legend=True,
        draw_main_legend=False,
        figsize=(10, 10),
        dpi=400,
        lw=3.2,
        ax=None,
        debug=False,
        spec_name=f"{name}_underlay",
    )
    if save_data:
        save_var(spec_underlay, os.path.join(plots_data_folder, f"{name}_underlay_spec.pkl"))

    fig, ax, spec = figure_outputs(
        folder=os.path.join(load_folder, '2.fb.ae_fixed'),
        gain=gains,
        pert_params={
            "angle": deg,
            "amplitude_factor": 1.0,
            "translation": [0.0, 0.0]
        },
        title='',
        linestyle='solid',
        iteration=0,
        mean_iters=False,
        plot_targets=False,
        label=labels['2.fb.ae_fixed'],
        add_feedback_legend=False,
        draw_main_legend=True,
        figsize=(10, 10),
        dpi=400,
        lw=3.2,
        ax=ax,
        debug=False,
        spec_name=name,
    )

    # Combined spec (both models) for a single source-data CSV; the saved
    # *_spec.pkl files stay split (main here, underlay separately) so replay
    # renders the dashed underlay + solid main correctly.
    for s in spec_underlay['panels'][0]['series']:
        s['name'] = f"0.fb_lr-4.1000.fb.r_init_{s['name']}"
    combined = dict(spec)
    combined['panels'] = [dict(spec['panels'][0])]
    combined['panels'][0]['series'] = spec_underlay['panels'][0]['series'] + spec['panels'][0]['series']

    save_plot(
        fig,
        name,
        plots_folder,
        savepdf=save_pdf,
        savetiff=save_tiff,
        savedata=save_data,
        data=spec,
        csv_data=combined,
        plotsdatafolder=plots_data_folder,
        dpi=400,
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)

# %%
# figure 3
title = 'S4'

for model, panel in [
    ('0.fb_lr-4.1000.fb.r_init', 'a'),
    ('2.fb.ae_fixed', 'b'),
]:
    name = f'{title}{panel}'
    fig, ax, spec = figure_outputs(
        folder=os.path.join(load_folder, model),
        gain=[0, 90],
        pert_params={
            "angle": 40,
            "amplitude_factor": 1.0,
            "translation": [0.0, 0.0]
        },
        title='',
        linestyle='solid',
        iteration=0,
        mean_iters=False,
        plot_targets=True,
        label=None,
        add_feedback_legend=False,
        draw_main_legend=True,
        figsize=(10, 10),
        dpi=200,
        lw=3.6,
        ax=None,
        spec_name=name,
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
        dpi=400,
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)

# %% [markdown]
# # Neural activity

# %%
loading_folder = load_folder
name = '2.fb.ae_fixed'

perturbation_params = {
    'angle': 20,
    'amplitude_factor': 1.,
    'translation': [0., 0.]
}

lims = [
    [0.02, 1.30],
    [0.8, 2.57],
    [-7.9, -3.2],
]
lw = 3.2
order = [5, 4, 6, 3, 7, 0, 8, 1, 2]
view = (25, -77)

fig, spec = figure_feedback_latent_3d(
    loading_folder=loading_folder,
    name='2.fb.ae_fixed',
    iteration='3',
    perturbation_params=perturbation_params,
    gain='strong_fb',
    save_name='Neur_act_StrongFB',
    title='',
    add_target_traces=True,
    overlay_specs=[{
        'proj_key': 'strong_fb',
        'start_index': 30,
        'highlight_time': None,
        'cmap_name': 'rainbow',
        'color_range': (0, 1),
        'pcs': np.array([0, 1, 2]),
        'lw': lw,
        'lims': lims,
        'view': view,
        'alpha': 1,
        'fontsize': 26,
        'order': order,
        'lightness': 1,
        'label': 'Strong feedback',
    }],
    lims=lims,
    lw=lw,
    order=order,
    view=view,
    fontsize=26,
    label='Strong feedback',
)
save_plot(
    fig,
    '2e',
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


# %% [markdown]
# # PRE WASHOUT

# %%
output_folder = os.path.join(DATA_DIR, 'pre.washout')

fig, spec = figure_washout_metric(
    output_folder=output_folder,
    metric_key='mse',
    ylabel='error (adaptation)',
    normalize=True,
    unperturbed_period=20,
    washout_start=50,
    stop=100,
    fontsize=26,
    spines_lw=1.3,
    coeff=180 / np.pi,
    ticks_length=5,
    minortickson=True,
    lw=3,
    xlabel='epoch',
    spine_color='k',
    xticks=[-20, 0, 50, 100],
    fixed_net_color='gray',
    fixed_lw=3,
    plastic_net_color='k',
    plastic_lw=3.5,
    s_window=5,
    lims=(5e-9, 5e-2),
    yticks=[1e-7, 1e-5, 1e-3],
    spec_name="2g",
)
save_plot(
    fig,
    "2g",
    plots_folder,
    savepdf=save_pdf,
    savetiff=save_tiff,
    savedata=save_data,
    data=spec,
    plotsdatafolder=plots_data_folder,
    dpi=200,
    transparent=True,
    bbox_inches='tight'
)
plt.show()

output_folder = os.path.join(DATA_DIR, 'pre.washout')

metric_key = 'takeoff_angles'
ylabel = 'mean takeoff angle error'

fig, spec = figure_washout_metric(
    output_folder=output_folder,
    metric_key=metric_key,
    ylabel=ylabel,
    normalize=True,
    unperturbed_period=20,
    washout_start=50,
    stop=100,
    fontsize=26,
    spines_lw=1.3,
    coeff=180 / np.pi,
    ticks_length=5,
    minortickson=False,
    lw=3,
    xlabel='epoch',
    spine_color='k',
    xticks=[-20, 0, 50, 100],
    fixed_net_color='gray',
    fixed_lw=3,
    plastic_net_color='k',
    plastic_lw=3.5,
    s_window=5,
    lims=(-30, 30),
    yticks=[-20, 0, 20],
    spec_name="S5",
)
save_plot(
    fig,
    "S5",
    plots_folder,
    savepdf=save_pdf,
    savetiff=save_tiff,
    savedata=save_data,
    data=spec,
    plotsdatafolder=plots_data_folder,
    dpi=200,
    transparent=True,
    bbox_inches='tight'
)
plt.show()


