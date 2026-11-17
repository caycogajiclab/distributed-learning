"""Plotting helpers for demo.ipynb. Imported by the notebook; no need to read them to follow the demo."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def plot_model_schematic(mode="compound"):
    "Block diagram. mode='controller': error-feedback loop. mode='compound': controller + adapter."
    compound = mode == "compound"
    fig, ax = plt.subplots(figsize=(10, 3.9 if compound else 3.1))
    ax.set_xlim(0, 12); ax.set_ylim(0.1, 3.9 if compound else 3.4); ax.set_aspect("equal"); ax.axis("off")
    blue, orange, grey, red, gold = "#dbe7f5", "#fbe3c6", "#eeeeee", "#b03a2e", "#b9770e"

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=fc, ec="k", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(p0, p1, color="k", rad=0.0, ls="-"):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
            arrowstyle="-|>", lw=1.4, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))

    def line(xs, ys, color="k"):
        ax.plot(xs, ys, color=color, lw=1.4, solid_capstyle="butt")

    def node(x, y, text):
        ax.add_patch(Circle((x, y), 0.28, fc="white", ec="k", lw=1.2, zorder=3))
        ax.text(x, y - 0.03, text, ha="center", va="center", fontsize=13, zorder=4)

    y0 = 2.6 if compound else 1.5
    box(0.1, y0 - 0.4, 1.8, 0.8, "task input\n(target cue)", grey)
    box(2.7, y0 - 0.55, 3.2, 1.1, "Controller\nrecurrent network (cortex)", blue)
    box(8.4, y0 - 0.4, 1.5, 0.8, "Plant\n(arm)", grey)
    box(10.5, y0 - 0.4, 1.4, 0.8, "hand\n(x, y)", grey)
    arrow((1.9, y0), (2.7, y0)); arrow((9.9, y0), (10.5, y0))

    if not compound:
        arrow((5.9, y0), (8.4, y0))
        ax.text(7.15, y0 - 0.12, "motor command (joint angles)", ha="center", va="top", fontsize=8.5)
        yn = 2.85
        line([7.0, 7.0], [y0, yn - 0.28]); ax.plot([7.0], [y0], "ko", ms=4)
        node(7.0, yn, "−"); ax.text(7.45, yn + 0.02, "+", fontsize=12, va="center", color="k")
        ax.text(9.2, yn, "target angles", ha="left", va="center", fontsize=9.5)
        arrow((9.15, yn), (7.58, yn))
        arrow((7.0, y0), (7.0, yn - 0.28), color=red)
        line([6.72, 4.3], [yn, yn], color=red); arrow((4.3, yn), (4.3, y0 + 0.55), color=red)
        ax.text(5.5, yn + 0.12, "error feedback × gain", ha="center", fontsize=9.5, color=red)
    else:
        node(7.4, y0, "+")
        arrow((5.9, y0), (7.12, y0)); arrow((7.68, y0), (8.4, y0))
        ax.text(6.5, y0 + 0.12, "command", ha="center", va="bottom", fontsize=8.5)
        box(2.7, 0.35, 3.2, 1.1, "Adapter\nfeed-forward network (cerebellum)", orange)
        line([1.0, 1.0], [y0 - 0.4, 0.9]); arrow((1.0, 0.9), (2.7, 0.9))
        ax.text(1.85, 1.0, "task input", ha="center", fontsize=8.5)
        arrow((4.3, y0 - 0.55), (4.3, 1.45)); ax.text(4.45, 1.85, "copy of command", fontsize=8.5, va="center")
        arrow((5.9, 0.9), (7.4, y0 - 0.28), color=gold, rad=-0.25)
        ax.text(7.6, 1.1, "readout:\ncorrection added\nto the command", fontsize=8.5, color=gold, va="center")
        arrow((2.7, 1.2), (2.7, y0 - 0.45), color=red, rad=-0.55)
        ax.text(1.85, 1.95, "feedback\n(× gain)", ha="center", va="center", fontsize=8.5, color=red)
    plt.tight_layout()
    return fig


def plot_task(targets_ref, targets_rot, angle, n_arrows=2):
    "Unperturbed vs. rotated targets. Arrows show the rotation around the common starting point."
    fig, ax = plt.subplots(figsize=(4, 4))
    t0, t1 = _np(targets_ref), _np(targets_rot)
    for k in range(t0.shape[1]):
        ax.plot(t0[:, k, 0], t0[:, k, 1], color="0.7", lw=2, label="unperturbed" if k == 0 else None)
        ax.plot(t1[:, k, 0], t1[:, k, 1], color="C3", lw=2, label=f"rotated {angle}°" if k == 0 else None)
    start = t0[0].mean(0)                                       # all reaches share the same starting point
    for i, k in enumerate(np.linspace(0, t0.shape[1], n_arrows, endpoint=False).astype(int)):
        v0, v1 = t0[-1, k] - start, t1[-1, k] - start
        a0 = np.arctan2(v0[1], v0[0])
        da = (np.arctan2(v1[1], v1[0]) - a0 + np.pi) % (2 * np.pi) - np.pi     # signed rotation in (-pi, pi]
        th = a0 + np.linspace(0, da, 40)
        arc = start[:, None] + 0.75 * np.linalg.norm(v0) * np.stack([np.cos(th), np.sin(th)])
        ax.plot(arc[0, :-2], arc[1, :-2], color="k", lw=1.6, label="rotation" if i == 0 else None)
        ax.annotate("", xy=arc[:, -1], xytext=arc[:, -3],
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6, shrinkA=0, shrinkB=0))
        arc = start[:, None] + 0.75 * np.linalg.norm(v0) * np.stack([-np.cos(th), -np.sin(th)])
        ax.plot(arc[0, :-2], arc[1, :-2], color="k", lw=1.6)
        ax.annotate("", xy=arc[:, -1], xytext=arc[:, -3],
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6, shrinkA=0, shrinkB=0))
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    return fig

def plot_controller_outputs(runs, targets, colors):
    "runs: list of (label, output, alpha). Left: hand paths. Right: x and y over time (dashed = target)."
    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], hspace=0.12, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); ax_x = fig.add_subplot(gs[0, 1]); ax_y = fig.add_subplot(gs[1, 1], sharex=ax_x)
    tg = _np(targets)
    for label, out, alpha in runs:
        out = _np(out)
        for k in range(out.shape[1]):
            ax.plot(out[:, k, 0], out[:, k, 1], color=colors[k], alpha=alpha, lw=2.3)
            ax.scatter(out[::5, k, 0], out[::5, k, 1], color=colors[k], alpha=alpha / 2, marker="o", edgecolor="grey", s=10, zorder=2)
            ax_x.plot(out[:, k, 0], color=colors[k], alpha=alpha, lw=1.8)
            ax_y.plot(out[:, k, 1], color=colors[k], alpha=alpha, lw=1.8)
    for k in range(tg.shape[1]):
        ax_x.plot(tg[:, k, 0], color=colors[k], ls="--", lw=1, alpha=0.8)
        ax_y.plot(tg[:, k, 1], color=colors[k], ls="--", lw=1, alpha=0.8)
    ax.scatter(tg[-1, :, 0], tg[-1, :, 1], c=colors, marker="x", s=80, zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("hand paths")
    handles = [Line2D([0], [0], color="k", alpha=a, lw=2.3, label=l) for l, _, a in runs]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=1, label="target"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax_x.set_ylabel("x"); ax_y.set_ylabel("y"); ax_y.set_xlabel("time step")
    plt.setp(ax_x.get_xticklabels(), visible=False); ax_x.set_title("position over time")
    return fig


def plot_systematic_error(angles, errors):
    "errors: dict {effective gain: list of errors, one per angle}"
    fig, ax = plt.subplots(figsize=(5, 4))
    for (g, err), c in zip(errors.items(), plt.cm.inferno(np.linspace(0.1, 0.6, len(errors)))):
        ax.plot(angles, err, "o-", color=c, label=f"effective gain {g}")
    ax.set_yticks([0.0, 0.01,0.02])
    ax.set_xlabel("rotation angle (deg)"); ax.set_ylabel("error"); ax.legend()
    return fig


def plot_latent_3d(lats, tgts, order, colors, titles, ref_tgt=None):
    """lats: list of (time, tasks, 3) latent trajectories, tgts: matching joint-angle targets (time, tasks, 2).
    One panel per condition, stacked vertically, shared axes. ref_tgt: targets of the first condition, drawn in
    every panel after the first as light dashed lines, with an arrow from old to new target end point (task 0)."""
    extra = [ref_tgt] if ref_tgt is not None else []
    xs = np.concatenate([a[..., 0].ravel() for a in lats + tgts + extra])
    ys = np.concatenate([a[..., 1].ravel() for a in lats + tgts + extra])
    zs = np.concatenate([a[..., 2].ravel() for a in lats])
    z0 = zs.min()
    fig = plt.figure(figsize=(8, 7 * len(lats)))
    for i, (lat, tgt, title) in enumerate(zip(lats, tgts, titles)):
        ax = fig.add_subplot(len(lats), 1, i + 1, projection="3d")
        second = ref_tgt is not None and i > 0
        for k in order:
            if second:
                ax.plot(ref_tgt[:, k, 0], ref_tgt[:, k, 1], z0, "--", color=colors[k], lw=1.8, alpha=0.25)
            ax.plot(lat[:, k, 0], lat[:, k, 1], lat[:, k, 2], color=colors[k], lw=2.5, solid_capstyle="round")
            ax.plot(lat[:, k, 0], lat[:, k, 1], z0,           color=colors[k], lw=1.8, alpha=0.3)
            ax.plot(tgt[:, k, 0], tgt[:, k, 1], z0, "-." if second else "--",
                    color=colors[k], lw=1.8, alpha=0.6 if second else 0.5)
        if second:
            (x0, y0), (x1, y1) = ref_tgt[-1, 0, :2], tgt[-1, 0, :2]
            ax.quiver(x0, y0, z0, x1 - x0, y1 - y0, 0, color="k", lw=1.5, arrow_length_ratio=0.3)
        handles = [
            Line2D([0], [0], color="k", lw=2.5, label="controller activity (colour = reach direction)"),
            Line2D([0], [0], color="k", lw=1.8, alpha=0.3, label="its projection on the floor"),
        ]
        if second:
            handles += [Line2D([0], [0], color="k", lw=1.8, ls="--", alpha=0.25, label="initial targets"),
                        Line2D([0], [0], color="k", lw=1.8, ls="-.", alpha=0.6, label="new targets (rotated)")]
        else:
            handles.append(Line2D([0], [0], color="k", lw=1.8, ls="--", alpha=0.5, label="targets"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=9)
        ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max()); ax.set_zlim(zs.min(), zs.max())
        ax.set_xlabel("Latent 1"); ax.set_ylabel("Latent 2"); ax.set_zlabel("Latent 3")
        ax.text2D(0.5, 0.99, title, transform=ax.transAxes, ha="center", va="top", fontsize=11)
        ax.view_init(elev=20, azim=-77)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.pane.fill = False
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid'].update({'linewidth': 0})
            axis.line.set_linewidth(0.5)
            axis.pane.set_edgecolor('w')
    plt.tight_layout()
    return fig

def plot_washout(results, n_perturbation):
    "results: dict {label: {'mse': per-epoch error, 'unperturbed': error before any perturbation}}"
    styles = {"fixed controller": dict(color="gray", ls="-", lw=3), "plastic controller": dict(color="k", ls="--", lw=3.2)}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, r in results.items():
        epochs = np.arange(1, len(r["mse"]) + 1)
        ax.semilogy(epochs, r["mse"], label=label, **styles[label])
    ax.axhline(results["fixed controller"]["unperturbed"], color="gray", lw=1.2, ls=":")
    ax.axvspan(0.5, n_perturbation + 0.5, color="k", alpha=0.08, lw=0)
    ymax = ax.get_ylim()[1]
    ax.text(n_perturbation / 2, ymax, "perturbation", ha="center", va="top")
    ax.text((n_perturbation + len(r["mse"])) / 2, ymax, "washout", ha="center", va="top")
    ax.set_xlabel("epoch"); ax.set_ylabel("error (MSE)"); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.text(0.99, 0.20, "dotted line: error before\nany perturbation", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.4")
    return fig

def plot_reaches(ax, out, targets, title, colors):
    "Hand paths (solid) and targets (dashed), both coloured by reach direction."
    out, targets = _np(out), _np(targets)
    for k in range(targets.shape[1]):
        ax.plot(targets[:, k, 0], targets[:, k, 1], "--", color=colors[k], lw=2.2, alpha=0.5, zorder=1)
        ax.plot(out[:, k, 0],     out[:, k, 1],           color=colors[k], lw=1.6, zorder=2)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.axis("off")

def _mse(a, b):
    return float(((_np(a) - _np(b)) ** 2).mean())


def plot_ablations(rows, y, colors):
    "rows: {model name: {ablation name: output}}. One row per model, one column per ablation."
    n_rows, n_cols = len(rows), len(next(iter(rows.values())))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.4 * n_rows), squeeze=False)
    for r, (model_name, outs) in enumerate(rows.items()):
        for c, (name, out) in enumerate(outs.items()):
            plot_reaches(axs[r, c], out, y, f"{name}\nMSE = {_mse(out, y):.2e}", colors)
        axs[r, 0].text(-0.05, 0.5, model_name, transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=11)
    plt.tight_layout()
    return fig

def plot_adaptation_loss(hists):
    fig, axs = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.5), sharey=True)
    for ax, (name, hist) in zip(np.atleast_1d(axs), hists.items()):
        ax.semilogy(hist["mse"],  "k",       lw=1.5, label="full model")
        ax.semilogy(hist["nocb"], "#0c7b11", lw=1.5, label="adapter off")
        ax.set_title(name); ax.set_xlabel("epoch")
    np.atleast_1d(axs)[0].set_ylabel("MSE"); np.atleast_1d(axs)[0].legend()
    return fig


def plot_adaptation_reaches(before, models_outs, y, colors):
    "models_outs: dict {name: (full-model output, adapter-off output)}"
    fig, axs = plt.subplots(len(models_outs), 3, figsize=(11, 3.5 * len(models_outs)), squeeze=False)
    for r, (name, (full, nocb)) in enumerate(models_outs.items()):
        plot_reaches(axs[r, 0], before, y, f"{name}: before\nMSE = {_mse(before, y):.2e}", colors)
        plot_reaches(axs[r, 1], full,   y, f"{name}: full model, after\nMSE = {_mse(full, y):.2e}", colors)
        plot_reaches(axs[r, 2], nocb,   y, f"{name}: adapter off, after\nMSE = {_mse(nocb, y):.2e}", colors)
    plt.tight_layout()
    return fig

def plot_savings(hist, phases):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p, (angle, _) in enumerate(phases):
        ax.semilogy(hist["mse"][hist["phase"] == p], lw=1.8, label=f"phase {p}: {angle:+d} deg")
    ax.set_xlabel("epoch within phase"); ax.set_ylabel("MSE"); ax.legend()
    return fig
