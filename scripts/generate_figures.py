#!/usr/bin/env python
"""Generate all paper figures in one go.

Runs every figure script in ``scripts/figures/`` (all *.py except
``plot_utils.py``), one isolated subprocess each, and prints a compact,
one-line-per-script summary of the figures produced.

The homogeneous plotting settings that used to be duplicated at the top of every
figure script now live here (single source of truth) and are passed to each
script's argument parser. Per-script data-folder settings still live in the
scripts themselves.
"""
import os
import subprocess
import sys

from ccdl.paths import SOURCE_DATA_DIR, FIGURES_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR_SCRIPTS = os.path.join(HERE, 'figures')

############################################
# SETTINGS (single source of truth)
############################################

UPDATE_METRICS = False
REBUILD_SOURCE_DATA = False

plots_data_folder = str(SOURCE_DATA_DIR)
plots_folder = str(FIGURES_DIR)
save_pdf = True
save_tiff = False

# Helper module, not a figure script.
EXCLUDE = {'plot_utils.py'}
# Each figure script prints this prefix once per saved figure (see the scripts).
FIG_MARKER = '[figure] '


def main():
    scripts = sorted(f for f in os.listdir(FIGURES_DIR_SCRIPTS)
                     if f.endswith('.py') and f not in EXCLUDE)

    common_args = [
        '--plots-data-folder', plots_data_folder,
        '--plots-folder', plots_folder,
        '--update-metrics', str(UPDATE_METRICS),
        '--rebuild-source-data', str(REBUILD_SOURCE_DATA),
        '--save-pdf', str(save_pdf),
        '--save-tiff', str(save_tiff),
    ]

    # Agg backend + silenced warnings keep the console clean; both affect only
    # display/warnings, never the figure content.
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONWARNINGS': 'ignore'}

    print(f"Generating figures from {len(scripts)} scripts -> {plots_folder}\n")

    total, failed = 0, []
    for i, script in enumerate(scripts, 1):
        prefix = f"[{i}/{len(scripts)}] {script:<18} |"
        print(f"{prefix} Executing...", flush=True)

        # Capture stdout to filter the scripts' noisy prints; keep only the
        # per-figure markers. stderr is inherited so real errors still surface.
        proc = subprocess.Popen(
            [sys.executable, script, *common_args],
            cwd=FIGURES_DIR_SCRIPTS, env=env,
            stdout=subprocess.PIPE, text=True,
        )
        names = []
        for line in proc.stdout:
            if line.startswith(FIG_MARKER):
                names.append(line[len(FIG_MARKER):].strip())
        proc.wait()
        total += len(names)

        status = "Completed" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        summary = f"{prefix} {status:<9} {len(names):>2} figures: {', '.join(names)}"
        if proc.returncode != 0:
            failed.append(script)
        print(summary, flush=True)

    print(f"\nDone: {total} figures in {plots_folder}")
    print(f"      source data in {plots_data_folder}")
    if failed:
        print(f"      FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
