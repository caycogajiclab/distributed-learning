#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
generate_data.py

Full pipeline:
  1. Cortex pretraining    → WORKDIR/cx_pre      (via run_jobs)
  2. Cereb  pretraining    → WORKDIR/cb_pre       (direct call, single config)
  3. Collect pretrainings  → WORKDIR/pretrainings/{cortex,cerebellum}
  4. Normal training       → WORKDIR/normal_training (via run_jobs)
"""

import shutil
import subprocess
import sys
import json
from pathlib import Path

from ccdl.paths import PROJECT_ROOT, DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
HOME    = PROJECT_ROOT
WORKDIR = DATA_DIR
SCRIPTS = HOME / "scripts"
CONFIGS = SCRIPTS / "configs"
RUN_JOBS = SCRIPTS / "run_jobs.py"

# ── Parallelism settings ───────────────────────────────────────────────────
# CPU_JOBS    = 2
# CPU_THREADS = 7
# GPU_JOBS    = 1
# GPU_THREADS = 2 # CPU threads allocated to each GPU job# CPU_JOBS    = 2

CPU_JOBS    = 2
CPU_THREADS = 7
GPU_JOBS    = 0
GPU_THREADS = 2 # CPU threads allocated to each GPU job# CPU_JOBS    = 2

# CPU_JOBS    = 1
# CPU_THREADS = 12
# GPU_JOBS    = 0
# GPU_THREADS = 2 # CPU threads allocated to each GPU job

CB_CPU_JOBS    = 0
CB_CPU_THREADS = 0
CB_GPU_JOBS    = 1
CB_GPU_THREADS = 6 # CPU threads allocated to each GPU job


# ── Helpers ────────────────────────────────────────────────────────────────
def banner(step: str) -> None:
    print(f"\n{'='*60}\n  {step}\n{'='*60}", flush=True)

def run_step(cmd: list, step: str) -> None:
    banner(step)
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        sys.exit(f"Error: '{step}' failed (exit {result.returncode}).")

def run_jobs(output_folder: str, config_dir: Path, script: Path = None, cb_pre: bool = False, 
             force: bool = False, text: str = None, ignore_pretrain_paths: bool = False) -> None:
    cpu_jobs    = CB_CPU_JOBS    if cb_pre else CPU_JOBS
    cpu_threads = CB_CPU_THREADS if cb_pre else CPU_THREADS
    gpu_jobs    = CB_GPU_JOBS    if cb_pre else GPU_JOBS
    gpu_threads = CB_GPU_THREADS if cb_pre else GPU_THREADS
    
    # Save specs once, here
    outdir = WORKDIR / output_folder
    if outdir.exists():
        if force:
            shutil.rmtree(outdir)
        else:
            answer = input(f"Output folder already exists ({outdir}). Delete and recreate? (y/n) ")
            if answer.strip().lower() == "y":
                shutil.rmtree(outdir)
            else:
                sys.exit("Interrupted. Please specify an available path.")
    outdir.mkdir(parents=True)
    save_specs(outdir, cb_pre=cb_pre)
    
    cmd = [
        RUN_JOBS, output_folder,
        "--workdir",     WORKDIR,
        "--home",        HOME,
        "--config-dir",  config_dir,
        "--cpu-jobs",    cpu_jobs,
        "--cpu-threads", cpu_threads,
        "--gpu-jobs",    gpu_jobs,
        "--gpu-threads", gpu_threads,
    ]
    if script is not None:
        cmd += ["--script", script]
    if force:
        cmd += ["--force"]
    if cb_pre:
        cmd += ["--no-prep-configs"]
    if ignore_pretrain_paths:
        cmd += ["--ignore-pretrain-paths"]
    text = output_folder if text is None else text
    run_step(cmd, f"run_jobs → {output_folder}")

def collect_pretrainings(src: Path, dst: Path) -> None:
    banner(f"Collecting pretrainings: {src.name} → {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    params_path = src / "params.json"
    params = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        print(f"  Warning: no params.json found in {src}, train_info.json will only contain source path.")
    params["pretrainings_data_source"] = str(src)
    with open(dst / "train_info.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved train_info.json -> {dst / 'train_info.json'}")

    # Detect structure from first matching item in src
    numbered_dirs  = sorted([d for d in src.iterdir() if d.is_dir()  and d.name.isdigit()],  key=lambda d: int(d.name))
    numbered_files = sorted([f for f in src.iterdir() if f.is_file() and f.stem.isdigit()],  key=lambda f: int(f.stem))

    if numbered_dirs:
        items = numbered_dirs
        def copy_one(item: Path) -> Path:
            pkl = item / "cortex_state_dict.pkl"
            if not pkl.exists():
                return None
            shutil.copy2(pkl, dst / item.name)
            return dst / item.name
    elif numbered_files:
        items = numbered_files
        def copy_one(item: Path) -> Path:
            shutil.copy2(item, dst / item.name)
            return dst / item.name
    else:
        sys.exit(f"Error: no numbered subdirs or files found in {src}.")

    copied = 0
    for item in items:
        result = copy_one(item)
        if result:
            print(f"  Copied {item.name} -> {result}")
            copied += 1
        else:
            print(f"  Warning: expected file not found in {item}, skipping.")

    if copied != len(items):
        sys.exit(f"Error: expected {len(items)} files, copied {copied}.")

def save_specs(outdir: str, cb_pre: bool = False) -> None:
    cpu_jobs   = CB_CPU_JOBS   if cb_pre else CPU_JOBS
    cpu_threads = CB_CPU_THREADS if cb_pre else CPU_THREADS
    gpu_jobs   = CB_GPU_JOBS   if cb_pre else GPU_JOBS
    gpu_threads = CB_GPU_THREADS if cb_pre else GPU_THREADS

    specs = {
        "cpu_jobs": cpu_jobs,
        "cpu_threads": cpu_threads,
        "gpu_jobs": gpu_jobs,
        "gpu_threads": gpu_threads,
    }
    
    (outdir / "hw_specs.json").write_text(json.dumps(specs, indent=2))

#########################################
# template code to resume jobs
# 
# run_step([
#     SCRIPTS / "resume_jobs.py", "DL-BP",
#     "--workdir", WORKDIR,
#     "--home", HOME,
# ], "resume DL-BP")



# ── Cortex Pretraining ────────────────────────────────────────────────────────
subprocess.run([sys.executable, str(SCRIPTS/"pretrain_cortex.py"),
    "--home",        str(HOME),
    "--workdir",     str(WORKDIR),
    "--output",      "cx_pre",
    "--config-dir",  str(CONFIGS/"cx_pretrain"),
    # "--force", # Use to overwrite existing data.
])

# ── Cerebellum Pretraining ──────────────────────────────────────────────────
run_jobs(
    output_folder = "cereb_pre",
    config_dir    = CONFIGS / "cb_pretrain",
    script        = SCRIPTS / "pretrain_cereb.py",
    force         = True,
    cb_pre        = True,
    text          = "Cerebellar pretraining",
)

# # # ── Collect pretrainings ────────────────────────────────────────────────────
collect_pretrainings(
    src=WORKDIR / "cx_pre/2.fb.ae_fixed", 
    dst=WORKDIR / "pretrainings" / "cortex"
)
collect_pretrainings(
    src=WORKDIR / "cereb_pre" / "2x1024_noseqlr_1.0e-04_lr_scheduler",
    dst=WORKDIR / "pretrainings" / "cerebellum"
)
collect_pretrainings(
    src=WORKDIR / "cx_pre" / "3.RNN", 
    dst=WORKDIR / "pretrainings" / "RNN"
)

# # # # ── Pretrain washout ────────────────────────────────────────────────────
subprocess.run([sys.executable, str(SCRIPTS/"pre_washout.py"),
    "--home",        str(HOME),
    "--workdir",     str(WORKDIR),
    "--pretrainings", WORKDIR / "pretrainings",
    "--output",      "pre.washout",
    "--force", # Use to overwrite existing data.
])

# # # ── Fixed Cortex ────────────────────────────────────────────────────────────

run_jobs(
    output_folder = "fixed_cortex",
    config_dir    = CONFIGS / "fixed_cortex",
    script        = SCRIPTS / "train_model.py",
    force         = True,
    text          = "Fixed cortex trainings",
)

run_jobs(
    output_folder = "fixed_cortex_outputs",
    config_dir    = CONFIGS / "fixed_cortex_outputs",
    script        = SCRIPTS / "train_model.py",
    force         = True,
    text          = "Fixed cortex trainings",
)

# # # ── DL - backprop ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "DL-BP",
    config_dir    = CONFIGS / "DL-BP",
    script        = SCRIPTS / "train_model.py",
    force         = True,
    text          = "DL-BP trainings",
)

# # # ── DL - local ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "DL-local",
    config_dir    = CONFIGS / "DL-local",
    script        = SCRIPTS / "train_model.py",
    force         = True,
    text          = "DL-local trainings",
)


# # ── damage ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "damage",
    config_dir    = CONFIGS / "damage",
    script        = SCRIPTS / "train_damage.py",
    force         = True,
    text          = "damage trainings",
)


# # # ── savings ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "savings",
    config_dir    = CONFIGS / "savings",
    script        = SCRIPTS / "train_continual.py",
    force         = True,
    text          = "savings trainings (continual task)",
)


# # ── continual test ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "continual",
    config_dir    = CONFIGS / "continual",
    script        = SCRIPTS / "train_continual.py",
    force         = True,
    text          = "continual trainings",
)

# # ── washout ────────────────────────────────────────────────────────
run_jobs(
    output_folder = "washout",
    config_dir    = CONFIGS / "washout",
    script        = SCRIPTS / "train_washout.py",
    force         = True,
    text          = "washout trainings",
)

# # ── washout extra (alternative learning rates) ──────────────────────
run_jobs(
    output_folder = "washout-extra",
    config_dir    = CONFIGS / "washout-extra",
    script        = SCRIPTS / "train_washout.py",
    force         = True,
    text          = "washout extra trainings",
)