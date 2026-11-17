#!/usr/bin/env python3
"""
run_jobs.py

Calls prepare_configs.py to generate config files, then runs the training
script for each config using a shared job queue, dispatched across CPU and
GPU worker slots.

Usage:
  python run_jobs.py <output_folder>
    --workdir <path>
    --home <path>
    [--script <path>]          default: HOME/scripts/train_model.py
    --config-dir <path>
    [--no-prep-configs]        use configs.json directly (single config)
    [--force]                  overwrite existing output folder
    [--cpu-jobs 1]             parallel CPU workers
    [--cpu-threads 1]          OMP threads per CPU worker
    [--gpu-jobs 0]             parallel GPU workers
    [--gpu-threads 1]          OMP threads per GPU worker
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
from multiprocessing import Manager, Process
from pathlib import Path
import torch

_print_lock = None


def run_one(script: str, config_file: str, job_index: int, home: Path,
            device: str = "cpu", num_threads: int = 1,
            single_job: bool = False) -> tuple:
    """Run a single training job as a subprocess with a controlled thread budget.

    :param script: Path to the training script.
    :param config_file: Path to the config JSON for this job.
    :param job_index: Index passed as --id to the training script.
    :param home: Path to the repository root.
    :param device: 'cpu' or 'cuda'.
    :param num_threads: OMP_NUM_THREADS for this job.
    :param single_job: If True, inherit terminal directly (tqdm-friendly).
    :return: Tuple of (job_index, returncode).
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)
    srcdir = str(Path(home) / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{srcdir}:{existing}" if existing else srcdir

    cmd = [sys.executable, "-u", script,
           "--config", config_file,
           "--id", str(job_index),
           "--device", device,
           "--force"]

    if single_job:
        result = subprocess.run(cmd, env=env)
        return job_index, result.returncode

    prefix = f"[Job {job_index}]"
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        with _print_lock:
            print(f"{prefix} {line}", end="", flush=True)
    proc.wait()
    return job_index, proc.returncode


def _registry_update(job_registry: Path, lock, job_id: int, **kwargs):
    """Update a single job entry in job_registry.json under the multiprocessing lock."""
    lock.acquire()
    try:
        entries = json.loads(job_registry.read_text())
        for e in entries:
            if e["job"] == job_id:
                e.update(kwargs)
                break
        job_registry.write_text(json.dumps(entries, indent=2))
    finally:
        lock.release()


def _worker(script, home, device, num_threads, config_queue, lock,
            failed_list, single_job, job_registry):
    global _print_lock
    _print_lock = lock
    while True:
        try:
            i, config_file = config_queue.get_nowait()
        except Exception:
            break

        # Register job as running
        with lock:
            entries = json.loads(job_registry.read_text())
            entries = [e for e in entries if e["job"] != i]  # drop stale on resume
            entries.append({
                "job":       i,
                "name":      None,          # filled on first _registry_mark_iteration call
                "device":    device,
                "threads":   num_threads,
                "status":    "running",
                "total":     None,          # n_values × iterations_per_value
                "completed": [],            # list of [value_index, iteration_index] pairs
            })
            job_registry.write_text(json.dumps(entries, indent=2))

        status = "failed"
        try:
            _, rc = run_one(script, config_file, i, home,
                            device, num_threads, single_job)
            if rc == 0:
                status = "completed"
            else:
                failed_list.append(i)
        except Exception as exc:
            print(f" Job {i} raised an unhandled exception: {exc}")
            failed_list.append(i)

        _registry_update(job_registry, lock, i,
                         status=status)


def main() -> None:
    """Parse arguments, generate configs, and dispatch training jobs."""
    parser = argparse.ArgumentParser(
        description="Run training jobs on a single machine with CPU and GPU worker pools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_folder", help="Output folder name, relative to --workdir")
    parser.add_argument("--workdir",         required=True, help="Base output directory")
    parser.add_argument("--home",            required=True, help="Repo root, sets config['home']")
    parser.add_argument("--script",          default=None,  help="Training script (default: HOME/scripts/train_model.py)")
    parser.add_argument("--config-dir",      required=True, help="Folder containing config files")
    parser.add_argument("--no-prep-configs", action="store_true", help="Skip prepare_configs.py, use configs.json directly")
    parser.add_argument("--force",           action="store_true", help="Overwrite existing output folder without prompting")
    parser.add_argument("--cpu-jobs",        type=int, default=1, help="Parallel CPU workers")
    parser.add_argument("--cpu-threads",     type=int, default=1, help="OMP threads per CPU worker")
    parser.add_argument("--gpu-jobs",        type=int, default=0, help="Parallel GPU workers")
    parser.add_argument("--gpu-threads",     type=int, default=1, help="OMP threads per GPU worker")
    parser.add_argument("--ignore-pretrain-paths", action="store_true",
                    help="Do not inject cx/cb/rnn pretraining paths into configs, "
                         "even if the folders exist on disk.")
    args = parser.parse_args()

    workdir    = Path(args.workdir)
    config_dir = Path(args.config_dir)
    home       = Path(args.home)

    if args.script is None:
        args.script = str(home / "scripts" / "train_model.py")
    if not config_dir.is_dir():
        sys.exit(f"Error: config folder '{config_dir}' does not exist.")

    # Output folder setup
    output_folder = workdir / args.output_folder
    job_registry = output_folder / "job_registry.json"
    job_registry.write_text("[]")

    # Config preparation
    cx_pre  = workdir / "pretrainings" / "cortex"
    cb_pre  = workdir / "pretrainings" / "cerebellum"
    rnn_pre = workdir / "pretrainings" / "RNN"
    if not args.ignore_pretrain_paths:
        pre_args  = ["--cx-pre",  str(cx_pre)]  if cx_pre.is_dir()  else []
        pre_args += ["--cb-pre",  str(cb_pre)]  if cb_pre.is_dir()  else []
        pre_args += ["--rnn-pre", str(rnn_pre)] if rnn_pre.is_dir() else []
    else:
        pre_args = []

    if not args.no_prep_configs:
        print("Running prepare_configs.py...")
        prep_config_script = home / "scripts" / "prepare_configs.py"
        result = subprocess.run([
            sys.executable, prep_config_script,
            str(config_dir / "configs_list.json"),
            str(config_dir / "template_config.json"),
            str(output_folder),
            str(home),
            *pre_args,
        ])
        if result.returncode != 0:
            sys.exit("Error: prepare_configs.py failed.")
        print("Configuration files prepared successfully.")
        config_files = sorted(
            (output_folder / "configs").glob("config_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
    else:
        src_config = config_dir / "configs.json"
        dst_config = output_folder / "config.json"
        if not src_config.exists():
            sys.exit(f"Error: expected single config at '{src_config}'.")
        with open(src_config, "r") as f:
            config = json.load(f)
        config["home"]     = str(home)
        config["work_dir"] = str(output_folder)
        with open(dst_config, "w") as f:
            json.dump(config, f, indent=4)
        config_files = [dst_config]

    if not config_files:
        sys.exit("Error: no config files found.")
    n_total = len(config_files)
    print(f"Found {n_total} config(s).")

    # Snapshot
    snapshot_dir = output_folder / "snapshot"
    snapshot_dir.mkdir()
    args_header = (
        "# === Run arguments ===\n"
        + "\n".join(f"# {k}: {v}" for k, v in vars(args).items())
        + "\n# =====================\n\n"
    )
    with open(Path(__file__).resolve()) as f:
        original_source = f.read()
    with open(snapshot_dir / "run_jobs.py", "w") as f:
        f.write(args_header + original_source)
    if args.no_prep_configs:
        shutil.copy2(config_dir / "configs.json", snapshot_dir / "configs.json")
    else:
        shutil.copy2(config_dir / "configs_list.json",    snapshot_dir / "configs_list.json")
        shutil.copy2(config_dir / "template_config.json", snapshot_dir / "template_config.json")
    shutil.copy2(args.script, snapshot_dir / Path(args.script).name)
    
    # Check for CUDA availability if GPU jobs are requested
    if args.gpu_jobs > 0 and not torch.cuda.is_available():
        print(f"Warning: --gpu-jobs > 0 but CUDA is not available. Falling back to CPU-only, using preselected cpus n_jobs {args.cpu_jobs}.")
        args.gpu_jobs = 0
    
    # Dispatch
    single_job = (n_total == 1)
    print(f"Dispatching {n_total} job(s): "
          f"{args.gpu_jobs} GPU worker(s) ({args.gpu_threads} thread(s) each), "
          f"{args.cpu_jobs} CPU worker(s) ({args.cpu_threads} thread(s) each)...")

    manager      = Manager()
    lock         = manager.Lock()
    failed       = manager.list()
    config_queue = manager.Queue()
    for i, cfg in enumerate(config_files):
        config_queue.put((i, str(cfg)))

    processes = []
    for _ in range(args.gpu_jobs):
        processes.append(Process(target=_worker,
                                 args=(args.script, home, "cuda", args.gpu_threads,
                                       config_queue, lock, failed, single_job, job_registry)))
    for _ in range(args.cpu_jobs):
        processes.append(Process(target=_worker,
                                 args=(args.script, home, "cpu", args.cpu_threads,
                                       config_queue, lock, failed, single_job, job_registry)))

    for p in processes: p.start()
    for p in processes: p.join()

    failed = sorted(failed)
    print(f"{'='*40}")
    print(f"Done: {n_total - len(failed)}/{n_total} jobs succeeded.")
    if failed:
        print(f"Failed job indices: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
