#!/usr/bin/env python3
"""Re-run incomplete jobs from an existing output folder,
reusing original device/thread settings from job_registry.json.
Incomplete means: total_iterations is None, or len(completed_iterations) < total_iterations."""

import argparse
import json
import sys
from datetime import datetime
from multiprocessing import Manager, Process
from pathlib import Path

from run_jobs import run_one, _registry_update
import run_jobs

_print_lock = None  

def _worker(script, home, device, num_threads, config_queue, lock,
            failed_list, single_job, job_registry, _preserved_entries=None):
    global _print_lock
    _print_lock = lock
    run_jobs._print_lock = lock
    while True:
        try:
            i, config_file = config_queue.get_nowait()
        except Exception:
            break

        # Restore preserved fields instead of zeroing them out
        preserved = {}
        if _preserved_entries:
            for e in _preserved_entries:
                if e["job"] == i:
                    preserved = e
                    break

        lock.acquire()
        try:
            entries = json.loads(job_registry.read_text())
            entries = [e for e in entries if e["job"] != i]
            entries.append({
                "job": i,
                "name": preserved.get("name"),
                "device": device,
                "threads": num_threads,
                "status": "running",
                "total": preserved.get("total"),
                "completed": preserved.get("completed", []),
            })
            job_registry.write_text(json.dumps(entries, indent=2))
        finally:
            lock.release()

        status = "failed"
        try:
            _, rc = run_one(script, config_file, i, home,
                            device, num_threads, single_job)
            if rc == 0:
                status = "completed"
            else:
                failed_list.append(i)
        except Exception as exc:
            print(f"  Job {i} raised an unhandled exception: {exc}")
            failed_list.append(i)

        _registry_update(job_registry, lock, i, status=status)

def infer_device_and_threads(gpu_jobs, gpu_threads, cpu_threads):
    """
    Fallback rule for jobs with no device/threads in job_registry.
    Adjust if you want a more complex CPU/GPU scheduling pattern.
    Here: if any GPU jobs were configured, we send missing jobs to GPU,
    otherwise we send them to CPU.
    """
    if gpu_jobs and gpu_jobs > 0:
        return "cuda", gpu_threads
    else:
        return "cpu", cpu_threads

parser = argparse.ArgumentParser()
parser.add_argument("output_folder", help="Existing output folder (relative to --workdir)")
parser.add_argument("--workdir", required=True)
parser.add_argument("--home",    required=True)
parser.add_argument("--only", type=int, nargs="*",
                    help="Force specific job indices (overrides auto-detection)")

args = parser.parse_args()

output_folder = Path(args.workdir) / args.output_folder
home          = Path(args.home)
job_registry  = output_folder / "job_registry.json"

if not job_registry.exists():
    sys.exit(f"Error: job_registry.json not found in '{output_folder}'.")

all_entries = json.loads(job_registry.read_text())
entries_by_job = {e["job"]: e for e in all_entries}

configs_dir = output_folder / "configs"
config_jobs = set()

if configs_dir.exists():
    for cfg in configs_dir.glob("config_*.json"):
        try:
            job_idx = int(cfg.stem.split("_")[-1])
            config_jobs.add(job_idx)
        except ValueError:
            pass
elif (output_folder / "config.json").exists():
    # single-config fallback
    config_jobs.add(0)

specs_path = output_folder / "hw_specs.json"
if specs_path.exists():
    specs = json.loads(specs_path.read_text())
    cpu_jobs = specs.get("cpu_jobs")
    cpu_threads = specs.get("cpu_threads")
    gpu_jobs = specs.get("gpu_jobs")
    gpu_threads = specs.get("gpu_threads")
else:
    cpu_jobs = cpu_threads = gpu_jobs = gpu_threads = None

# merge missing jobs into registry view
for job_idx in sorted(config_jobs):
    if job_idx not in entries_by_job:
        device, threads = infer_device_and_threads(gpu_jobs, gpu_threads, cpu_threads)
        entries_by_job[job_idx] = {
            "job": job_idx,
            "name": None,
            "device": device,
            "threads": threads,
            "status": "not_started",
            "total": None,
            "completed": [],
        }

all_entries = [entries_by_job[i] for i in sorted(entries_by_job)]
job_registry.write_text(json.dumps(all_entries, indent=2))

# Auto-detect incomplete jobs
def is_incomplete(e: dict) -> bool:
    if e.get("status") == "completed":
        return False
    total = e.get("total")
    completed = e.get("completed", [])
    if total is None:
        return True
    return len(completed) < total

if args.only:
    entries = [e for e in all_entries if e["job"] in args.only]
else:
    entries = [e for e in all_entries if is_incomplete(e)]

if not entries:
    print("All jobs already completed. Nothing to resume.")
    sys.exit(0)
    
print(f"Resuming {len(entries)}/{len(all_entries)} incomplete job(s):")
for e in entries:
    total     = e.get("total", "?")
    completed = e.get("completed", [])
    name      = e.get("name") or "(unknown)"
    print(f"  Job {e['job']:>3}  [{name}]  {len(completed)}/{total} (value, iter) pairs done  "
          f"device={e['device']}")

# Resolve config per job
def get_config(job_index: int) -> str:
    cfg = output_folder / "configs" / f"config_{job_index}.json"
    if not cfg.exists():
        cfg = output_folder / "config.json"
    if not cfg.exists():
        sys.exit(f"Error: could not find config file for job {job_index}.")
    return str(cfg)

# Append to resume log
resume_log = output_folder / "snapshot" / "resume_log.txt"
resume_log.parent.mkdir(exist_ok=True)
with open(resume_log, "a") as f:
    f.write(
        f"\n# === Resume {datetime.now().isoformat(timespec='seconds')} ===\n"
        + "\n".join(f"# {k}: {v}" for k, v in vars(args).items())
        + "\n# jobs: " + str([e["job"] for e in entries])
        + "\n# ================================================\n"
    )

# Recover script path from snapshot
snapshot_run_jobs = output_folder / "snapshot" / "run_jobs.py"
script = None
if snapshot_run_jobs.exists():
    for line in snapshot_run_jobs.read_text().splitlines():
        if line.startswith("# script:"):
            script = line.split("# script:")[-1].strip()
            break
if not script or script == "None":
    script = str(home / "scripts" / "train_model.py")
    print(f"Warning: could not read script path from snapshot, falling back to: {script}")
if not Path(script).exists():
    sys.exit(f"Error: script not found at '{script}'.")

# Dispatch — use hw_specs pool sizes, not one process per job
manager = Manager()
lock = manager.Lock()
failed = manager.list()
config_queue = manager.Queue()
for e in entries:
    config_queue.put((e["job"], get_config(e["job"])))

single_job = (len(entries) == 1)
preserved_entries = list(entries)

# Determine pool sizes from hw_specs (fall back to 1 CPU if unset)
n_gpu_workers = gpu_jobs if gpu_jobs else 0
n_cpu_workers = cpu_jobs if cpu_jobs else 1

# Check CUDA availability
import torch
if n_gpu_workers > 0 and not torch.cuda.is_available():
    print("Warning: gpu_jobs > 0 but CUDA is not available. Falling back to CPU-only.")
    n_cpu_workers = max(n_cpu_workers, n_gpu_workers)
    n_gpu_workers = 0

print(
    f"Dispatching with {n_gpu_workers} GPU worker(s) "
    f"({gpu_threads} thread(s) each) and {n_cpu_workers} CPU worker(s) "
    f"({cpu_threads} thread(s) each)..."
)

processes = []
for _ in range(n_gpu_workers):
    processes.append(Process(
        target=_worker,
        args=(script, home, "cuda", gpu_threads,
              config_queue, lock, failed, single_job, job_registry, preserved_entries)
    ))
for _ in range(n_cpu_workers):
    processes.append(Process(
        target=_worker,
        args=(script, home, "cpu", cpu_threads,
              config_queue, lock, failed, single_job, job_registry, preserved_entries)
    ))

for p in processes: p.start()
for p in processes: p.join()

failed = sorted(failed)
print(f"\nDone: {len(entries) - len(failed)}/{len(entries)} jobs succeeded.")
if failed:
    print(f"Still incomplete: {failed}")
    sys.exit(1)
