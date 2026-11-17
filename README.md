# Cerebello-Cortical Distributed Learning framework (ccdl)


This repository contains the code associated with the paper titled "[Distributed learning across fast and slow systems enables efficient adaptation of learned representations](https://doi.org/10.1101/2025.06.01.657238)". The code implements training the neural network model and the analyses described in the paper. 

## Requirements

Two setups are available (`setup.sh` asks which one to install):

- **legacy (paper)**: Python `3.8.10` + the exact versions that produced the paper's results (`requirements.txt`). **Use this to reproduce the results.**
- **updated**: Python `3.12` + recent versions (`updated-requirements.txt`). **Not fully tested.**

Legacy setup:

- **Python `3.8.10`**
- **Main dependencies (full list in requirements.txt)**
  | | | | |
  |---|---|---|---|
  | numpy 1.22.3 | scipy 1.10.0 | pandas 1.5.3 | matplotlib 3.5.2 |
  | scikit-learn 1.2.0 | openpyxl 3.1.2 | torch 1.11.0+cu113 | tensorboard 2.9.0 |

- **OS:** Linux (tested); macOS with the CPU torch build; Windows via conda.
- **Hardware:** a CUDA 11.3 GPU to reproduce GPU training; none needed for the
  demo or the CPU torch build.


## Installation

1. Clone this repository to your local machine:
```
git clone https://github.com/caycogajiclab/distributed-learning
cd distributed-learning
```
2. Install Python (skip this step if you use conda, which installs Python for you).

   With venv, you need Python 3.8 for the `legacy` setup (reproduces the paper's results) or Python 3.12 for the `updated` setup (runs, but not yet fully validated); see [Requirements](#requirements). On Debian/Ubuntu:
```bash
sudo bash setup_python.sh      # asks for 3.8 or 3.12
```
   On macOS and Windows, use conda.

3. Create the environment, install the deps, editable-install the repo. The script asks for the setup (legacy/updated), the manager (conda/venv), the env folder/name, CPU-only torch, and whether to unpack `data_archive.zip` (only needed for the paper's figures):
```bash
./setup.sh
```
On Windows, run the conda steps from an Anaconda Prompt instead (`conda env create -f environment.yml`, or `environment-updated.yml`; then `pip install -e . --no-deps`) and unzip `data_archive.zip` into the repo root by hand.

4. Activate with `conda activate ccdl` (or
`source .venv/bin/activate`; use the name/folder you chose if not the default).

Notes: 
- for **CPU-only** torch, answer yes when `setup.sh` asks (Linux).
- on recent Linux (glibc >= 2.41), the legacy torch 1.11 fails to load (`cannot enable executable stack`); `setup.sh` fixes this automatically with `patchelf --clear-execstack`.
- Typical install time: ~5–15 min on a normal desktop, dominated by the torch download.
- Run `./setup.sh --help` for all options.

## Usage

### Repository layout

| Path             | Contents |
|------------------|----------|
| `src/ccdl/`      | The `ccdl` library (models, training, task, helpers). |
| `scripts/`       | Simulation + config scripts and the figure driver. |
| `scripts/configs/` | Per-experiment configuration folders. Descriptions of the different `.json` can be found in `_templates/`|
| `scripts/figures/` | One script per paper figure (reloads source data, draws it). |
| `demo/`     | Notebook showing basic usage of the network. It uses `ccdl/pretrained` to avoid the heavy pretraining runs. |
| `tests/`    | Smoke tests: each core module's `__main__` self-test is run. |
| `data_archive.zip` | Contents of `data/`, `figures/` and `source_data/`, unpacked by `./setup.sh --extract-data`. |
| `data/` · `figures/` · `source_data/` | Working folders (git-ignored). `source_data/` holds the data behind each figure. |


### Configurations & simulations. 
Each experiment has a folder under `scripts/configs/<experiment>/` (a `template_config.json` plus a
`configs_list.json` enumerating the parameter sweep). The `scripts/train_*.py` 
scripts run one experiment/simulation type; `scripts/generate_data.py` chains the full
pipeline of simulations described in the paper.
`scripts/configs/_templates/` contains annotated reference templates that document every config key, including the optional ones. They are for reading only; the configs actually used by each experiment are the ones in its own folder.


### Figures generation

`source_data/` (unpack it first with `./setup.sh --extract-data`) is the demo dataset: per figure panel, a `<name>_spec.pkl` (plot specification) and a `<name>_sourcedata.csv.gz` (the plotted values; read with `pandas.read_csv`).
Each figure script reloads its panel from it and renders. The pretrained `data/pretrainings` are not needed.

```bash
python scripts/figures/DL-backprop.py
```

**Expected Output:** PDFs/SVGs in `figures/pdf/` and `figures/svg/` (here `4b`, `4c`,
`4d`); the pre-rendered figures already in those folders are the reference.


## Tests

Each core module (`utils`, `ccnet`, `layers`, `trainer`) has a `__main__` self-test. Run them all with:
```bash
python tests/test_main_blocks.py -v     # or: pytest tests/ -v
```

## License
Code released under the BSD-3 license; see `LICENSE`.