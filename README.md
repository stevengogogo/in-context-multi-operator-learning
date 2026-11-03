# In-Context Multi-Operator Learning with DeepOSets

Official code for the NeurIPS 2026 paper **In-Context Multi-Operator Learning with DeepOSets**.

DeepOSets combines a DeepSets encoder with a DeepONet decoder, so a single network learns many operators from a prompt of input-output examples, with no weight updates at inference time.

## Installation

```bash
git clone https://github.com/stevengogogo/in-context-multi-operator-learning.git
cd in-context-multi-operator-learning
uv sync                     # add --extra icon for the PyTorch ICON comparison
bash scripts/run_smoke.sh   # quick end-to-end check
```

Run scripts with `uv run`, e.g. `uv run python scripts/train_1d.py --help`.

## Repository structure

```
deeposets/                    Models, PDE solvers, data samplers, training, evaluation
scripts/                      Generate data, train, evaluate, plot
experiments/icon_comparison/  DeepOSets vs ICON and nearest-neighbor baseline (PyTorch)
third_party/icon-tutorial/    Vendored ICON reference code
docs/                         Experiment guide, code guide, reviewer guide
tests/                        Smoke tests
```

## Experiments

Each experiment follows the same three steps: generate data, train, evaluate. Commands are run from the repository root, prefixed with `uv run`.

| Paper | Experiment | Entry points |
|-------|------------|--------------|
| Sec. 6.1 | Burgers | `generate_1d_burgers.py`, `train_1d.py --problem burgers` |
| Sec. 6.2 | 1D Darcy | `generate_1d_poisson.py`, `train_1d.py --problem poisson` |
| Sec. 6.3 | 2D Darcy | `generate_2d_poisson.py`, `train_2d.py` |
| Sec. 6.6, App. B.5, D | DeepOSets vs ICON, nearest-neighbor baseline | `experiments/icon_comparison/` |

Example (1D Darcy):

```bash
python scripts/generate_1d_poisson.py
python scripts/train_1d.py --problem poisson
python scripts/evaluate_1d.py --problem poisson
```

Other docs: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (data generation and per-experiment details), [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) (usage and custom data) and [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md) (equations, data shapes, code map).

## Reproducibility

To regenerate the data and rerun the Burgers, 1D Darcy, 2D Darcy and ICON-comparison experiments:

```bash
uv sync --extra icon --extra cuda
bash scripts/run_all.sh
```

Data is written to `data/` and results to `experiments/output/` (both git-ignored). Use `STAGES="burgers darcy1d darcy2d icon"` to run a subset.

* **Settings.** The defaults are the settings used for the paper: dataset sizes as in the paper, 40 pairs per operator, 100k steps, Adam with lr 1e-4 and cosine decay, and seeds fixed in the scripts.
* **Hardware.** The Burgers and Darcy runs used one NVIDIA RTX 4060 Ti. Results vary slightly with hardware, library versions and seed; training times depend on the GPU.
* **Full scale.** The runs take several GPU-hours and tens of GB of disk. The ICON comparison alone takes about 9 hours.
* **Not included.** The 12-task experiment (Sec. 6.4), the extrapolation experiment (Sec. 6.5) and the prompt ablations (App. C).

## Authors

- Shao-Ting Chiu\* (Electrical and Computer Engineering, Texas A&M University)
- Aditya Nambiar\* (Mathematics, Texas A&M University)
- Jonathan W. Siegel (Mathematics, Texas A&M University)
- Ali Syed (Mathematics, Texas A&M University)
- Ulisses Braga-Neto (Electrical and Computer Engineering, Texas A&M University)

\* Equal contribution.

## Citation

This work is accepted to the NeurIPS 2026 main track ([arXiv:2512.16074](https://arxiv.org/abs/2512.16074)).

```bibtex
@article{chiu2025deeposets,
  title         = {In-Context Multi-Operator Learning with {DeepOSets}},
  author        = {Chiu, Shao-Ting and Nambiar, Aditya and Siegel, Jonathan W. and Syed, Ali and Braga-Neto, Ulisses},
  journal       = {arXiv preprint arXiv:2512.16074},
  year          = {2025},
  eprint        = {2512.16074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

[MIT](LICENSE). Third-party code in `third_party/icon-tutorial/` keeps its original license.
