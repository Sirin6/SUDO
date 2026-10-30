# SUDO

This folder contains code for **Simulation-free Unbalanced
Dynamic Optimal Transport with General Growth Penalty (SUDO)**.

SUDO learns continuous transport and population change from unpaired temporal
snapshots by solving dynamic unbalanced optimal transport with general convex growth penalty. For a chosen convex growth penalty, it learns the travelling-Dirac
path and its dynamic cost, solves the induced semi-couplings, and trains the
velocity, score, and growth fields with unbalanced flow matching.

## Contents

- `utils.py`: transport plans, bridge models, simulation, metrics, and plotting. The experiment and the utils.py are both adopted from USB [https://github.com/Sirin6/USB].
- `sudo_*.py`: main SUDO experiments on simulation, Dyngen, Gaussian-1000D,
  EMT, and Mouse-50D data.
- `SUDO_minibatch_eb100.py`: EB-100D mini-batch OT ablation.
- `simulation_penalty_*.ipynb`: only-growth, only-death, and no-preference
  penalties on the simulation data.
- `train_mouse_growth_*.ipynb`: Mouse WFR and only-growth model training.
- `render_*.ipynb`: Gaussian, Mouse, and travelling-Dirac figure generation.
- `data/`: CSV snapshots used by the experiments.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate SUDO
```

Alternatively, install the dependencies with:

```bash
pip install -r requirements.txt
```