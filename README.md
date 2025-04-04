# PEPT: policy-enhanced partially-tightened MPC
This repo is linked with the [paper](http://arxiv.org/abs/2504.02710) (*submitted for publication*) where we propose policy-enhanced partial tightening (PEPT) an efficient method to enhance model predictive control with a reinforcement learning policy.

Goal of this repo is to illustrate how to efficiently implement advanced MPC schemes using [`acados`](https://github.com/acados/acados) and its [novel multiphase framework](https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.3234) .
We implement
- policy-enhanced partial tightening (PEPT)
- closed-loop costing (CLC)
- partial tightening (PT)

The methods are validated using [`safe-control-gym`](https://github.com/utiasDSL/safe-control-gym) on the 3D quadcopter tracking a lemniscate.
Both CLC and PEPT make use of the trained PPO policy available in the `safe-control-gym` library.
In the comparison among the MPC-based approaches we also include the pure RL policy.

## Install on Ubuntu/macOS

### Clone repo

```bash
git clone https://github.com/aghezz1/pept
cd pept
```

### Create a `conda` environment

Create and access a Python 3.10 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n safe python=3.10
conda activate safe
```

### Install `pept` repo

#### Install `safe-control-gym`

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

#### Install `L4CasADi` (CPU only installation)

Ensure all build dependencies are installed
```
setuptools>=68.1
scikit-build>=0.17
cmake>=3.27
ninja>=1.11
```
Run
```bash
python -m pip install l4casadi --no-build-isolation
```
**NOTE**: I experienced issues with the compilation of CasADi functions on macOS as the default compiler in l4casadi is `gcc`, I solved by aliasing `gcc` to `clang`.
Another option is to open the l4casadi package installed in your environment and edit [L337](https://github.com/Tim-Salzmann/l4casadi/blob/eb6fc5c81aee29340b7e4b96e71226a88e1fa54c/l4casadi/l4casadi.py#L337) from `gcc` to `clang`.

More info at [l4casadi github](https://github.com/Tim-Salzmann/l4casadi)

#### Install `acados`

You need to separately install [`acados`](https://github.com/acados/acados) (>= v0.4.4) for fast MPC implementations.

- To build and install acados, see their [installation guide](https://docs.acados.org/installation/index.html).
- To set up the acados python interface **in the same conda environment!**, check out [these installation steps](https://docs.acados.org/python_interface/index.html).
  ```bash
  python -m pip install -e PATH_TO_ACADOS_DIR/interfaces/acados_template
  ```


## Usage
The developed code is limited to the files `**/mpc_acados_**`, `example/results_analysis.py`

The workflow to run a simulation is the following:
1) open `examples/mpc/mpc_experiment.sh`check that the field `ALGO` is set to `mpc_acados` or `mpc_acados_m`, the latter is for running the multiphase formulations
2) if you want to run multiple episodes or save data or show a gui modify the header of function `run` in `examples/mpc/mpc_experiment.py`
3) according to which `ALGO` you set, open the yaml file `mpc_acados_quadrotor_3D_tracking.yaml` (or `mpc_acados_m_quadrotor_3D_tracking.yaml`) contained in `examples/mpc/config_overrides/quadrotor_3D`, here you can change the hyperparameter of your MPC controller (horizon length, cost weights, toggle RTI, change initialization strategy, change method for the second phase, ...)
4) in your shell execute `./mpc_experiment.sh`

- If you store results and want to reproduce the figures in the paper, open `example/results_analysis.py`, edit the parameters and execute the file

- To change the scaling factor of the lemniscate (as done in the paper to make the tracking task harder), open `quadrotor_3D_tracking.yaml` in the folder `examples/mpc/config_overrides/quadrotor_3D` and modify the parameter `trajectory_scale`

## References
```
@misc{Ghezzi2025a,
      title={A Numerically Efficient Method to Enhance Model Predictive Control Performance with a Reinforcement Learning Policy},
      author={Andrea Ghezzi and Rudolf Reiter and Katrin Baumgärtner and Alberto Bemporad and Moritz Diehl},
      year={2025},
      eprint={2504.02710},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2504.02710},
}
```

## Bug reporting / contributions
They are welcome via PR

