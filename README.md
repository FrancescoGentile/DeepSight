<div align="center">

# DeepSight

<h4>A Deep Learning framework for Computer Vision</h4>

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>


## Setup

First of all, clone the repository and move into the project folder:

```bash
git clone https://github.com/FrancescoGentile/deepsight.git
cd deepsight
```

Then, use any virtual environment manager to create a new environment with python 3.12. For example, using [pyenv](https://github.com/pyenv/pyenv):

```bash
pyenv install 3.12
pyenv virtualenv 3.12 deepsight
pyenv local deepsight
```

Or if you prefer using conda/mamba:

```bash
mamba create -n deepsight python=3.12
mamba activate deepsight
```

Before installing the dependencies, make sure that pdm will use the virtual environment you just created instead of creating a new one (this should not happen if you used conda/mamba but it may happen if you used pyenv virtualenv). This is necessary because some dependencies will not be installed using pdm but using pip, and the virtual environment created by pdm does not provide pip.

```bash
pdm use

# or directly specify the path to the virtual environment
pdm use /path/to/virtual/environment
```

Finally, install the dependencies:

```bash
# to install only production dependencies
pdm sync --prod

# to install all dependencies
pdm sync
```

Additionally install pytorch and torchmetrics:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install torchmetrics
```