# LEGOLAS 🧝‍♀️🏹
A fast and accurate machine learning model built on PyTorch for predicting protein chemical shifts from PDB structures and molecular dynamics trajectories. This model is designed to be run on either CPUs or GPUs.

![Model Architecture](images/model_architecture.png "Model Architecture")

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation
1. Clone the repository:
```bash
 git clone https://github.com/mdarrows/legolas.git
 cd legolas
```
2. Set up a conda environment:
Use the provided legolas_env.yaml file to create the conda environment:
```bash
conda env create -n legolas -f legolas_env.yaml
conda activate legolas
```
Versions when installing using conda:

- python 3.10
- cuda 11.8
- pytorch 2.5.1

3. Install TorchANI:
Procedure from: [TorchANI Installation](external/torchani_sandbox/README.md)

**Perform the following commands within `external/torchani_sandbox`:**

LEGOLAS is most efficient when run using the torchani compiled cuAEV extension, but it is not required.

You have two options, depending on whether you want to install the torchani compiled extensions. To install torchani with no compiled extensions run:
```bash
pip install --no-deps -v .
```

To install torchani with the cuAEV compiled extension run instead:

```bash
# Use 'ext-all-sms' instead of 'ext' if you want to build for all possible GPUs
pip install --config-settings=--global-option=ext --no-build-isolation --no-deps -v .
```

In both cases you can add the editable, `-e`, flag after the verbose, `-v`,
flag if you want an editable install (for developers). The `-v` flag can of
course be omitted, but it is sometimes handy to have some extra information
about the installation process.

After this you can perform some optional steps if you installed the required
dev dependencies:

```bash
# Download files needed for testing and building the docs (optional)
bash ./download.sh

# Build the documentation (optional)
sphinx-build docs/src docs/build

# Manually run unit tests (optional)
cd ./tests
pytest -v .
```

This process works for most use cases, but for more details regarding building
the CUDA and C++ extensions refer to [TorchANI CSRC](external/torchani_sandbox/torchani/csrc).

## Usage

### To run LEGOLAS, use the following command:
```bash
# atypes = HA, H, CA, CB, C, N
python legolas.py {coordinates_file(s)} [-b {BATCH_SIZE}] [-atype {INTERESTED_ATYPES}] [-t {TOPOLOGY}]
```

### Examples:
```bash
# All atom types:
python legolas.py data/304temp.pdb

# Specfy atom types:
python legolas.py data/304temp.pdb -atype H,C,N

# Run on molecular dynamics trajectory:
python legolas.py data/304_BBL.nc -t data/304_BBL.parm7
```

## Contributing

If you find a bug or have some feature request, please feel free to open an issue on GitHub or send us a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Citation

Please cite the following paper if you use LEGOLAS:

Mikayla Y. Darrows, Dimuthu Kodituwakku, Jinze Xue, Ignacio Pickering, and Adrian E. Roitberg. LEGOLAS: a machine learning method for rapid and accurate predictions of protein NMR chemical shifts.
