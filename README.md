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
