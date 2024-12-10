# External Submodules

This directory contains submodules required for Legolas.

## torchani_sandbox

The `torchani_sandbox` submodule links to the [TorchANI](https://github.com/aiqm/torchani) repository. TorchANI provides functions essential for `legolas`, including molecular simulations and ANI model functions.

### Note
- Ensure the submodule is initialized and updated before running any code that depends on it:

  ```bash
  git submodule update --init --recursive
