# ConvTranspose3d Parameter Calculator

A lightweight Python utility to find valid parameters (`stride`, `padding`, `output_padding`) for `torch.nn.ConvTranspose3d` (or equivalent in other frameworks) such that the output spatial size is **exactly** `input_size × m` (where `m` is the desired upsampling/multiplier factor).

This code accompanies the paper:

**Proofs for parameter calculation for isotropic kernels with uniform dilation of 1**  
Hardik Ghoshal  
Technische Universität Dresden  
[arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Features

- Computes exact upsampling parameters with dilation=1
- Restricts `stride = m` (common choice for clean upsampling)
- Supports `output_padding` ∈ {0,1} (most practical cases)
- Proves success condition: `kernel_size ≥ m - 1`
- Includes colored terminal demo output for quick testing

## Installation

No external dependencies required (uses only Python standard library + optional `math`).

```bash
# Just copy the file or clone this repo
git clone https://github.com/yourusername/convtranspose3d-params.git
