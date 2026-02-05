# get_conv_transpose_3d_params.py
# Copyright (c) 2026 Hardik Ghoshal (TU Dresden)
# SPDX-License-Identifier: MIT
# Link to paper: https://arxiv.org/abs/xxxx.xxxxx

from math import floor
from typing import Tuple, Optional


def get_conv_transpose_3d_params(
    input_size: int,
    multiplier: int,
    kernel_size: int,
    allow_output_padding: bool = True
) -> Optional[Tuple[int, int, int]]:
    """
    Finds stride, padding, output_padding for ConvTranspose3d such that
    output spatial size = input_size * multiplier exactly,
    using stride = multiplier and dilation = 1.

    Returns (stride, padding, output_padding) or None if not possible.

    Args:
        input_size: Spatial dimension of the input (D/H/W - same for isotropic)
        multiplier: Desired scale factor m (output = m * input)
        kernel_size: 3D kernel size (cubic / isotropic)
        allow_output_padding: If False, forces output_padding=0 (more restrictive)

    Returns:
        Tuple (stride, padding, output_padding) or None
    """
    if multiplier < 1:
        raise ValueError("Multiplier must be >= 1")

    stride = multiplier
    target = multiplier * input_size

    possible_op = [0, 1] if allow_output_padding else [0]

    for op in possible_op:
        # This is equivalent to: n = kernel_size + op - multiplier
        # (the i terms cancel out)
        n = kernel_size + op - multiplier
        if n % 2 == 0:
            padding = n // 2
            if padding >= 0:
                # Verify
                computed_output = (input_size - 1) * stride - 2 * padding + (kernel_size - 1) + op + 1
                if computed_output == target:
                    return stride, padding, op
    return None

RESET   = "\033[0m"
GREEN   = "\033[32m"
RED     = "\033[31m"
GRAY    = "\033[90m"
YELLOW  = "\033[33m"
BOLD    = "\033[1m"

def demo():
  # input size, multiplier, kernel size
    test_cases = [
        (32, 2, 2),   # even k=2
        (32, 2, 3),   # odd  k=3
        (64, 2, 4),
        (16, 3, 3),
        (16, 3, 2),   # k = m-1, should work
        (16, 4, 2),   # k < m-1, should fail
        (128, 4, 2),   # k < m-1, should fail
        (128, 2, 1),  # too small kernel but k = m-1, should work
    ]
    print(f"{BOLD}{YELLOW}"
          f"{'Input':>6}   "
          f"{'m':>3}   "
          f"{'Kernel':>7}   "
          f"{'Stride':>7}   "
          f"{'Padding':>8}   "
          f"{'OutPad':>7}   "
          f"{'Output':>8}   "
          f"{'Status':<12}"
          f"{RESET}")
    print(f"{GRAY}{'-' * 80}{RESET}")
    for i, m, k in test_cases:
        result = get_conv_transpose_3d_params(i, m, k)
        if result:
            s, p, op = result
            out = (i - 1) * s - 2 * p + (k - 1) + op + 1
            status = f"{GREEN}Success{RESET}"
            color_s  = f"{YELLOW}{s:>7}{RESET}"
            color_p  = f"{YELLOW}{p:>8}{RESET}"
            color_op = f"{YELLOW}{op:>7}{RESET}"
            color_out = f"{YELLOW}{out:>8}{RESET}"
        else:
            status = f"{RED}Not possible{RESET}"
            color_s   = f"{GRAY}{'-':>7}{RESET}"
            color_p   = f"{GRAY}{'-':>8}{RESET}"
            color_op  = f"{GRAY}{'-':>7}{RESET}"
            color_out = f"{GRAY}{'-':>8}{RESET}"
        print(f"{i:>6}   "
              f"{m:>3}   "
              f"{k:>7}   "
              f"{color_s}   "
              f"{color_p}   "
              f"{color_op}   "
              f"{color_out}   "
              f"{status}")
    print(f"{GRAY}{'-' * 80}{RESET}")

if __name__ == "__main__":
    demo()