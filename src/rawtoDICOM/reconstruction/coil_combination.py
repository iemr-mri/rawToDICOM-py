"""Sum-of-squares coil combination.

Translates combineCoils.m.  The MATLAB original applies a forward FFT on k-space
(ifftshift → fft2 → fftshift), which is equivalent to an IFFT for Hermitian k-space
but is technically incorrect.  The Python version uses ifft2c consistently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from rawtoDICOM.reconstruction.kspace import ifft2c


def combine_coils(kspace: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Sum-of-squares coil combination.

    Translates combineCoils.m.

    Args:
        kspace: Complex array [x, y, slices, frames, coils].

    Returns:
        Real magnitude images [x, y, slices, frames] — coil axis removed.
    """
    images: npt.NDArray[Any] = ifft2c(kspace)  # [x, y, slices, frames, coils]
    result: npt.NDArray[Any] = np.sqrt(np.sum(np.abs(images) ** 2, axis=-1))
    return result
