from __future__ import annotations

from typing import Tuple

import numpy as np

Array = np.ndarray


def sym(A: Array) -> Array:
    """Symmetric part of a second-order tensor."""
    return 0.5 * (A + A.T)


def deviator(A: Array) -> Array:
    """Deviatoric part of a second-order tensor."""
    return A - np.trace(A) / 3.0 * np.eye(3)


def frob_norm(A: Array) -> float:
    """Frobenius norm."""
    return float(np.linalg.norm(A))


def spectral_decomposition_symmetric(A: Array) -> Tuple[Array, Array]:
    """
    Eigen-decomposition for a symmetric 3x3 tensor.

    Returns
    -------
    eigvals : (3,) array
        Eigenvalues sorted in ascending order.
    eigvecs : (3, 3) array
        Columns are the corresponding orthonormal eigenvectors.
    """
    A = sym(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvals, eigvecs


def dyad(a: Array, b: Array) -> Array:
    """Outer product a ⊗ b."""
    return np.outer(a, b)


def voigt_map_ijkl_to_mn(i: int, j: int) -> int:
    """
    Standard engineering Voigt mapping for symmetric 2nd-order tensors:
        11->0, 22->1, 33->2, 23->3, 13->4, 12->5
    """
    if (i, j) in [(0, 0)]:
        return 0
    if (i, j) in [(1, 1)]:
        return 1
    if (i, j) in [(2, 2)]:
        return 2
    if set((i, j)) == {1, 2}:
        return 3
    if set((i, j)) == {0, 2}:
        return 4
    if set((i, j)) == {0, 1}:
        return 5
    raise ValueError(f"Invalid Voigt indices ({i}, {j})")


def tensor4_to_voigt(C4: Array) -> Array:
    """
    Convert a 4th-order tensor (3x3x3x3) to a 6x6 matrix using engineering Voigt notation.
    """
    V = np.zeros((6, 6))
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            V[a, b] = C4[i, j, k, l]
    return V
