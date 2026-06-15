from __future__ import annotations

import numpy as np

from .finite_strain_von_mises_model import FiniteStrainVonMisesIsotropicHardening
from .material_state import InternalState, MaterialParameters


def example_usage() -> None:
    """Small usage example."""
    params = MaterialParameters(
        mu=80_000.0,
        kappa=160_000.0, # 体积模量
        H=5_000.0,
        sigma_y0=250.0,
    )
    model = FiniteStrainVonMisesIsotropicHardening(params)

    state = InternalState.identity(epbar=0.0)

    F = np.array([
        [1.10, 0.02, 0.00],
        [0.00, 0.95, 0.01],
        [0.00, 0.00, 1.02],
    ])

    out = model.update(F, state, compute_tangent=True)

    print("Yielded:", out.yielded)
    print("delta_gamma:", out.delta_gamma)
    print("epbar_new:", out.epbar)
    print("Cauchy stress:\n", out.stress)
    if out.tangent_voigt is not None:
        print("Tangent (Voigt):\n", out.tangent_voigt)


if __name__ == "__main__":
    example_usage()
