from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class MaterialParameters:
    """Material parameters for finite-strain J2 plasticity."""
    mu: float
    kappa: float
    H: float
    sigma_y0: float = 0.0


@dataclass
class InternalState:
    """
    Internal variables at step n.
    Cp_inv : inverse plastic right Cauchy-Green tensor at step n
    epbar  : equivalent plastic strain at step n
    """
    Cp_inv: Array
    epbar: float

    @staticmethod
    def identity(epbar: float = 0.0) -> "InternalState":
        return InternalState(Cp_inv=np.eye(3), epbar=epbar)


@dataclass
class UpdateResult:
    """Output of the constitutive update."""
    stress: Array
    kirchhoff_dev: Array
    kirchhoff: Array
    b_e: Array
    Cp_inv: Array
    epbar: float
    delta_gamma: float
    yielded: bool
    principal_directions: Array
    principal_lambdas: Array
    tangent_voigt: Array | None = None
