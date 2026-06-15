from __future__ import annotations

from typing import Tuple

import numpy as np

from .material_state import InternalState, MaterialParameters, UpdateResult
from .tensor_ops import (
    Array,
    dyad,
    spectral_decomposition_symmetric,
    sym,
    tensor4_to_voigt,
)


class FiniteStrainVonMisesIsotropicHardening:
    """
    Finite-strain, rate-independent von Mises plasticity with isotropic hardening.
    """

    def __init__(self, params: MaterialParameters):
        self.p = params

    def pressure(self, J: float) -> float:
        """
        Mean stress / pressure term.
        """
        return self.p.kappa * np.log(J) / J

    def trial_state(self, F: Array, Cp_inv_n: Array) -> Tuple[float, Array, Array, Array, Array]:
        """
        Build trial state.
        """
        J = float(np.linalg.det(F))
        b_e_trial = F @ Cp_inv_n @ F.T
        b_e_trial = sym(b_e_trial)

        eigvals, eigvecs = spectral_decomposition_symmetric(b_e_trial)
        eigvals = np.clip(eigvals, 1e-16, None) # 因为是正定矩阵，所以特征值应该是正的，但数值误差可能导致小的负值
        lambdas_trial = np.sqrt(eigvals)

        log_lam = np.log(lambdas_trial)
        tau_dev_trial_principal = 2.0 * self.p.mu * log_lam - (2.0 / 3.0) * self.p.mu * np.log(J)
        tau_dev_trial_principal = tau_dev_trial_principal - np.mean(tau_dev_trial_principal)

        return J, b_e_trial, lambdas_trial, eigvecs, tau_dev_trial_principal

    def equivalent_stress(self, tau_dev_principal: Array) -> float:
        """Von Mises equivalent Kirchhoff stress."""
        return float(np.sqrt(3.0 / 2.0) * np.linalg.norm(tau_dev_principal))

    def yield_function(self, tau_dev_principal: Array, epbar_n: float) -> float:
        """f = q - sigma_y(epbar)."""
        q = self.equivalent_stress(tau_dev_principal)
        sigma_y = self.p.sigma_y0 + self.p.H * epbar_n
        return q - sigma_y

    def radial_return(
        self,
        tau_dev_trial_principal: Array,
        epbar_n: float,
    ) -> Tuple[Array, float, float, bool]:
        """
        Radial return algorithm in principal space.
        """
        f_trial = self.yield_function(tau_dev_trial_principal, epbar_n)

        if f_trial <= 0.0:
            return tau_dev_trial_principal.copy(), 0.0, f_trial, False

        q_trial = self.equivalent_stress(tau_dev_trial_principal)
        if q_trial <= 1e-16:
            return tau_dev_trial_principal.copy(), 0.0, f_trial, False

        delta_gamma = f_trial / (3.0 * self.p.mu + self.p.H)
        scale = max(0.0, 1.0 - 3.0 * self.p.mu * delta_gamma / q_trial)
        tau_dev_new_principal = scale * tau_dev_trial_principal

        return tau_dev_new_principal, delta_gamma, f_trial, True

    def update_b_e(self, lambdas_new: Array, principal_directions: Array) -> Array:
        """Reconstruct elastic left Cauchy-Green tensor from principal data."""
        b_e = np.zeros((3, 3))
        for a in range(3):
            n = principal_directions[:, a]
            b_e += (lambdas_new[a] ** 2) * dyad(n, n)
        return sym(b_e)

    def update_stress(
        self,
        J: float,
        tau_dev_principal: Array,
        principal_directions: Array,
    ) -> Tuple[Array, Array, Array]:
        """Assemble full Kirchhoff and Cauchy stress tensors."""
        p = self.pressure(J)
        tau_dev = np.zeros((3, 3))
        for a in range(3):
            n = principal_directions[:, a]
            tau_dev += tau_dev_principal[a] * dyad(n, n)

        tau = tau_dev + p * np.eye(3)
        sigma = tau / J
        return sigma, tau_dev, tau

    def principal_tangent_coefficients(
        self,
        tau_dev_trial_principal: Array,
        delta_gamma: float,
        yielded: bool,
    ) -> Array:
        """
        Compute the 3x3 principal tangent coefficients c_ab.
        """
        c_ab = np.zeros((3, 3))
        q_trial = self.equivalent_stress(tau_dev_trial_principal)
        if not yielded or q_trial <= 1e-16:
            for a in range(3):
                for b in range(3):
                    c_ab[a, b] = 2.0 * self.p.mu * (1.0 if a == b else 0.0) - (2.0 / 3.0) * self.p.mu
            return c_ab

        
        nu = np.sqrt(3.0 / 2.0) * tau_dev_trial_principal / np.linalg.norm(tau_dev_trial_principal)

        for a in range(3):
            for b in range(3):
                delta_ab = 1.0 if a == b else 0.0
                first = (1.0 - 3.0 * self.p.mu * delta_gamma / q_trial) * (
                    2.0 * self.p.mu * delta_ab - (2.0 / 3.0) * self.p.mu
                )
                second = -2.0 * self.p.mu * nu[a] * nu[b] * (
                    2.0 * self.p.mu / (3.0 * self.p.mu + self.p.H)
                    - 2.0 * self.p.mu * np.sqrt(2.0 / 3.0) * delta_gamma / np.linalg.norm(tau_dev_trial_principal)
                )
                c_ab[a, b] = first + second

        return c_ab

    def consistent_tangent_principal_basis(
        self,
        J: float,
        lambdas_trial: Array,
        principal_directions: Array,
        sigma_dev_principal: Array,
        c_ab: Array,
    ) -> Array:
        """
        Build a 4th-order tangent tensor in the current principal basis.
        """
        C4 = np.zeros((3, 3, 3, 3))

        for a in range(3):
            na = principal_directions[:, a]
            for b in range(3):
                nb = principal_directions[:, b]
                C4 += (c_ab[a, b] / J) * np.einsum("i,j,k,l->ijkl", na, na, nb, nb) # 构造四阶张量

        for a in range(3):
            na = principal_directions[:, a]
            C4 -= 2.0 * sigma_dev_principal[a] * np.einsum("i,j,k,l->ijkl", na, na, na, na)

        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                la2 = lambdas_trial[a] ** 2
                lb2 = lambdas_trial[b] ** 2
                denom = la2 - lb2
                if abs(denom) < 1e-14:
                    continue
                coeff = (sigma_dev_principal[a] * lb2 - sigma_dev_principal[b] * la2) / denom
                na = principal_directions[:, a]
                nb = principal_directions[:, b]
                C4 += coeff * (
                    np.einsum("i,j,k,l->ijkl", na, na, nb, nb)
                    + np.einsum("i,j,k,l->ijkl", nb, nb, na, na)
                )

        # 消除数值误差，满足对称性
        C4 = 0.25 * (C4 + C4.swapaxes(0, 1) + C4.swapaxes(2, 3) + C4.swapaxes(0, 1).swapaxes(2, 3)) # 小对称，对应应力和应变率对称
        C4 = 0.5 * (C4 + C4.swapaxes(0, 2).swapaxes(1, 3)) # 大对称，因为应变能二阶导数连续，可交换求导顺序
        return C4

    def update(
        self,
        F: Array,
        state_n: InternalState,
        compute_tangent: bool = False,
    ) -> UpdateResult:
        """
        Constitutive update for one load increment.
        """
        J, b_e_trial, lambdas_trial, principal_directions, tau_dev_trial_principal = self.trial_state(
            F, state_n.Cp_inv
        )

        tau_dev_new_principal, delta_gamma, f_trial, yielded = self.radial_return(
            tau_dev_trial_principal, state_n.epbar
        )

        if yielded:
            nu = np.sqrt(3.0 / 2.0) * tau_dev_trial_principal / np.linalg.norm(tau_dev_trial_principal)
            log_lam_new = np.log(lambdas_trial) - delta_gamma * nu
            lambdas_new = np.exp(log_lam_new)
        else:
            lambdas_new = lambdas_trial.copy()

        b_e_new = self.update_b_e(lambdas_new, principal_directions)
        F_inv = np.linalg.inv(F)
        Cp_inv_new = F_inv @ b_e_new @ F_inv.T
        Cp_inv_new = sym(Cp_inv_new)

        sigma, tau_dev_new, tau_new = self.update_stress(J, tau_dev_new_principal, principal_directions)
        epbar_new = state_n.epbar + delta_gamma

        tangent_voigt = None
        if compute_tangent:
            c_ab = self.principal_tangent_coefficients(tau_dev_trial_principal, delta_gamma, yielded)
            C4 = self.consistent_tangent_principal_basis(
                J=J,
                lambdas_trial=lambdas_trial,
                principal_directions=principal_directions,
                sigma_dev_principal=tau_dev_new_principal if yielded else tau_dev_trial_principal,
                c_ab=c_ab,
            )
            tangent_voigt = tensor4_to_voigt(C4) # voigt格式的切线模量矩阵，6x6

        return UpdateResult(
            stress=sigma,
            kirchhoff_dev=tau_dev_new,
            kirchhoff=tau_new,
            b_e=b_e_new,
            Cp_inv=Cp_inv_new,
            epbar=epbar_new,
            delta_gamma=delta_gamma,
            yielded=yielded,
            principal_directions=principal_directions,
            principal_lambdas=lambdas_new,
            tangent_voigt=tangent_voigt,
        )
