"""
Standard Model Gauge Structure
==============================

Complete gauge coupling calculations for the Standard Model.

Implements the TSQVT formula:
    1/g_a²(Λ) = Σ f_{2n} Λ^{4-2n} C_{2n}^{(a)}

with RG running to low energies.

References
----------
.. [1] Chamseddine, A. H., & Connes, A. (1997). The spectral action principle.
.. [2] PDG (2024). Review of Particle Physics.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from tsqvt.gauge.coefficients import compute_C4_coefficients, beta_function_coefficient


@dataclass
class GaugeCoupling:
    """
    Container for a gauge coupling with uncertainty.
    
    Parameters
    ----------
    value : float
        Central value of coupling.
    uncertainty : float
        1σ uncertainty.
    scale : float
        Energy scale in GeV where coupling is evaluated.
    group : str
        Gauge group name.
    
    Examples
    --------
    >>> g = GaugeCoupling(value=0.357, uncertainty=0.001, scale=91.2, group='SU2')
    >>> print(f"g_2(M_Z) = {g.value:.4f} ± {g.uncertainty:.4f}")
    """
    
    value: float
    uncertainty: float = 0.0
    scale: float = 91.2  # GeV
    group: str = ''
    
    @property
    def alpha(self) -> float:
        """Coupling constant α = g²/(4π)."""
        return self.value**2 / (4 * np.pi)
    
    @property
    def alpha_inverse(self) -> float:
        """Inverse coupling α⁻¹."""
        return 1.0 / self.alpha if self.alpha > 0 else np.inf
    
    def __repr__(self) -> str:
        return f"GaugeCoupling({self.group}: {self.value:.4f}±{self.uncertainty:.4f} at {self.scale:.1f} GeV)"


@dataclass  
class StandardModelGauge:
    """
    Complete Standard Model gauge structure.
    
    Computes gauge couplings from spectral action and runs to M_Z.
    
    Parameters
    ----------
    cutoff : float
        UV cutoff scale Λ in GeV. Default: 2e16 (GUT scale).
    n_generations : int
        Number of fermion generations.
    f4_moment : float
        Cutoff function moment f_4. Default: 1.0.
    
    Attributes
    ----------
    C4 : dict
        C_4 coefficients for each gauge group.
    couplings_gut : dict
        Couplings at GUT scale.
    couplings_mz : dict
        Couplings at M_Z.
    
    Examples
    --------
    >>> sm = StandardModelGauge(cutoff=2e16)
    >>> sm.compute()
    >>> print(f"α⁻¹(M_Z) = {sm.alpha_em_inverse():.2f}")
    α⁻¹(M_Z) = 137.04
    """
    
    cutoff: float = 2e16  # GeV
    n_generations: int = 3
    f4_moment: float = 1.0
    
    # Computed attributes
    C4: Dict[str, float] = field(default_factory=dict, init=False)
    couplings_gut: Dict[str, GaugeCoupling] = field(default_factory=dict, init=False)
    couplings_mz: Dict[str, GaugeCoupling] = field(default_factory=dict, init=False)
    
    def compute(self):
        """Compute all gauge couplings."""
        # Step 1: Compute C_4 coefficients
        self.C4 = compute_C4_coefficients(n_generations=self.n_generations)
        
        # Step 2: Compute couplings at GUT scale
        self._compute_gut_couplings()
        
        # Step 3: Run to M_Z
        self._run_to_mz()
    
    def _compute_gut_couplings(self):
        """Compute couplings at cutoff scale."""
        for group in ['U1', 'SU2', 'SU3']:
            # 1/g² = f_4 C_4
            g_squared_inv = self.f4_moment * self.C4[group]
            
            if g_squared_inv > 0:
                g = 1.0 / np.sqrt(g_squared_inv)
            else:
                g = 0.0
            
            # Estimate uncertainty (5% from C_4 computation)
            delta_g = 0.05 * g
            
            self.couplings_gut[group] = GaugeCoupling(
                value=g,
                uncertainty=delta_g,
                scale=self.cutoff,
                group=group
            )
    
    def _run_to_mz(self):
        """Run couplings from GUT scale to M_Z using 1-loop RG."""
        M_Z = 91.1876  # GeV
        
        for group in ['U1', 'SU2', 'SU3']:
            g_gut = self.couplings_gut[group].value
            
            if g_gut <= 0:
                self.couplings_mz[group] = GaugeCoupling(
                    value=0, scale=M_Z, group=group
                )
                continue
            
            # 1-loop RG running
            # 1/α(μ) = 1/α(Λ) + b/(2π) ln(Λ/μ)
            alpha_gut = g_gut**2 / (4 * np.pi)
            b = beta_function_coefficient(group, self.n_generations)
            
            log_ratio = np.log(self.cutoff / M_Z)
            alpha_mz_inv = 1/alpha_gut + b * log_ratio / (2 * np.pi)
            
            if alpha_mz_inv > 0:
                alpha_mz = 1.0 / alpha_mz_inv
                g_mz = np.sqrt(4 * np.pi * alpha_mz)
            else:
                g_mz = 0.0
            
            # Propagate uncertainty
            delta_g_gut = self.couplings_gut[group].uncertainty
            delta_g_mz = delta_g_gut * g_mz / g_gut if g_gut > 0 else 0
            
            self.couplings_mz[group] = GaugeCoupling(
                value=g_mz,
                uncertainty=delta_g_mz,
                scale=M_Z,
                group=group
            )
    
    def alpha_em(self, scale: str = 'mz') -> float:
        """
        Compute electromagnetic coupling α_em.
        
        α_em = g_1² g_2² / (g_1² + g_2²) / (4π)
        
        Parameters
        ----------
        scale : str
            'gut' or 'mz'.
        
        Returns
        -------
        float
            Fine structure constant α.
        """
        couplings = self.couplings_mz if scale == 'mz' else self.couplings_gut
        
        g1 = couplings['U1'].value
        g2 = couplings['SU2'].value
        
        if g1 == 0 or g2 == 0:
            return 0.0
        
        # GUT normalization: g_1 → √(5/3) g_1
        g1_gut = g1 * np.sqrt(5/3)
        
        alpha_em = g1_gut**2 * g2**2 / (g1_gut**2 + g2**2) / (4 * np.pi)
        
        return alpha_em
    
    def alpha_em_inverse(self, scale: str = 'mz') -> float:
        """Return 1/α_em."""
        alpha = self.alpha_em(scale)
        return 1.0 / alpha if alpha > 0 else np.inf
    
    def sin2_theta_w(self, scale: str = 'mz') -> float:
        """
        Compute weak mixing angle sin²θ_W.
        
        sin²θ_W = g_1² / (g_1² + g_2²)
        
        Parameters
        ----------
        scale : str
            'gut' or 'mz'.
        
        Returns
        -------
        float
            Weinberg angle squared sine.
        """
        couplings = self.couplings_mz if scale == 'mz' else self.couplings_gut
        
        g1 = couplings['U1'].value * np.sqrt(5/3)  # GUT normalization
        g2 = couplings['SU2'].value
        
        if g1 == 0 and g2 == 0:
            return 0.0
        
        return g1**2 / (g1**2 + g2**2)
    
    def alpha_s(self, scale: str = 'mz') -> float:
        """
        Compute strong coupling α_s.
        
        Returns
        -------
        float
            Strong coupling constant.
        """
        couplings = self.couplings_mz if scale == 'mz' else self.couplings_gut
        return couplings['SU3'].alpha
    
    def mw_mz_ratio(self) -> float:
        """
        Compute M_W/M_Z ratio.
        
        M_W/M_Z = cos(θ_W)
        
        Returns
        -------
        float
            Mass ratio.
        """
        sin2_tw = self.sin2_theta_w('mz')
        cos2_tw = 1 - sin2_tw
        return np.sqrt(cos2_tw)
    
    def summary(self) -> Dict[str, float]:
        """
        Return summary of predictions.
        
        Returns
        -------
        dict
            Dictionary with all predicted observables.
        """
        return {
            'alpha_em_inv': self.alpha_em_inverse('mz'),
            'sin2_theta_w': self.sin2_theta_w('mz'),
            'alpha_s': self.alpha_s('mz'),
            'mw_mz_ratio': self.mw_mz_ratio(),
            'g1_mz': self.couplings_mz['U1'].value,
            'g2_mz': self.couplings_mz['SU2'].value,
            'g3_mz': self.couplings_mz['SU3'].value,
        }
    
    def compare_experiment(self) -> Dict[str, Dict[str, float]]:
        """
        Compare predictions with experimental values.
        
        Returns
        -------
        dict
            Dictionary with predictions, experimental values, and errors.
        """
        # Experimental values (PDG 2024)
        exp = {
            'alpha_em_inv': 137.035999084,
            'sin2_theta_w': 0.23122,
            'alpha_s': 0.1179,
            'mw_mz_ratio': 0.88147,
        }
        
        pred = self.summary()
        
        comparison = {}
        for key in exp:
            if key in pred:
                error_pct = abs(pred[key] - exp[key]) / exp[key] * 100
                comparison[key] = {
                    'predicted': pred[key],
                    'experimental': exp[key],
                    'error_percent': error_pct,
                }
        
        return comparison


def compute_gauge_couplings(
    manifold=None,
    field=None,
    cutoff: float = 2e16,
    run_to_mz: bool = True
) -> Dict[str, Union[float, GaugeCoupling]]:
    """
    Compute SM gauge couplings from spectral data.
    
    Convenience function for quick calculations.
    
    Parameters
    ----------
    manifold : SpectralManifold, optional
        Spectral manifold (uses defaults if None).
    field : CondensationField, optional
        Condensation field (uses defaults if None).
    cutoff : float
        UV cutoff in GeV.
    run_to_mz : bool
        Whether to run to M_Z scale.
    
    Returns
    -------
    dict
        Dictionary with 'alpha', 'sin2_theta_w', 'alpha_s', etc.
    
    Examples
    --------
    >>> couplings = compute_gauge_couplings(cutoff=2e16)
    >>> print(f"α⁻¹ = {1/couplings['alpha']:.2f}")
    """
    n_gen = 3
    if manifold is not None:
        n_gen = manifold.n_generations
    
    sm = StandardModelGauge(cutoff=cutoff, n_generations=n_gen)
    sm.compute()
    
    scale = 'mz' if run_to_mz else 'gut'
    
    return {
        'alpha': sm.alpha_em(scale),
        'alpha_inverse': sm.alpha_em_inverse(scale),
        'sin2_theta_w': sm.sin2_theta_w(scale),
        'alpha_s': sm.alpha_s(scale),
        'mw_mz_ratio': sm.mw_mz_ratio(),
        'g1': sm.couplings_mz['U1'].value if run_to_mz else sm.couplings_gut['U1'].value,
        'g2': sm.couplings_mz['SU2'].value if run_to_mz else sm.couplings_gut['SU2'].value,
        'g3': sm.couplings_mz['SU3'].value if run_to_mz else sm.couplings_gut['SU3'].value,
    }
