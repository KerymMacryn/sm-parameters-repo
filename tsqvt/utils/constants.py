"""
Physical Constants and Parameters
==================================

Fundamental constants and Standard Model parameters.

All values in SI units unless otherwise specified.
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

# Planck's constant (J·s)
HBAR = 1.054571817e-34

# Speed of light (m/s)
C = 299792458.0

# Gravitational constant (m³ kg⁻¹ s⁻²)
G = 6.67430e-11

# Fine structure constant (dimensionless)
ALPHA_EM = 1.0 / 137.035999084

# Electron charge (C)
E_CHARGE = 1.602176634e-19

# Boltzmann constant (J/K)
K_B = 1.380649e-23

# ============================================================================
# PARTICLE MASSES (GeV)
# ============================================================================

# Gauge bosons
M_Z = 91.1876  # Z boson
M_W = 80.377   # W boson
M_H = 125.10   # Higgs boson

# Leptons
M_E = 0.5109989461e-3  # electron
M_MU = 0.1056583745    # muon
M_TAU = 1.77686        # tau

# Quarks (MS-bar at 2 GeV)
M_U = 2.16e-3   # up
M_D = 4.67e-3   # down
M_S = 93.4e-3   # strange
M_C = 1.27      # charm
M_B = 4.18      # bottom
M_T = 172.69    # top (pole mass)

# ============================================================================
# GAUGE COUPLINGS (at m_Z)
# ============================================================================

ALPHA_EM_MZ = 1.0 / 127.955  # Electromagnetic (at m_Z)
ALPHA_S_MZ = 0.1179         # Strong (at m_Z)
SIN2_THETAW = 0.23122       # Weinberg angle

# ============================================================================
# CKM MATRIX ELEMENTS
# ============================================================================

V_UD = 0.97401
V_US = 0.22650
V_UB = 0.00361
V_CD = 0.22636
V_CS = 0.97320
V_CB = 0.04053
V_TD = 0.00854
V_TS = 0.03978
V_TB = 0.999105

# Jarlskog invariant
J_CP = 3.08e-5

# ============================================================================
# PMNS MATRIX ELEMENTS
# ============================================================================

# Solar angle
THETA12_PMNS = np.arcsin(np.sqrt(0.307))  # sin²θ₁₂ = 0.307

# Atmospheric angle  
THETA23_PMNS = np.arcsin(np.sqrt(0.545))  # sin²θ₂₃ = 0.545

# Reactor angle
THETA13_PMNS = np.arcsin(np.sqrt(0.0220))  # sin²θ₁₃ = 0.0220

# CP phase (unknown)
DELTA_CP_PMNS = 1.36 * np.pi  # ≈ 222°

# ============================================================================
# PLANCK SCALE QUANTITIES
# ============================================================================

# Planck mass (kg)
M_PLANCK_KG = np.sqrt(HBAR * C / G)

# Planck mass (GeV)
M_PLANCK = M_PLANCK_KG * C**2 / E_CHARGE / 1e9  # ~1.22e19 GeV

# Planck length (m)
L_PLANCK = np.sqrt(HBAR * G / C**3)  # ~1.616e-35 m

# Planck time (s)
T_PLANCK = np.sqrt(HBAR * G / C**5)  # ~5.391e-44 s

# ============================================================================
# TSQVT PARAMETERS
# ============================================================================

# Geometric parameters (default values)
TSQVT_DEFAULTS = {
    'V_Sigma': 1.85e-61,      # m⁴ (spectral manifold volume)
    'theta_twist': 0.198,     # rad (fibration angle)
    'rho_EW': 0.742,          # dimensionless (EW condensation)
    'xi_Yukawa': 2.34,        # dimensionless (Yukawa normalization)
}

# Spectral scale
A_SPECTRAL = 1.85e16  # GeV

# ============================================================================
# STANDARD MODEL PARAMETER DICTIONARY
# ============================================================================

SM_PARAMETERS = {
    # Gauge couplings
    'alpha_em_mZ': ALPHA_EM_MZ,
    'alpha_s_mZ': ALPHA_S_MZ,
    'sin2_thetaW': SIN2_THETAW,
    
    # Gauge boson masses
    'm_Z': M_Z,
    'm_W': M_W,
    'm_H': M_H,
    
    # Lepton masses
    'm_e': M_E,
    'm_mu': M_MU,
    'm_tau': M_TAU,
    
    # Quark masses
    'm_u': M_U,
    'm_d': M_D,
    'm_s': M_S,
    'm_c': M_C,
    'm_b': M_B,
    'm_t': M_T,
    
    # CKM elements
    'V_us': V_US,
    'V_cb': V_CB,
    'V_ub': V_UB,
    'J_CP': J_CP,
    
    # PMNS angles
    'sin2_theta12': np.sin(THETA12_PMNS)**2,
    'sin2_theta23': np.sin(THETA23_PMNS)**2,
    'sin2_theta13': np.sin(THETA13_PMNS)**2,
}

# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

# Energy
GEV_TO_JOULE = E_CHARGE * 1e9
JOULE_TO_GEV = 1.0 / GEV_TO_JOULE

# Length
METER_TO_GEV_INV = HBAR * C / GEV_TO_JOULE
GEV_INV_TO_METER = 1.0 / METER_TO_GEV_INV

# Time
SECOND_TO_GEV_INV = HBAR / GEV_TO_JOULE
GEV_INV_TO_SECOND = 1.0 / SECOND_TO_GEV_INV

# ============================================================================
# BETA FUNCTION COEFFICIENTS (1-loop, MS-bar)
# ============================================================================

# SU(3) × SU(2) × U(1)
BETA_1LOOP = {
    'b1': 41/6,     # U(1)_Y
    'b2': -19/6,    # SU(2)_L
    'b3': -7,       # SU(3)_C
}

# 2-loop
BETA_2LOOP = {
    'b11': 199/50,
    'b12': 27/10,
    'b13': 44/5,
    'b22': -35/6,
    'b23': 9,
    'b33': -52/3,
}

# ============================================================================
# EXPERIMENTAL UNCERTAINTIES
# ============================================================================

EXPERIMENTAL_ERRORS = {
    'alpha_em_mZ': 0.000023,
    'alpha_s_mZ': 0.0010,
    'sin2_thetaW': 0.00015,
    'm_Z': 0.0021,  # GeV
    'm_W': 0.012,
    'm_t': 0.76,
    'm_e': 0.0000000013e-3,
    'm_mu': 0.000000024,
    'm_tau': 0.00012,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_particle_mass(particle: str) -> float:
    """
    Get particle mass in GeV.
    
    Parameters
    ----------
    particle : str
        Particle name ('e', 'mu', 'tau', 't', etc.).
    
    Returns
    -------
    float
        Mass in GeV.
    
    Examples
    --------
    >>> get_particle_mass('t')
    172.69
    """
    mass_dict = {
        'e': M_E, 'mu': M_MU, 'tau': M_TAU,
        'u': M_U, 'd': M_D, 's': M_S,
        'c': M_C, 'b': M_B, 't': M_T,
        'Z': M_Z, 'W': M_W, 'H': M_H,
    }
    
    if particle not in mass_dict:
        raise ValueError(f"Unknown particle: {particle}")
    
    return mass_dict[particle]


def hbar_c() -> float:
    """
    Return ℏc in GeV·fm.
    
    Returns
    -------
    float
        ℏc ≈ 0.1973 GeV·fm
    """
    return HBAR * C * 1e15 / GEV_TO_JOULE


def fine_structure_constant(energy: float = M_Z) -> float:
    """
    Running fine structure constant (approximate).
    
    Parameters
    ----------
    energy : float, optional
        Energy scale in GeV (default: m_Z).
    
    Returns
    -------
    float
        α(E)
    
    Notes
    -----
    Uses simple approximation α(E) ≈ α(m_Z) [1 + α/(3π) log(E/m_Z)]
    """
    alpha_mZ = ALPHA_EM_MZ
    if abs(energy - M_Z) < 1e-6:
        return alpha_mZ
    
    # Simple 1-loop running
    L = np.log(energy / M_Z)
    dalpha = alpha_mZ**2 / (3 * np.pi) * L
    
    return alpha_mZ * (1 + dalpha / alpha_mZ)
