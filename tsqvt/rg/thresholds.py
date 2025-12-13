"""Threshold Corrections"""
def compute_threshold_correction(mass, scale):
    """Compute threshold correction at mass scale."""
    import numpy as np
    return np.log(scale / mass) if scale > mass else 0.0
