"""Tests for gauge module"""
import pytest
import numpy as np
from tsqvt.gauge import compute_C4_coefficients

def test_c4_computation():
    yukawa = {'e': 0.3e-5, 'mu': 6e-4, 'tau': 0.01}
    majorana = {'nu1': 1e12, 'nu2': 1e13, 'nu3': 1e14}
    
    C4 = compute_C4_coefficients(yukawa, majorana)
    
    assert 'C4_U1' in C4
    assert 'C4_SU2' in C4
    assert 'C4_SU3' in C4
    assert all(v > 0 for v in C4.values())
