"""Tests for spectral module"""
import pytest
import numpy as np
from tsqvt.spectral import HeatKernel

def test_heat_kernel_creation():
    hk = HeatKernel(dimension=4)
    assert hk.dimension == 4

def test_a0_coefficient():
    hk = HeatKernel()
    assert hk.compute_a0() == 1.0

def test_a2_coefficient():
    hk = HeatKernel()
    a2 = hk.compute_a2(R=0.1, E=0.01)
    assert a2 > 0
