"""Tests for RG module"""
import pytest
import numpy as np
from tsqvt.rg import RGRunner

def test_rg_runner_creation():
    rg = RGRunner(n_loops=2)
    assert rg.n_loops == 2

def test_rg_running():
    rg = RGRunner()
    g_final = rg.run(g_initial=0.5, energy_initial=1e16, energy_final=91.1876)
    assert 0 < g_final < 1
