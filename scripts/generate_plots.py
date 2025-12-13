#!/usr/bin/env python3
"""Generate plots for TSQVT predictions"""
import numpy as np
import matplotlib.pyplot as plt
from tsqvt.utils.plotting import plot_rg_running

def main():
    # Example: RG running
    energies = np.logspace(2, 16, 100)
    alpha_inv = 137 + 10 * np.log(energies / 91.1876)
    
    fig = plot_rg_running([energies], [alpha_inv], ['α⁻¹'])
    fig.savefig('rg_running.png', dpi=300)
    print("Plot saved: rg_running.png")

if __name__ == "__main__":
    main()
