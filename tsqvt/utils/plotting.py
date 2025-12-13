"""Plotting Utilities"""
import matplotlib.pyplot as plt
import numpy as np

def plot_rg_running(energies, couplings, labels):
    """Plot RG running of couplings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for coupling, label in zip(couplings, labels):
        ax.plot(energies, coupling, label=label)
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('Coupling')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)
    return fig
