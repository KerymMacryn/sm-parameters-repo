"""Collapse Predictions"""
import numpy as np

class CollapsePredictor:
    def __init__(self, mass, Delta_x, rho_particle=0.95):
        self.mass = mass
        self.Delta_x = Delta_x
        self.rho = rho_particle
        self.G = 6.674e-11
        self.hbar = 1.054571817e-34
    
    def compute_collapse_time(self):
        """Compute collapse time in seconds."""
        Delta_E = self.G * self.mass**2 / self.Delta_x
        tau = self.hbar / Delta_E * (1 - self.rho)
        return tau
