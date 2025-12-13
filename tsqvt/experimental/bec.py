"""BEC Simulator"""
import numpy as np

class BECSimulator:
    def __init__(self, rho_target=2/3):
        self.rho = rho_target
        self.c = 299792458
    
    def compute_sound_speed(self):
        """Compute sound speed."""
        rho = self.rho
        c_s = self.c * np.sqrt(rho * (4 - 3*rho) / (3*(1-rho)))
        return c_s
