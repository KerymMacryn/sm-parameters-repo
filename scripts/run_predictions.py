#!/usr/bin/env python3
"""Run TSQVT Predictions Pipeline"""
import sys
import os

# Add parent directory to path (so tsqvt module is found)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tsqvt

def main():
    parser = argparse.ArgumentParser(description="Run TSQVT predictions")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("="*70)
    print("TSQVT: Running Predictions Pipeline")
    print("="*70)
    print(f"Version: {tsqvt.__version__}")
    
    # Create theory
    from tsqvt.core import SpectralManifold, CondensationField
    manifold = SpectralManifold()
    field = CondensationField()
    
    # Compute predictions
    from tsqvt.gauge import compute_C4_coefficients
    yukawa = {'e': 0.3e-5, 'mu': 6e-4, 'tau': 0.01}
    majorana = {'nu1': 1e12}
    
    C4 = compute_C4_coefficients(yukawa, majorana)
    
    print("\nC_4 Coefficients:")
    for key, val in C4.items():
        print(f"  {key} = {val:.6f}")
    
    # Experimental predictions
    from tsqvt.experimental import CollapsePredictor
    predictor = CollapsePredictor(mass=1e-14, Delta_x=100e-9)
    tau = predictor.compute_collapse_time()
    print(f"\nCollapse time: {tau*1000:.1f} ms")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
