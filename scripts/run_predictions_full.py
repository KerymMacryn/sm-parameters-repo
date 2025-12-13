#!/usr/bin/env python3
"""
TSQVT: Run Complete Predictions
================================

This script runs the complete TSQVT pipeline and generates
all predictions.

Usage:
    python run_predictions.py [--output OUTPUT_DIR] [--verbose]

Author: Kerym Sanjuan
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import tsqvt
except ImportError:
    print("Error: TSQVT package not found.")
    print("Install with: pip install -e .")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TSQVT predictions pipeline"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with experimental values"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TSQVT: Running Complete Predictions Pipeline")
    print("="*70)
    print(f"TSQVT version: {tsqvt.__version__}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Initialize theory
    if args.verbose:
        print("Initializing TSQVT theory with default parameters...")
    
    theory = tsqvt.TSQVTTheory()
    
    # Run full pipeline
    results = theory.run_full_pipeline(verbose=args.verbose)
    
    # Save results
    if args.save_json:
        output_file = output_dir / "predictions.json"
        
        # Convert numpy types to Python types for JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer))
                    else v for k, v in value.items()
                }
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    # Compare with experiment
    if args.compare:
        print("\n" + "="*70)
        print("COMPARISON WITH EXPERIMENT")
        print("="*70)
        
        comparison = theory.compare_with_experiment()
        
        print(f"\n{'Observable':<15} {'TSQVT':>12} {'Experiment':>12} "
              f"{'Error (%)':>12} {'Status':>10}")
        print("-"*70)
        
        for obs, data in comparison.items():
            status = "✓" if data['agreement'] else "✗"
            print(f"{obs:<15} {data['predicted']:>12.4f} "
                  f"{data['experimental']:>12.4f} "
                  f"{data['error_percent']:>12.2f} "
                  f"{status:>10}")
    
    # Print summary
    print("\n" + theory.get_summary())
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
