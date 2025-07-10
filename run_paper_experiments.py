#!/usr/bin/env python3
"""
Script to reproduce SparseEA-AGDS paper experiments
This script follows the exact experimental setup described in the paper
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from experiments import ExperimentRunner, ResultsAnalyzer
from config import StandardConfigs
from problems import create_smop_problem
from algorithms import SparseEAAGDS


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce SparseEA-AGDS paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small configurations
  python run_paper_experiments.py --quick
  
  # Full paper reproduction (WARNING: Takes many hours!)
  python run_paper_experiments.py --full
  
  # Run specific problem
  python run_paper_experiments.py --problem SMOP1 --dimension 100 --objectives 2
  
  # Show available configurations
  python run_paper_experiments.py --list-configs
        """
    )
    
    # Experiment type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--quick', action='store_true',
                      help='Run quick test configurations (recommended for testing)')
    group.add_argument('--full', action='store_true',
                      help='Run full paper reproduction experiments')
    group.add_argument('--problem', type=str,
                      help='Run specific problem (e.g., SMOP1)')
    group.add_argument('--list-configs', action='store_true',
                      help='List available configurations')
    
    # Problem parameters (for specific problem runs)
    parser.add_argument('--dimension', type=int, default=100,
                       help='Problem dimension (default: 100)')
    parser.add_argument('--objectives', type=int, default=2,
                       help='Number of objectives (default: 2)')
    parser.add_argument('--runs', type=int, default=30,
                       help='Number of independent runs (default: 30)')
    
    # Algorithm parameters
    parser.add_argument('--population-size', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--max-generations', type=int,
                       help='Maximum generations (default: 100*D function evaluations)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze results after completion')
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    if args.list_configs:
        list_configurations()
        return
    
    if args.quick:
        print("🧪 Running Quick Test Experiments")
        print("=" * 50)
        print("This will run smaller configurations for testing the implementation.")
        print("Estimated time: 5-10 minutes")
        print()
        
        results = runner.run_paper_reproduction(quick_test=True)
        
        if args.analyze:
            analyze_results(args.output_dir, quick_test=True)
    
    elif args.full:
        print("📊 Running Full Paper Reproduction")
        print("=" * 50)
        print("This will run ALL configurations from the paper.")
        print("WARNING: This may take several hours to complete!")
        print()
        
        confirm = input("Are you sure you want to proceed? (y/N): ")
        if confirm.lower() != 'y':
            print("Experiment cancelled.")
            return
        
        results = runner.run_paper_reproduction(quick_test=False)
        
        if args.analyze:
            analyze_results(args.output_dir, quick_test=False)
    
    elif args.problem:
        print(f"🎯 Running Specific Problem: {args.problem}")
        print("=" * 50)
        
        # Create custom configuration
        config = StandardConfigs.create_custom_config(
            problem_name=args.problem,
            dimension=args.dimension,
            num_objectives=args.objectives,
            population_size=args.population_size,
            max_generations=args.max_generations,
            num_runs=args.runs
        )
        
        results = runner.run_complete_experiment(config)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"Results saved to: {Path(args.output_dir) / config.name}")


def list_configurations():
    """List available experiment configurations"""
    print("📋 Available Experiment Configurations")
    print("=" * 60)
    
    print("\n🧪 Quick Test Configurations:")
    quick_configs = StandardConfigs.get_quick_test_configs()
    for config in quick_configs:
        print(f"  {config.name:<25} - {config.problem.name} (D={config.problem.dimension}, M={config.problem.num_objectives})")
    
    print(f"\nTotal quick test experiments: {len(quick_configs)}")
    print("Estimated runtime: 5-10 minutes")
    
    print("\n📊 Full Paper Configurations:")
    full_configs = StandardConfigs.get_paper_configs()
    
    # Group by problem
    problems = {}
    for config in full_configs:
        problem_name = config.problem.name
        if problem_name not in problems:
            problems[problem_name] = []
        problems[problem_name].append(config)
    
    for problem_name, configs in problems.items():
        print(f"\n  {problem_name}:")
        for config in configs:
            print(f"    D={config.problem.dimension}, M={config.problem.num_objectives}")
    
    print(f"\nTotal full paper experiments: {len(full_configs)}")
    print("Estimated runtime: Several hours")
    
    print("\n🎯 Available Problems:")
    problems_list = ["SMOP1", "SMOP2", "SMOP3", "SMOP4", "SMOP5", "SMOP6", "SMOP7", "SMOP8"]
    for problem in problems_list:
        print(f"  {problem}")


def analyze_results(output_dir: str, quick_test: bool = False):
    """Analyze experiment results"""
    print("\n📈 Analyzing Results")
    print("=" * 30)
    
    analyzer = ResultsAnalyzer(output_dir)
    df = analyzer.generate_paper_table(quick_test=quick_test)
    
    if not df.empty:
        analyzer.print_comparison_table(df)
        
        # Save table to CSV
        table_file = Path(output_dir) / f"summary_table_{'quick' if quick_test else 'full'}.csv"
        df.to_csv(table_file, index=False)
        print(f"\n📄 Summary table saved to: {table_file}")
    else:
        print("❌ No results found for analysis.")


def run_single_test():
    """Run a single test for verification"""
    print("🔬 Running Single Test (SMOP1, D=10, M=2)")
    print("=" * 45)
    
    # Create small test configuration
    config = StandardConfigs.create_custom_config(
        problem_name="SMOP1",
        dimension=10,
        num_objectives=2,
        population_size=20,
        max_generations=50,
        num_runs=3
    )
    
    runner = ExperimentRunner()
    results = runner.run_complete_experiment(config)
    
    print(f"\n✅ Test completed!")
    print(f"Final IGD: {results['metrics'].get('igd_mean', 'N/A')}")
    print(f"Final sparsity: {results['metrics'].get('mean_sparsity_mean', 'N/A')}")


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("🚀 SparseEA-AGDS Paper Reproduction")
        print("=" * 40)
        print()
        print("This script reproduces the experiments from the SparseEA-AGDS paper.")
        print("Use --help for detailed options.")
        print()
        print("Quick start options:")
        print("  --quick      : Run quick test (5-10 minutes)")
        print("  --full       : Run full reproduction (several hours)")
        print("  --list-configs : Show all available configurations")
        print()
        print("Example: python run_paper_experiments.py --quick")
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 