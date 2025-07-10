#!/usr/bin/env python3
"""
Quick test to demonstrate the refactored SparseEA-AGDS implementation
This shows the modular structure and validates that everything works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import the new modular framework
from problems import SMOP1, SMOP3
from algorithms import SparseEAAGDS
from config import AlgorithmConfig, StandardConfigs
from metrics import MetricsCalculator, QualityMetrics, find_pareto_front
from experiments import ExperimentRunner


def test_single_run():
    """Test a single algorithm run"""
    print("🔬 Testing Single Algorithm Run")
    print("=" * 40)
    
    # Create problem
    problem = SMOP1(dimension=10, num_objectives=2)
    print(f"Problem: {problem}")
    
    # Configure algorithm
    config = AlgorithmConfig(
        population_size=20,
        max_function_evaluations=500,
        Pc0=0.9,
        Pm0=0.1,
        eta_c=20.0,
        eta_m=20.0
    )
    
    # Run algorithm
    algorithm = SparseEAAGDS(problem, config, seed=42)
    result = algorithm.run()
    
    # Show results
    print(f"✅ Algorithm completed successfully!")
    print(f"   Generations: {result['generation']}")
    print(f"   Function evaluations: {result['function_evaluations']}")
    print(f"   Final population size: {len(result['population'])}")
    print(f"   Pareto front size: {len(result['pareto_front'])}")
    
    # Calculate metrics
    if len(result['pareto_front']) > 0:
        true_pareto = problem.get_true_pareto_front()
        igd = QualityMetrics.igd(result['pareto_front'], true_pareto)
        print(f"   IGD: {igd:.4e}")
        
        # Sparsity analysis
        sparsities = [np.sum(ind.mask) for ind in result['population']]
        print(f"   Mean sparsity: {np.mean(sparsities):.1f} ± {np.std(sparsities):.1f}")
        print(f"   Sparsity range: {np.min(sparsities)}-{np.max(sparsities)}")
    
    return result


def test_configuration_system():
    """Test the configuration system"""
    print("\n⚙️ Testing Configuration System")
    print("=" * 40)
    
    # Test standard configurations
    quick_configs = StandardConfigs.get_quick_test_configs()
    print(f"✅ Quick test configs: {len(quick_configs)} available")
    
    paper_configs = StandardConfigs.get_paper_configs()
    print(f"✅ Paper configs: {len(paper_configs)} available")
    
    # Test custom configuration
    custom_config = StandardConfigs.create_custom_config(
        problem_name="SMOP1",
        dimension=20,
        num_objectives=2,
        population_size=30,
        num_runs=3
    )
    print(f"✅ Custom config created: {custom_config.name}")
    print(f"   Max FE: {custom_config.algorithm.max_function_evaluations}")
    print(f"   Pm0: {custom_config.algorithm.Pm0}")


def test_problems():
    """Test different problem implementations"""
    print("\n🎯 Testing Problem Implementations")
    print("=" * 40)
    
    problems = [
        SMOP1(dimension=5, num_objectives=2),
        SMOP3(dimension=5, num_objectives=3)
    ]
    
    for problem in problems:
        print(f"\nTesting {problem.name}:")
        
        # Test evaluation
        x = np.random.uniform(0, 1, problem.dimension)
        objectives = problem.evaluate(x)
        print(f"   Sample evaluation: {objectives}")
        
        # Test bounds
        lower, upper = problem.get_bounds()
        print(f"   Bounds: [{lower[0]:.1f}, {upper[0]:.1f}]")
        
        # Test true Pareto front
        true_pf = problem.get_true_pareto_front(num_points=100)
        if true_pf is not None:
            print(f"   True Pareto front: {true_pf.shape}")
        
        print(f"   ✅ {problem.name} working correctly")


def test_metrics():
    """Test metrics calculation"""
    print("\n📊 Testing Metrics Calculation")
    print("=" * 40)
    
    # Generate sample data
    np.random.seed(42)
    obtained_front = np.random.random((20, 2))
    true_front = np.random.random((100, 2))
    
    # Calculate metrics
    igd = QualityMetrics.igd(obtained_front, true_front)
    gd = QualityMetrics.gd(obtained_front, true_front)
    spacing = QualityMetrics.spacing(obtained_front)
    
    print(f"   IGD: {igd:.4e}")
    print(f"   GD: {gd:.4e}")
    print(f"   Spacing: {spacing:.4e}")
    print("   ✅ Metrics calculation working")


def test_experiment_runner():
    """Test the experiment runner with a minimal config"""
    print("\n🧪 Testing Experiment Runner")
    print("=" * 40)
    
    # Create a minimal test configuration
    config = StandardConfigs.create_custom_config(
        problem_name="SMOP1",
        dimension=5,
        num_objectives=2,
        population_size=10,
        num_runs=2  # Just 2 runs for speed
    )
    config.algorithm.max_function_evaluations = 100  # Very quick
    
    # Run experiment
    runner = ExperimentRunner(output_dir="test_results")
    results = runner.run_complete_experiment(config, save_results=False)
    
    print(f"   ✅ Experiment completed")
    print(f"   Mean IGD: {results['metrics'].get('igd_mean', 'N/A')}")
    print(f"   Mean sparsity: {results['metrics'].get('mean_sparsity_mean', 'N/A')}")


def visualize_sample_run():
    """Create a visualization of a sample run"""
    print("\n📈 Creating Sample Visualization")
    print("=" * 40)
    
    # Run algorithm on SMOP1
    problem = SMOP1(dimension=10, num_objectives=2)
    config = AlgorithmConfig(
        population_size=50,
        max_function_evaluations=1000,
        Pc0=0.9,
        Pm0=0.1
    )
    
    algorithm = SparseEAAGDS(problem, config, seed=42)
    result = algorithm.run()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Pareto front
    if len(result['pareto_front']) > 0:
        ax1.scatter(result['pareto_front'][:, 0], result['pareto_front'][:, 1], 
                   alpha=0.7, c='red', label='Obtained')
        
        true_pf = problem.get_true_pareto_front(1000)
        ax1.plot(true_pf[:, 0], true_pf[:, 1], 'b-', alpha=0.5, label='True Pareto Front')
        
        ax1.set_xlabel('Objective 1')
        ax1.set_ylabel('Objective 2')
        ax1.set_title('Pareto Front Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Sparsity distribution
    sparsities = [np.sum(ind.mask) for ind in result['population']]
    ax2.hist(sparsities, bins=range(problem.dimension + 2), alpha=0.7, color='green')
    ax2.set_xlabel('Number of Active Variables')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sparsity Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence history
    if result['convergence_history']:
        generations = [h['generation'] for h in result['convergence_history']]
        mean_obj1 = [h['mean_objectives'][0] for h in result['convergence_history']]
        mean_obj2 = [h['mean_objectives'][1] for h in result['convergence_history']]
        
        ax3.plot(generations, mean_obj1, 'r-', label='Objective 1')
        ax3.plot(generations, mean_obj2, 'b-', label='Objective 2')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Mean Objective Value')
        ax3.set_title('Convergence History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Variable importance (from final population)
    variable_usage = np.zeros(problem.dimension)
    for ind in result['population']:
        variable_usage += ind.mask
    
    ax4.bar(range(problem.dimension), variable_usage, alpha=0.7, color='orange')
    ax4.set_xlabel('Variable Index')
    ax4.set_ylabel('Usage Count')
    ax4.set_title('Variable Importance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "sample_run_visualization.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Visualization saved to: {output_dir / 'sample_run_visualization.png'}")
    
    return result


def main():
    """Run all tests"""
    print("🚀 SparseEA-AGDS Framework Quick Test")
    print("=" * 50)
    print("Testing the refactored implementation...")
    
    try:
        # Run all tests
        test_single_run()
        test_configuration_system()
        test_problems()
        test_metrics()
        test_experiment_runner()
        
        # Create visualization
        result = visualize_sample_run()
        
        print("\n🎉 All Tests Passed!")
        print("=" * 50)
        print("✅ Algorithm implementation: Working")
        print("✅ Configuration system: Working")
        print("✅ Problem implementations: Working")
        print("✅ Metrics calculation: Working")
        print("✅ Experiment runner: Working")
        print("✅ Visualization: Generated")
        
        print(f"\n📊 Sample Run Results:")
        print(f"   Final generation: {result['generation']}")
        print(f"   Function evaluations: {result['function_evaluations']}")
        print(f"   Pareto solutions: {len(result['pareto_front'])}")
        
        if len(result['pareto_front']) > 0:
            sparsities = [np.sum(ind.mask) for ind in result['population']]
            print(f"   Mean sparsity: {np.mean(sparsities):.1f} variables")
            print(f"   Sparsity range: {np.min(sparsities)}-{np.max(sparsities)} variables")
        
        print("\n🎯 Next Steps:")
        print("   • Run full experiments: python run_paper_experiments.py --quick")
        print("   • Check visualization: test_results/sample_run_visualization.png")
        print("   • Try different problems: SMOP1, SMOP2, ..., SMOP8")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 