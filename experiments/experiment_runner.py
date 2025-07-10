"""
Experiment runner for reproducing SparseEA-AGDS paper results
Automates the execution of 30 independent runs and statistical analysis
"""

import numpy as np
import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import framework components
import sys
sys.path.append('..')
from problems import create_smop_problem
from algorithms import SparseEAAGDS
from config import ExperimentConfig, StandardConfigs
from metrics import QualityMetrics, StatisticalTests, MetricsCalculator


class ExperimentRunner:
    """
    Automated experiment runner for SparseEA-AGDS
    
    Features:
    - Automated 30-run execution
    - Statistical analysis
    - Result saving and loading
    - Progress tracking
    - Reproducible experiments
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_single_experiment(self, config: ExperimentConfig, 
                             run_id: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a single experimental run
        
        Args:
            config: Experiment configuration
            run_id: Run identifier (0-29 for paper reproduction)
            seed: Random seed (if None, uses run_id)
            
        Returns:
            Dictionary with run results
        """
        if seed is None:
            seed = run_id
        
        # Create problem instance
        problem = create_smop_problem(
            config.problem.name,
            config.problem.dimension,
            config.problem.num_objectives
        )
        
        # Set up metrics calculator
        true_pareto_front = problem.get_true_pareto_front()
        metrics_calculator = MetricsCalculator(true_pareto_front)
        
        # Create and run algorithm
        algorithm = SparseEAAGDS(problem, config.algorithm, seed=seed)
        
        start_time = time.time()
        result = algorithm.run()
        end_time = time.time()
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_all_metrics(
            result['pareto_front'],
            result['population'],
            reference_point=np.ones(problem.num_objectives) * 2.0  # Default reference point
        )
        
        # Prepare result dictionary
        run_result = {
            'run_id': run_id,
            'seed': seed,
            'config_name': config.name,
            'problem': config.problem.name,
            'dimension': config.problem.dimension,
            'num_objectives': config.problem.num_objectives,
            'runtime_seconds': end_time - start_time,
            'final_generation': result['generation'],
            'function_evaluations': result['function_evaluations'],
            'metrics': metrics,
            'convergence_history': result['convergence_history']
        }
        
        # Add basic statistics
        run_result['num_pareto_solutions'] = len(result['pareto_front'])
        if len(result['pareto_front']) > 0:
            run_result['pareto_front_mean'] = np.mean(result['pareto_front'], axis=0).tolist()
            run_result['pareto_front_std'] = np.std(result['pareto_front'], axis=0).tolist()
        
        return run_result
    
    def run_complete_experiment(self, config: ExperimentConfig, 
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Execute complete experiment with multiple runs
        
        Args:
            config: Experiment configuration
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with aggregated results
        """
        print(f"🚀 Starting experiment: {config.name}")
        print(f"   Problem: {config.problem.name}")
        print(f"   Dimension: {config.problem.dimension}")
        print(f"   Objectives: {config.problem.num_objectives}")
        print(f"   Runs: {config.num_runs}")
        
        all_run_results = []
        start_time = time.time()
        
        # Execute all runs
        for run_id in range(config.num_runs):
            if config.verbose:
                print(f"   Running {run_id + 1}/{config.num_runs}...", end=" ")
            
            run_result = self.run_single_experiment(config, run_id)
            all_run_results.append(run_result)
            
            if config.verbose:
                igd = run_result['metrics'].get('igd', np.inf)
                sparsity = run_result['metrics'].get('mean_sparsity', 0)
                print(f"IGD: {igd:.4e}, Sparsity: {sparsity:.1f}")
        
        total_time = time.time() - start_time
        
        # Calculate aggregate statistics
        aggregate_results = self._aggregate_results(all_run_results, config)
        aggregate_results['total_runtime_seconds'] = total_time
        aggregate_results['all_runs'] = all_run_results
        
        # Save results if requested
        if save_results:
            self._save_results(aggregate_results, config)
        
        print(f"✅ Experiment completed in {total_time:.2f} seconds")
        self._print_summary(aggregate_results)
        
        return aggregate_results
    
    def run_paper_reproduction(self, quick_test: bool = False) -> Dict[str, Any]:
        """
        Run complete paper reproduction experiments
        
        Args:
            quick_test: If True, run smaller test configurations
            
        Returns:
            Dictionary with all experiment results
        """
        if quick_test:
            configs = StandardConfigs.get_quick_test_configs()
            print("🧪 Running quick test configurations...")
        else:
            configs = StandardConfigs.get_paper_configs()
            print("📊 Running full paper reproduction...")
        
        print(f"Total experiments to run: {len(configs)}")
        
        all_results = {}
        overall_start_time = time.time()
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] ", end="")
            results = self.run_complete_experiment(config)
            all_results[config.name] = results
        
        total_time = time.time() - overall_start_time
        
        # Save summary
        summary_results = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(configs),
            'total_runtime_seconds': total_time,
            'quick_test': quick_test,
            'experiments': all_results
        }
        
        summary_file = self.output_dir / f"paper_reproduction_{'quick' if quick_test else 'full'}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        print(f"\n🎉 All experiments completed in {total_time/3600:.2f} hours")
        print(f"📄 Results saved to: {summary_file}")
        
        return summary_results
    
    def _aggregate_results(self, run_results: List[Dict[str, Any]], 
                          config: ExperimentConfig) -> Dict[str, Any]:
        """Aggregate statistics from multiple runs"""
        
        # Extract metric values
        metric_names = ['igd', 'gd', 'spacing', 'hypervolume', 'mean_sparsity', 'sparsity_ratio']
        aggregated_metrics = {}
        
        for metric_name in metric_names:
            values = []
            for run in run_results:
                if metric_name in run['metrics']:
                    value = run['metrics'][metric_name]
                    if np.isfinite(value):
                        values.append(value)
            
            if values:
                aggregated_metrics[f'{metric_name}_mean'] = float(np.mean(values))
                aggregated_metrics[f'{metric_name}_std'] = float(np.std(values))
                aggregated_metrics[f'{metric_name}_median'] = float(np.median(values))
                aggregated_metrics[f'{metric_name}_min'] = float(np.min(values))
                aggregated_metrics[f'{metric_name}_max'] = float(np.max(values))
                aggregated_metrics[f'{metric_name}_values'] = values
        
        # Calculate other statistics
        runtimes = [run['runtime_seconds'] for run in run_results]
        function_evals = [run['function_evaluations'] for run in run_results]
        generations = [run['final_generation'] for run in run_results]
        
        return {
            'config_name': config.name,
            'problem': config.problem.name,
            'dimension': config.problem.dimension,
            'num_objectives': config.problem.num_objectives,
            'num_runs': len(run_results),
            'metrics': aggregated_metrics,
            'runtime_mean': float(np.mean(runtimes)),
            'runtime_std': float(np.std(runtimes)),
            'function_evaluations_mean': float(np.mean(function_evals)),
            'function_evaluations_std': float(np.std(function_evals)),
            'generations_mean': float(np.mean(generations)),
            'generations_std': float(np.std(generations)),
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self, results: Dict[str, Any], config: ExperimentConfig):
        """Save experiment results to disk"""
        
        # Create experiment directory
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(exist_ok=True)
        
        # Save main results
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save configuration
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            import json
            from dataclasses import asdict
            json.dump(asdict(config), f, indent=2)
        
        # Save metrics summary as CSV
        if 'all_runs' in results:
            df_data = []
            for run in results['all_runs']:
                row = {
                    'run_id': run['run_id'],
                    'igd': run['metrics'].get('igd', np.nan),
                    'gd': run['metrics'].get('gd', np.nan),
                    'spacing': run['metrics'].get('spacing', np.nan),
                    'hypervolume': run['metrics'].get('hypervolume', np.nan),
                    'mean_sparsity': run['metrics'].get('mean_sparsity', np.nan),
                    'sparsity_ratio': run['metrics'].get('sparsity_ratio', np.nan),
                    'num_solutions': run['metrics'].get('num_solutions', 0),
                    'runtime': run['runtime_seconds'],
                    'function_evaluations': run['function_evaluations']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = exp_dir / "metrics_summary.csv"
            df.to_csv(csv_file, index=False)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print experiment summary"""
        metrics = results['metrics']
        
        print("📊 Experiment Summary:")
        print(f"   Problem: {results['problem']} (D={results['dimension']}, M={results['num_objectives']})")
        print(f"   Runs: {results['num_runs']}")
        
        if 'igd_mean' in metrics:
            print(f"   IGD: {metrics['igd_mean']:.4e} ± {metrics['igd_std']:.4e}")
        
        if 'mean_sparsity_mean' in metrics:
            print(f"   Sparsity: {metrics['mean_sparsity_mean']:.2f} ± {metrics['mean_sparsity_std']:.2f}")
        
        print(f"   Runtime: {results['runtime_mean']:.2f}s ± {results['runtime_std']:.2f}s")
        print(f"   Function Evaluations: {results['function_evaluations_mean']:.0f} ± {results['function_evaluations_std']:.0f}")
    
    def load_results(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load saved experiment results"""
        results_file = self.output_dir / config_name / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def compare_with_baseline(self, results1: Dict[str, Any], 
                             results2: Dict[str, Any], 
                             metric_name: str = 'igd') -> Dict[str, Any]:
        """Compare two sets of results using statistical tests"""
        
        values1 = results1['metrics'].get(f'{metric_name}_values', [])
        values2 = results2['metrics'].get(f'{metric_name}_values', [])
        
        if not values1 or not values2:
            return {'error': 'Insufficient data for comparison'}
        
        comparison = StatisticalTests.compare_algorithms(
            [{'igd': v} for v in values1],
            [{'igd': v} for v in values2],
            metric_name
        )
        
        return {
            'algorithm1': results1['config_name'],
            'algorithm2': results2['config_name'],
            'metric': metric_name,
            'mean1': comparison['mean1'],
            'std1': comparison['std1'],
            'mean2': comparison['mean2'],
            'std2': comparison['std2'],
            'p_value': comparison['test_result']['p_value'],
            'significant': comparison['test_result']['significant'],
            'symbol': comparison['test_result']['symbol']
        }


class ResultsAnalyzer:
    """Analyzer for experiment results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
    
    def generate_paper_table(self, quick_test: bool = False) -> pd.DataFrame:
        """Generate a table similar to those in the paper"""
        
        # Load paper reproduction results
        summary_file = self.results_dir / f"paper_reproduction_{'quick' if quick_test else 'full'}.json"
        
        if not summary_file.exists():
            print(f"Results file not found: {summary_file}")
            return pd.DataFrame()
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract data for table
        table_data = []
        for exp_name, results in data['experiments'].items():
            if 'metrics' in results:
                metrics = results['metrics']
                row = {
                    'Problem': results['problem'],
                    'D': results['dimension'],
                    'M': results['num_objectives'],
                    'IGD_mean': metrics.get('igd_mean', np.nan),
                    'IGD_std': metrics.get('igd_std', np.nan),
                    'Sparsity_mean': metrics.get('mean_sparsity_mean', np.nan),
                    'Sparsity_std': metrics.get('mean_sparsity_std', np.nan),
                    'Runtime_mean': results.get('runtime_mean', np.nan)
                }
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values(['Problem', 'D', 'M'])
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """Print results in paper format"""
        print("\n📊 SparseEA-AGDS Results Summary")
        print("=" * 80)
        print(f"{'Problem':<8} {'D':<5} {'M':<3} {'IGD (mean±std)':<20} {'Sparsity':<15} {'Runtime(s)':<10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            igd_str = f"{row['IGD_mean']:.2e}±{row['IGD_std']:.2e}" if not np.isnan(row['IGD_mean']) else "N/A"
            sparsity_str = f"{row['Sparsity_mean']:.1f}±{row['Sparsity_std']:.1f}" if not np.isnan(row['Sparsity_mean']) else "N/A"
            runtime_str = f"{row['Runtime_mean']:.1f}" if not np.isnan(row['Runtime_mean']) else "N/A"
            
            print(f"{row['Problem']:<8} {row['D']:<5} {row['M']:<3} {igd_str:<20} {sparsity_str:<15} {runtime_str:<10}")


if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner()
    
    # Run quick test for demonstration
    print("Running quick test experiments...")
    results = runner.run_paper_reproduction(quick_test=True)
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    df = analyzer.generate_paper_table(quick_test=True)
    analyzer.print_comparison_table(df)
    
    print("\n✨ To run full paper reproduction, use:")
    print("   runner.run_paper_reproduction(quick_test=False)") 