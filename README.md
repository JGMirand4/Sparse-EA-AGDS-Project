# SparseEA-AGDS: Professional Research Implementation

A scientifically rigorous implementation of the **SparseEA-AGDS** (Sparse Evolutionary Algorithm with Adaptive Genetic operators and Dynamic Scoring mechanism) algorithm following software engineering best practices for research reproducibility.

## 🎯 **Overview**

This implementation provides:

- **Complete algorithm implementation** following the paper specifications
- **Automated experiment reproduction** with 30 independent runs  
- **Statistical analysis** including IGD, Wilcoxon tests, and significance testing
- **Modular architecture** with clear separation of concerns
- **Reproducible experiments** with controlled random seeds
- **Professional metrics** calculation and reporting

## 🏗️ **Project Structure**

```
sparse-ea-agds-project/
├── algorithms/           # Algorithm implementations
│   ├── __init__.py
│   └── sparse_ea_agds.py
├── problems/            # Optimization problems
│   ├── __init__.py
│   ├── base.py         # Abstract Problem base class
│   └── smop.py         # SMOP1-SMOP8 benchmark suite
├── config/             # Configuration management
│   ├── __init__.py
│   └── experiment_config.py
├── metrics/            # Quality metrics and statistical tests
│   ├── __init__.py
│   └── quality_metrics.py
├── experiments/        # Experiment automation
│   ├── __init__.py
│   └── experiment_runner.py
├── run_paper_experiments.py  # Main reproduction script
└── requirements.txt
```

## 🚀 **Quick Start**

### 1. Installation

```bash
git clone <repository>
cd sparse-ea-agds-project
pip install -r requirements.txt
```

### 2. Quick Test (5-10 minutes)

```bash
python run_paper_experiments.py --quick
```

### 3. Full Paper Reproduction (Several hours)

```bash
python run_paper_experiments.py --full
```

### 4. Run Specific Problem

```bash
python run_paper_experiments.py --problem SMOP1 --dimension 100 --objectives 2
```

## 📊 **Reproducing Paper Results**

### Standard Paper Configurations

The implementation follows **exactly** the experimental setup from the paper:

- **Population Size**: N = 100
- **Function Evaluations**: maxFE = 100 × D  
- **Crossover Probability**: Pc0 = 1.0
- **Mutation Probability**: Pm0 = 1/D
- **Distribution Index**: η = 20
- **Independent Runs**: 30 runs with different seeds
- **Problems**: SMOP1-SMOP8 with D ∈ {100, 500, 1000} and M ∈ {2, 3, 5, 8, 10, 15}

### Experiment Commands

```bash
# List all available configurations
python run_paper_experiments.py --list-configs

# Quick test (smaller dimensions for validation)
python run_paper_experiments.py --quick --analyze

# Full reproduction (exact paper setup)
python run_paper_experiments.py --full --analyze

# Custom experiment
python run_paper_experiments.py --problem SMOP3 \
    --dimension 500 --objectives 3 --runs 30
```

## 📈 **Results and Metrics**

### Automatically Calculated Metrics

- **IGD (Inverted Generational Distance)**: Primary quality metric from paper
- **GD (Generational Distance)**: Additional convergence measure  
- **Spacing**: Distribution uniformity
- **Hypervolume**: For 2-objective problems
- **Sparsity Metrics**: Mean/std number of active variables
- **Statistical Tests**: Wilcoxon rank-sum with significance symbols (+, -, =)

### Output Structure

```
results/
├── SMOP1_D100_M2/
│   ├── results.json          # Complete results
│   ├── config.json          # Experiment configuration
│   └── metrics_summary.csv  # Per-run metrics
├── paper_reproduction_quick.json  # Quick test summary
└── summary_table_quick.csv        # Paper-style table
```

## 🔬 **Algorithm Components**

### Phase 1: Initial Scoring Mechanism

- Creates D×D matrix of decision variables
- Uses identity matrix for binary masks  
- Evaluates temporary population G with single active variables
- Accumulates non-domination ranks to determine variable importance

### Phase 2: Adaptive Genetic Operators

- **Adaptive Probabilities**: Pc,i = Pc0 × Ps,i and Pm,i = Pm0 × Ps,i
- **Selection Probability**: Ps,i = (maxr - ri + 1) / maxr  
- **Operators**: Simulated Binary Crossover (SBX) and Polynomial Mutation
- **Rank-based adaptation**: Better solutions get higher genetic operator probabilities

### Phase 3: Dynamic Scoring Mechanism

- **Layer Scores**: Si,r = maxr - ri + 1
- **Weighted Scores**: SumS = Sr^T × mask  
- **Final Scores**: Sd = maxS - sumSd + 1
- **Mask Generation**: Binary tournament selection using updated variable importance

### Environmental Selection

- **Reference Points**: Das-Dennis method for structured diversity
- **Non-dominated Sorting**: Fast non-dominated sorting algorithm
- **Niche Selection**: Reference point-based diversity preservation

## 📋 **Configuration System**

### Using Configuration Files

```python
from config import ConfigManager, StandardConfigs

# Load from JSON/YAML
config = ConfigManager.load_from_json("my_config.json")

# Use standard configurations
paper_configs = StandardConfigs.get_paper_configs()
quick_configs = StandardConfigs.get_quick_test_configs()

# Create custom configuration
custom_config = StandardConfigs.create_custom_config(
    problem_name="SMOP1",
    dimension=100,
    num_objectives=2,
    population_size=100
)
```

### Example Configuration

```json
{
  "name": "SMOP1_D100_M2",
  "problem": {
    "name": "SMOP1", 
    "dimension": 100,
    "num_objectives": 2
  },
  "algorithm": {
    "population_size": 100,
    "max_function_evaluations": 10000,
    "Pc0": 1.0,
    "Pm0": 0.01,
    "eta_c": 20.0,
    "eta_m": 20.0
  },
  "num_runs": 30
}
```

## 🔧 **Advanced Usage**

### Programmatic Interface

```python
from problems import SMOP1
from algorithms import SparseEAAGDS
from config import AlgorithmConfig
from metrics import MetricsCalculator

# Create problem
problem = SMOP1(dimension=100, num_objectives=2)

# Configure algorithm
config = AlgorithmConfig(
    population_size=100,
    max_function_evaluations=10000,
    Pc0=1.0,
    Pm0=0.01
)

# Run algorithm
algorithm = SparseEAAGDS(problem, config, seed=42)
result = algorithm.run()

# Calculate metrics
metrics_calc = MetricsCalculator(problem.get_true_pareto_front())
metrics = metrics_calc.calculate_all_metrics(
    result['pareto_front'], 
    result['population']
)

print(f"IGD: {metrics['igd']:.4e}")
print(f"Sparsity: {metrics['mean_sparsity']:.2f}")
```

### Batch Experiments

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(output_dir="my_results")

# Run specific configuration
config = StandardConfigs.create_custom_config("SMOP1", 100, 2)
results = runner.run_complete_experiment(config)

# Compare algorithms
baseline_results = runner.load_results("baseline_experiment")
comparison = runner.compare_with_baseline(
    results, baseline_results, metric_name='igd'
)
print(f"Statistical significance: {comparison['symbol']}")
```

## 📊 **Expected Results**

Based on the paper, you should expect:

- **Sparsity**: 1-5 active variables out of 100+ total variables
- **IGD Values**: Problem-dependent, typically in range 1e-2 to 1e-4
- **Convergence**: Steady improvement over generations
- **Statistical Significance**: When comparing with baseline algorithms

### Sample Output

```
📊 SparseEA-AGDS Results Summary
================================================================================
Problem  D     M   IGD (mean±std)       Sparsity        Runtime(s)
--------------------------------------------------------------------------------
SMOP1    100   2   1.23e-03±2.45e-04   2.3±0.8         12.5
SMOP3    500   3   2.56e-03±3.21e-04   3.1±1.2         67.2
SMOP5    1000  5   4.78e-03±5.43e-04   4.2±1.5         198.7
```

## 🧪 **Testing and Validation**

### Verification Steps

1. **Algorithm Logic**: Verify equations 5-10 are correctly implemented
2. **Parameter Settings**: Confirm exact paper parameter values
3. **Random Seeds**: Ensure reproducible results with same seeds
4. **Metric Calculations**: Validate IGD computation against reference
5. **Statistical Tests**: Verify Wilcoxon test implementation

### Quick Validation

```bash
# Test single run
python -c "
from problems import SMOP1
from algorithms import SparseEAAGDS  
from config import AlgorithmConfig

problem = SMOP1(10, 2)
config = AlgorithmConfig(population_size=20, max_function_evaluations=500)
algorithm = SparseEAAGDS(problem, config, seed=42)
result = algorithm.run()
print(f'Final sparsity: {sum(ind.mask.sum() for ind in result[\"population\"]) / len(result[\"population\"]):.1f}')
"
```

## 📖 **Citation**

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{sparseea_agds_2024,
  title={SparseEA-AGDS: Sparse Evolutionary Algorithm with Adaptive Genetic operators and Dynamic Scoring mechanism},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 **Contributing**

1. **Code Quality**: Follow PEP 8 style guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Reproducibility**: Ensure deterministic behavior with seeds

## 📋 **Troubleshooting**

### Common Issues

**Import Errors**: Ensure all packages are installed with `pip install -r requirements.txt`

**Memory Issues**: For large problems (D=1000+), consider reducing population size

**Slow Performance**: Use `--quick` flag for testing, full reproduction takes hours

**Statistical Tests**: Ensure sufficient runs (≥30) for reliable statistical analysis

### Performance Tips

- Use smaller dimensions for algorithm development/testing
- Run experiments in parallel on multiple cores
- Monitor memory usage for high-dimensional problems
- Save intermediate results with `save_intermediate=True`

## 🏆 **Features for Reproducible Research**

✅ **Exact Paper Implementation**: All equations and parameters match the paper  
✅ **Controlled Randomness**: Reproducible results with seeds  
✅ **Automated Statistics**: Mean, std, and statistical significance testing  
✅ **Professional Metrics**: IGD, GD, spacing, hypervolume, sparsity analysis  
✅ **Experiment Automation**: 30-run execution with progress tracking  
✅ **Result Storage**: JSON, CSV export with complete experiment metadata  
✅ **Configuration Management**: Parameter files for different experimental setups  
✅ **Modular Design**: Easy to extend with new problems or algorithms  

---

**📧 Contact**: For questions about the implementation or paper reproduction, please open an issue or contact the authors. 