"""
Configuration management for SparseEA-AGDS experiments
"""

import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ProblemConfig:
    """Configuration for optimization problems"""
    name: str
    dimension: int
    num_objectives: int
    bounds: Optional[List[List[float]]] = None
    
    def __post_init__(self):
        if self.bounds is None:
            self.bounds = [[0.0, 1.0]] * self.dimension


@dataclass
class AlgorithmConfig:
    """Configuration for SparseEA-AGDS algorithm"""
    population_size: int = 100
    max_function_evaluations: int = 10000
    Pc0: float = 1.0
    Pm0: float = 0.01  # Will be adjusted to 1/D
    eta_c: float = 20.0
    eta_m: float = 20.0
    seed: Optional[int] = None
    
    def adjust_for_dimension(self, dimension: int):
        """Adjust parameters based on problem dimension"""
        if self.Pm0 == 0.01:  # Default value
            self.Pm0 = 1.0 / dimension
        
        if self.max_function_evaluations == 10000:  # Default value
            self.max_function_evaluations = 100 * dimension


@dataclass
class ExperimentConfig:
    """Configuration for complete experiment setup"""
    name: str
    problem: ProblemConfig
    algorithm: AlgorithmConfig
    num_runs: int = 30
    output_dir: str = "results"
    save_intermediate: bool = False
    verbose: bool = True
    
    def __post_init__(self):
        # Adjust algorithm parameters for problem dimension
        self.algorithm.adjust_for_dimension(self.problem.dimension)


class ConfigManager:
    """Manager for loading and saving experiment configurations"""
    
    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Load configuration from dictionary"""
        
        # Extract problem config
        problem_dict = config_dict["problem"]
        problem_config = ProblemConfig(**problem_dict)
        
        # Extract algorithm config
        algorithm_dict = config_dict.get("algorithm", {})
        algorithm_config = AlgorithmConfig(**algorithm_dict)
        
        # Extract experiment config
        experiment_dict = {
            "name": config_dict["name"],
            "problem": problem_config,
            "algorithm": algorithm_config,
            "num_runs": config_dict.get("num_runs", 30),
            "output_dir": config_dict.get("output_dir", "results"),
            "save_intermediate": config_dict.get("save_intermediate", False),
            "verbose": config_dict.get("verbose", True)
        }
        
        return ExperimentConfig(**experiment_dict)
    
    @staticmethod
    def load_from_json(file_path: str) -> ExperimentConfig:
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return ConfigManager.load_from_dict(config_dict)
    
    @staticmethod
    def load_from_yaml(file_path: str) -> ExperimentConfig:
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ConfigManager.load_from_dict(config_dict)
    
    @staticmethod
    def save_to_json(config: ExperimentConfig, file_path: str):
        """Save configuration to JSON file"""
        config_dict = asdict(config)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def save_to_yaml(config: ExperimentConfig, file_path: str):
        """Save configuration to YAML file"""
        config_dict = asdict(config)
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class StandardConfigs:
    """Standard configurations for reproducing paper results"""
    
    @staticmethod
    def get_paper_configs() -> List[ExperimentConfig]:
        """Get configurations for reproducing paper results"""
        configs = []
        
        # Table 8 configurations: M=3,5,8,10,15 and D=100,500,1000
        problem_names = ["SMOP1", "SMOP2", "SMOP3", "SMOP4", "SMOP5", "SMOP6", "SMOP7", "SMOP8"]
        dimensions = [100, 500, 1000]
        objectives = [3, 5, 8, 10, 15]
        
        for problem_name in problem_names:
            for D in dimensions:
                for M in objectives:
                    # Skip invalid combinations
                    if problem_name in ["SMOP1", "SMOP2", "SMOP4", "SMOP6", "SMOP8"] and M > 2:
                        continue
                    if problem_name == "SMOP3" and M > 3:
                        continue
                    if problem_name == "SMOP5" and M != 5:
                        continue
                    if problem_name == "SMOP7" and M > 8:
                        continue
                    
                    config = ExperimentConfig(
                        name=f"{problem_name}_D{D}_M{M}",
                        problem=ProblemConfig(
                            name=problem_name,
                            dimension=D,
                            num_objectives=M
                        ),
                        algorithm=AlgorithmConfig(
                            population_size=100,
                            max_function_evaluations=100 * D,
                            Pc0=1.0,
                            Pm0=1.0 / D,
                            eta_c=20.0,
                            eta_m=20.0
                        ),
                        num_runs=30
                    )
                    configs.append(config)
        
        return configs
    
    @staticmethod
    def get_quick_test_configs() -> List[ExperimentConfig]:
        """Get configurations for quick testing"""
        configs = []
        
        # Smaller configurations for testing
        problem_names = ["SMOP1", "SMOP3", "SMOP5"]
        dimensions = [10, 50]
        
        for problem_name in problem_names:
            for D in dimensions:
                if problem_name == "SMOP1":
                    M = 2
                elif problem_name == "SMOP3":
                    M = 3
                else:  # SMOP5
                    M = 5
                
                config = ExperimentConfig(
                    name=f"{problem_name}_D{D}_M{M}_quick",
                    problem=ProblemConfig(
                        name=problem_name,
                        dimension=D,
                        num_objectives=M
                    ),
                    algorithm=AlgorithmConfig(
                        population_size=20,
                        max_function_evaluations=50 * D,
                        Pc0=0.9,
                        Pm0=1.0 / D,
                        eta_c=20.0,
                        eta_m=20.0
                    ),
                    num_runs=3  # Only 3 runs for quick testing
                )
                configs.append(config)
        
        return configs
    
    @staticmethod
    def create_custom_config(
        problem_name: str,
        dimension: int,
        num_objectives: int,
        population_size: int = 100,
        max_generations: int = None,
        num_runs: int = 30
    ) -> ExperimentConfig:
        """Create a custom configuration"""
        
        if max_generations is None:
            max_function_evaluations = 100 * dimension
        else:
            max_function_evaluations = max_generations * population_size
        
        return ExperimentConfig(
            name=f"{problem_name}_D{dimension}_M{num_objectives}_custom",
            problem=ProblemConfig(
                name=problem_name,
                dimension=dimension,
                num_objectives=num_objectives
            ),
            algorithm=AlgorithmConfig(
                population_size=population_size,
                max_function_evaluations=max_function_evaluations,
                Pc0=1.0,
                Pm0=1.0 / dimension,
                eta_c=20.0,
                eta_m=20.0
            ),
            num_runs=num_runs
        )


# Example configurations in different formats
def create_example_configs():
    """Create example configuration files"""
    
    # Example JSON config
    json_config = {
        "name": "SMOP1_example",
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
        "num_runs": 30,
        "output_dir": "results",
        "save_intermediate": False,
        "verbose": True
    }
    
    # Save example JSON
    with open("config/example_config.json", "w") as f:
        json.dump(json_config, f, indent=2)
    
    # Example YAML config
    yaml_config = """
name: "SMOP3_example"
problem:
  name: "SMOP3"
  dimension: 500
  num_objectives: 3
algorithm:
  population_size: 100
  max_function_evaluations: 50000
  Pc0: 1.0
  Pm0: 0.002  # 1/500
  eta_c: 20.0
  eta_m: 20.0
num_runs: 30
output_dir: "results"
save_intermediate: true
verbose: true
"""
    
    # Save example YAML
    with open("config/example_config.yaml", "w") as f:
        f.write(yaml_config)


if __name__ == "__main__":
    # Create example configurations
    create_example_configs()
    
    # Test configuration loading
    config = StandardConfigs.create_custom_config("SMOP1", 100, 2)
    print("Custom config created:")
    print(f"  Problem: {config.problem.name}")
    print(f"  Dimension: {config.problem.dimension}")
    print(f"  Max FE: {config.algorithm.max_function_evaluations}")
    print(f"  Pm0: {config.algorithm.Pm0}")
    
    # Test paper configurations
    paper_configs = StandardConfigs.get_paper_configs()
    print(f"\nPaper configurations: {len(paper_configs)} total")
    for config in paper_configs[:5]:  # Show first 5
        print(f"  {config.name}") 