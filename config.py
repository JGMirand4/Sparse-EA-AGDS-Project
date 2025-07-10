"""
Sistema de Configuração para SparseEA-AGDS
==========================================

Gerencia parâmetros do algoritmo e configurações de experimentos
para garantir reprodutibilidade e facilitar a execução de testes.
"""

import json
import yaml
from typing import Dict, Any, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AlgorithmConfig:
    """Configuração específica do algoritmo SparseEA-AGDS"""
    
    # Parâmetros principais
    population_size: int = 100
    max_function_evaluations: int = 10000
    
    # Probabilidades base
    Pc0: float = 1.0  # Probabilidade base de crossover
    Pm0: float = 0.01  # Probabilidade base de mutação (será ajustada para 1/D)
    
    # Parâmetros dos operadores
    eta_c: float = 20.0  # Parâmetro do SBX
    eta_m: float = 20.0  # Parâmetro da mutação polinomial
    
    # Parâmetros específicos do SparseEA-AGDS
    reference_points_param: int = 12  # Parâmetro p para geração de pontos de referência
    initial_scoring_runs: int = 10  # Número de execuções para pontuação inicial
    
    # Controle de aleatoriedade
    random_seed: int = 42
    
    def adjust_for_problem_dimension(self, D: int):
        """Ajusta parâmetros baseados na dimensão do problema"""
        # Ajusta probabilidade de mutação: Pm0 = 1/D
        self.Pm0 = 1.0 / D
        
        # Ajusta número máximo de avaliações: maxFE = 100 * D
        self.max_function_evaluations = 100 * D


@dataclass
class ProblemConfig:
    """Configuração do problema a ser resolvido"""
    
    name: str = "SMOP1"  # Nome do problema
    dimension: int = 100  # Dimensão do problema (D)
    num_objectives: int = 2  # Número de objetivos (M)
    
    @property
    def problem_id(self) -> str:
        """Identificador único do problema"""
        return f"{self.name}_D{self.dimension}_M{self.num_objectives}"


@dataclass
class ExperimentConfig:
    """Configuração de um experimento completo"""
    
    # Configurações do problema e algoritmo
    problem: ProblemConfig
    algorithm: AlgorithmConfig
    
    # Parâmetros do experimento
    num_runs: int = 30  # Número de execuções independentes
    random_seeds: List[int] = None  # Seeds para cada execução
    
    # Configurações de saída
    save_results: bool = True
    results_dir: str = "results"
    save_populations: bool = False  # Salvar populações finais
    
    # Configurações de logging
    verbose: bool = True
    log_frequency: int = 100  # Frequência de logging (por avaliações)
    
    def __post_init__(self):
        """Inicialização pós-criação"""
        if self.random_seeds is None:
            # Gera seeds determinísticos baseados no seed base
            base_seed = self.algorithm.random_seed
            self.random_seeds = [base_seed + i for i in range(self.num_runs)]
        
        # Ajusta algoritmo para dimensão do problema
        self.algorithm.adjust_for_problem_dimension(self.problem.dimension)
    
    @property
    def experiment_id(self) -> str:
        """Identificador único do experimento"""
        return f"{self.problem.problem_id}_N{self.algorithm.population_size}_R{self.num_runs}"


class ConfigManager:
    """Gerenciador de configurações com suporte a arquivos JSON/YAML"""
    
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Carrega configuração de um arquivo JSON ou YAML
        
        Args:
            config_path: Caminho para o arquivo de configuração
            
        Returns:
            Configuração do experimento
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
        # Determina formato do arquivo
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {config_path.suffix}")
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_to_file(config: ExperimentConfig, config_path: Union[str, Path]):
        """
        Salva configuração em um arquivo JSON ou YAML
        
        Args:
            config: Configuração a ser salva
            config_path: Caminho onde salvar
        """
        config_path = Path(config_path)
        config_dict = ConfigManager._config_to_dict(config)
        
        # Cria diretório se não existir
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva baseado na extensão
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {config_path.suffix}")
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Converte dicionário para ExperimentConfig"""
        
        # Extrai configurações
        problem_dict = config_dict.get('problem', {})
        algorithm_dict = config_dict.get('algorithm', {})
        experiment_dict = {k: v for k, v in config_dict.items() 
                          if k not in ['problem', 'algorithm']}
        
        # Cria objetos de configuração
        problem_config = ProblemConfig(**problem_dict)
        algorithm_config = AlgorithmConfig(**algorithm_dict)
        
        return ExperimentConfig(
            problem=problem_config,
            algorithm=algorithm_config,
            **experiment_dict
        )
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Converte ExperimentConfig para dicionário"""
        result = asdict(config)
        return result
    
    @staticmethod
    def create_default_config(problem_name: str = "SMOP1", 
                            D: int = 100, 
                            M: int = 2) -> ExperimentConfig:
        """
        Cria configuração padrão para um problema
        
        Args:
            problem_name: Nome do problema
            D: Dimensão do problema
            M: Número de objetivos
            
        Returns:
            Configuração padrão
        """
        problem_config = ProblemConfig(
            name=problem_name,
            dimension=D,
            num_objectives=M
        )
        
        algorithm_config = AlgorithmConfig()
        
        return ExperimentConfig(
            problem=problem_config,
            algorithm=algorithm_config
        )


def create_benchmark_configs() -> List[ExperimentConfig]:
    """
    Cria configurações para replicar os experimentos do artigo
    
    Returns:
        Lista de configurações de experimento
    """
    configs = []
    
    # Configurações baseadas nas tabelas do artigo
    problem_configs = [
        # Problemas bi-objetivo
        ("SMOP1", 100, 2), ("SMOP1", 500, 2), ("SMOP1", 1000, 2),
        ("SMOP2", 100, 2), ("SMOP2", 500, 2), ("SMOP2", 1000, 2),
        ("SMOP4", 100, 2), ("SMOP4", 500, 2), ("SMOP4", 1000, 2),
        ("SMOP5", 100, 2), ("SMOP5", 500, 2), ("SMOP5", 1000, 2),
        ("SMOP6", 100, 2), ("SMOP6", 500, 2), ("SMOP6", 1000, 2),
        ("SMOP7", 100, 2), ("SMOP7", 500, 2), ("SMOP7", 1000, 2),
        
        # Problemas multi-objetivo (baseados na Tabela 8 do artigo)
        ("SMOP3", 100, 3), ("SMOP3", 500, 3), ("SMOP3", 1000, 3),
        ("SMOP3", 100, 5), ("SMOP3", 500, 5), ("SMOP3", 1000, 5),
        ("SMOP3", 100, 8), ("SMOP3", 500, 8), ("SMOP3", 1000, 8),
        ("SMOP3", 100, 10), ("SMOP3", 500, 10), ("SMOP3", 1000, 10),
        ("SMOP3", 100, 15), ("SMOP3", 500, 15), ("SMOP3", 1000, 15),
        
        ("SMOP8", 100, 3), ("SMOP8", 500, 3), ("SMOP8", 1000, 3),
        ("SMOP8", 100, 5), ("SMOP8", 500, 5), ("SMOP8", 1000, 5),
        ("SMOP8", 100, 8), ("SMOP8", 500, 8), ("SMOP8", 1000, 8),
        ("SMOP8", 100, 10), ("SMOP8", 500, 10), ("SMOP8", 1000, 10),
        ("SMOP8", 100, 15), ("SMOP8", 500, 15), ("SMOP8", 1000, 15),
    ]
    
    # Cria configuração para cada problema
    for problem_name, D, M in problem_configs:
        config = ConfigManager.create_default_config(problem_name, D, M)
        configs.append(config)
    
    return configs


def create_test_config() -> ExperimentConfig:
    """
    Cria configuração para testes rápidos
    
    Returns:
        Configuração de teste
    """
    problem_config = ProblemConfig(
        name="SMOP1",
        dimension=10,
        num_objectives=2
    )
    
    algorithm_config = AlgorithmConfig(
        population_size=20,
        max_function_evaluations=1000
    )
    
    return ExperimentConfig(
        problem=problem_config,
        algorithm=algorithm_config,
        num_runs=3,
        verbose=True
    )


# Exemplo de uso
if __name__ == "__main__":
    # Cria configuração padrão
    config = ConfigManager.create_default_config("SMOP1", 100, 2)
    
    # Salva em arquivo
    ConfigManager.save_to_file(config, "configs/default_config.json")
    
    # Carrega de arquivo
    loaded_config = ConfigManager.load_from_file("configs/default_config.json")
    
    print("Configuração criada e carregada com sucesso!")
    print(f"Problema: {loaded_config.problem.problem_id}")
    print(f"Algoritmo: N={loaded_config.algorithm.population_size}, maxFE={loaded_config.algorithm.max_function_evaluations}")
    print(f"Experimento: {loaded_config.num_runs} execuções") 