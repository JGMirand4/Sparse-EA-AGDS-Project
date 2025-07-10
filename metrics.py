"""
Métricas de Avaliação para SparseEA-AGDS
=======================================

Implementa métricas para avaliar a qualidade das soluções e comparar algoritmos,
incluindo IGD, análise de esparsidade e testes estatísticos.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, mannwhitneyu
import warnings
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Resultado de uma métrica de avaliação"""
    
    name: str
    value: float
    std_dev: float = 0.0
    best_value: float = None
    worst_value: float = None
    median_value: float = None
    
    def __post_init__(self):
        if self.best_value is None:
            self.best_value = self.value
        if self.worst_value is None:
            self.worst_value = self.value
        if self.median_value is None:
            self.median_value = self.value


@dataclass
class ExperimentResults:
    """Resultados completos de um experimento"""
    
    problem_name: str
    algorithm_name: str
    dimension: int
    num_objectives: int
    num_runs: int
    
    # Métricas de qualidade
    igd_values: List[float]
    hypervolume_values: List[float] = None
    
    # Métricas de esparsidade
    sparsity_values: List[float] = None
    sparsity_percentage: List[float] = None
    
    # Métricas computacionais
    function_evaluations: List[int] = None
    execution_times: List[float] = None
    
    def get_metric_summary(self, metric_name: str) -> MetricResult:
        """
        Retorna resumo estatístico de uma métrica
        
        Args:
            metric_name: Nome da métrica ('igd', 'hypervolume', 'sparsity', etc.)
            
        Returns:
            Resumo da métrica
        """
        if metric_name == 'igd':
            values = self.igd_values
        elif metric_name == 'hypervolume':
            values = self.hypervolume_values
        elif metric_name == 'sparsity':
            values = self.sparsity_values
        elif metric_name == 'sparsity_percentage':
            values = self.sparsity_percentage
        elif metric_name == 'function_evaluations':
            values = self.function_evaluations
        elif metric_name == 'execution_times':
            values = self.execution_times
        else:
            raise ValueError(f"Métrica desconhecida: {metric_name}")
        
        if values is None or len(values) == 0:
            return MetricResult(name=metric_name, value=np.nan)
        
        values = np.array(values)
        
        return MetricResult(
            name=metric_name,
            value=np.mean(values),
            std_dev=np.std(values),
            best_value=np.min(values) if metric_name in ['igd'] else np.max(values),
            worst_value=np.max(values) if metric_name in ['igd'] else np.min(values),
            median_value=np.median(values)
        )


class QualityMetrics:
    """Implementa métricas de qualidade para problemas multi-objetivo"""
    
    @staticmethod
    def calculate_igd(obtained_front: np.ndarray, 
                     true_front: np.ndarray, 
                     p: int = 2) -> float:
        """
        Calcula o Inverted Generational Distance (IGD)
        
        Args:
            obtained_front: Frente de Pareto obtida pelo algoritmo
            true_front: Frente de Pareto verdadeira
            p: Parâmetro da norma Minkowski (padrão: 2 para distância euclidiana)
            
        Returns:
            Valor do IGD
        """
        if len(obtained_front) == 0:
            return np.inf
        
        if len(true_front) == 0:
            return np.nan
        
        # Calcula distâncias de cada ponto da frente verdadeira para a frente obtida
        distances = cdist(true_front, obtained_front, metric='minkowski', p=p)
        
        # Pega a menor distância de cada ponto da frente verdadeira
        min_distances = np.min(distances, axis=1)
        
        # IGD é a média das menores distâncias
        igd = np.mean(min_distances)
        
        return igd
    
    @staticmethod
    def calculate_gd(obtained_front: np.ndarray, 
                    true_front: np.ndarray, 
                    p: int = 2) -> float:
        """
        Calcula o Generational Distance (GD)
        
        Args:
            obtained_front: Frente de Pareto obtida pelo algoritmo
            true_front: Frente de Pareto verdadeira
            p: Parâmetro da norma Minkowski
            
        Returns:
            Valor do GD
        """
        if len(obtained_front) == 0:
            return np.inf
        
        if len(true_front) == 0:
            return np.nan
        
        # Calcula distâncias de cada ponto da frente obtida para a frente verdadeira
        distances = cdist(obtained_front, true_front, metric='minkowski', p=p)
        
        # Pega a menor distância de cada ponto da frente obtida
        min_distances = np.min(distances, axis=1)
        
        # GD é a média das menores distâncias
        gd = np.mean(min_distances)
        
        return gd
    
    @staticmethod
    def calculate_hypervolume(front: np.ndarray, 
                            reference_point: np.ndarray = None) -> float:
        """
        Calcula o Hypervolume (HV) - implementação simplificada
        
        Args:
            front: Frente de Pareto
            reference_point: Ponto de referência (padrão: point dominado por todos)
            
        Returns:
            Valor do hypervolume
        """
        if len(front) == 0:
            return 0.0
        
        if reference_point is None:
            # Usa ponto dominado por todos os pontos da frente
            reference_point = np.max(front, axis=0) + 1
        
        # Implementação simplificada para 2D
        if front.shape[1] == 2:
            # Ordena pontos por primeiro objetivo
            sorted_indices = np.argsort(front[:, 0])
            sorted_front = front[sorted_indices]
            
            hv = 0.0
            for i, point in enumerate(sorted_front):
                if i == 0:
                    width = reference_point[0] - point[0]
                else:
                    width = sorted_front[i-1][0] - point[0]
                
                height = reference_point[1] - point[1]
                hv += width * height
            
            return hv
        
        # Para mais de 2 objetivos, usa aproximação
        else:
            # Método Monte Carlo simplificado
            n_samples = 100000
            
            # Gera pontos aleatórios no hiperrretângulo
            samples = np.random.uniform(
                low=np.min(front, axis=0),
                high=reference_point,
                size=(n_samples, front.shape[1])
            )
            
            # Conta quantos pontos são dominados pela frente
            dominated_count = 0
            for sample in samples:
                if np.any(np.all(front <= sample, axis=1)):
                    dominated_count += 1
            
            # Calcula volume do hiperrretângulo
            volume = np.prod(reference_point - np.min(front, axis=0))
            
            return volume * (dominated_count / n_samples)
    
    @staticmethod
    def calculate_spread(front: np.ndarray) -> float:
        """
        Calcula o Spread (Delta) - medida de distribuição
        
        Args:
            front: Frente de Pareto
            
        Returns:
            Valor do spread
        """
        if len(front) <= 2:
            return 0.0
        
        # Ordena pontos por primeiro objetivo
        sorted_indices = np.argsort(front[:, 0])
        sorted_front = front[sorted_indices]
        
        # Calcula distâncias entre pontos consecutivos
        distances = []
        for i in range(1, len(sorted_front)):
            dist = np.linalg.norm(sorted_front[i] - sorted_front[i-1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Spread é o desvio padrão das distâncias
        mean_dist = np.mean(distances)
        spread = np.sqrt(np.mean((distances - mean_dist)**2))
        
        return spread


class SparsityMetrics:
    """Implementa métricas de esparsidade para avaliar soluções esparsas"""
    
    @staticmethod
    def calculate_sparsity(population: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcula métricas de esparsidade para uma população
        
        Args:
            population: Lista de soluções (máscaras binárias)
            
        Returns:
            Dicionário com métricas de esparsidade
        """
        if len(population) == 0:
            return {}
        
        # Converte para array numpy se necessário
        if isinstance(population[0], list):
            population = [np.array(ind) for ind in population]
        
        # Calcula esparsidade para cada indivíduo
        sparsity_values = []
        dimension = len(population[0])
        
        for individual in population:
            # Conta número de variáveis ativas (não-zero)
            active_vars = np.sum(individual != 0)
            sparsity_values.append(active_vars)
        
        sparsity_values = np.array(sparsity_values)
        
        # Calcula métricas
        metrics = {
            'mean_sparsity': np.mean(sparsity_values),
            'std_sparsity': np.std(sparsity_values),
            'min_sparsity': np.min(sparsity_values),
            'max_sparsity': np.max(sparsity_values),
            'median_sparsity': np.median(sparsity_values),
            'mean_sparsity_percentage': np.mean(sparsity_values) / dimension * 100,
            'std_sparsity_percentage': np.std(sparsity_values) / dimension * 100,
        }
        
        return metrics
    
    @staticmethod
    def calculate_variable_frequency(population: List[np.ndarray]) -> np.ndarray:
        """
        Calcula frequência de uso de cada variável
        
        Args:
            population: Lista de soluções (máscaras binárias)
            
        Returns:
            Array com frequência de uso de cada variável
        """
        if len(population) == 0:
            return np.array([])
        
        # Converte para array numpy
        population_array = np.array(population)
        
        # Calcula frequência de cada variável
        variable_frequency = np.mean(population_array != 0, axis=0)
        
        return variable_frequency
    
    @staticmethod
    def calculate_diversity_index(population: List[np.ndarray]) -> float:
        """
        Calcula índice de diversidade das soluções esparsas
        
        Args:
            population: Lista de soluções
            
        Returns:
            Índice de diversidade
        """
        if len(population) <= 1:
            return 0.0
        
        # Converte para array numpy
        population_array = np.array(population)
        
        # Calcula distâncias hamming entre todas as soluções
        distances = []
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                # Distância Hamming (diferenças nas máscaras)
                hamming_dist = np.sum(population_array[i] != population_array[j])
                distances.append(hamming_dist)
        
        # Índice de diversidade é a média das distâncias
        diversity_index = np.mean(distances)
        
        return diversity_index


class StatisticalTests:
    """Implementa testes estatísticos para comparação de algoritmos"""
    
    @staticmethod
    def wilcoxon_rank_sum_test(sample1: List[float], 
                              sample2: List[float], 
                              alpha: float = 0.05) -> Dict[str, Any]:
        """
        Executa o teste de soma de ranks de Wilcoxon
        
        Args:
            sample1: Primeira amostra
            sample2: Segunda amostra
            alpha: Nível de significância
            
        Returns:
            Resultado do teste
        """
        try:
            # Usa Mann-Whitney U test (equivalente ao Wilcoxon rank-sum para amostras independentes)
            statistic, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
            
            # Determina significância
            is_significant = p_value < alpha
            
            # Determina qual algoritmo é melhor (assumindo que menor é melhor)
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)
            
            if is_significant:
                if mean1 < mean2:
                    result = "+"  # Primeiro algoritmo é significativamente melhor
                else:
                    result = "-"  # Segundo algoritmo é significativamente melhor
            else:
                result = "="  # Não há diferença significativa
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'is_significant': is_significant,
                'result': result,
                'mean1': mean1,
                'mean2': mean2,
                'alpha': alpha
            }
            
        except Exception as e:
            warnings.warn(f"Erro no teste estatístico: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'is_significant': False,
                'result': "?",
                'mean1': np.mean(sample1),
                'mean2': np.mean(sample2),
                'alpha': alpha,
                'error': str(e)
            }
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], 
                                     method: str = 'bonferroni') -> List[float]:
        """
        Aplica correção para múltiplas comparações
        
        Args:
            p_values: Lista de p-valores
            method: Método de correção ('bonferroni' ou 'holm')
            
        Returns:
            P-valores corrigidos
        """
        p_values = np.array(p_values)
        
        if method == 'bonferroni':
            # Correção de Bonferroni
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)
            
        elif method == 'holm':
            # Método de Holm-Bonferroni
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p = np.zeros_like(p_values)
            for i, p in enumerate(sorted_p):
                correction_factor = len(p_values) - i
                corrected_p[sorted_indices[i]] = min(p * correction_factor, 1.0)
            
        else:
            raise ValueError(f"Método de correção desconhecido: {method}")
        
        return corrected_p.tolist()


class ResultsAnalyzer:
    """Analisa e compara resultados de experimentos"""
    
    def __init__(self):
        self.quality_metrics = QualityMetrics()
        self.sparsity_metrics = SparsityMetrics()
        self.statistical_tests = StatisticalTests()
    
    def analyze_experiment(self, 
                          populations: List[List[np.ndarray]],
                          objective_values: List[List[np.ndarray]],
                          true_pareto_front: np.ndarray,
                          problem_info: Dict[str, Any]) -> ExperimentResults:
        """
        Analisa resultados de um experimento completo
        
        Args:
            populations: Lista de populações finais para cada execução
            objective_values: Lista de valores de objetivo para cada execução
            true_pareto_front: Frente de Pareto verdadeira
            problem_info: Informações do problema
            
        Returns:
            Resultados analisados
        """
        num_runs = len(populations)
        
        # Calcula métricas de qualidade
        igd_values = []
        hypervolume_values = []
        
        for i in range(num_runs):
            # IGD
            igd = self.quality_metrics.calculate_igd(
                objective_values[i], true_pareto_front
            )
            igd_values.append(igd)
            
            # Hypervolume
            hv = self.quality_metrics.calculate_hypervolume(objective_values[i])
            hypervolume_values.append(hv)
        
        # Calcula métricas de esparsidade
        sparsity_values = []
        sparsity_percentages = []
        
        for i in range(num_runs):
            # Extrai máscaras ou soluções esparsas
            if hasattr(populations[i][0], 'mask'):
                masks = [ind.mask for ind in populations[i]]
            else:
                masks = [ind for ind in populations[i]]
            
            sparsity_metrics = self.sparsity_metrics.calculate_sparsity(masks)
            sparsity_values.append(sparsity_metrics.get('mean_sparsity', 0))
            sparsity_percentages.append(sparsity_metrics.get('mean_sparsity_percentage', 0))
        
        # Cria objeto de resultados
        results = ExperimentResults(
            problem_name=problem_info.get('name', 'Unknown'),
            algorithm_name=problem_info.get('algorithm', 'SparseEA-AGDS'),
            dimension=problem_info.get('dimension', 0),
            num_objectives=problem_info.get('num_objectives', 0),
            num_runs=num_runs,
            igd_values=igd_values,
            hypervolume_values=hypervolume_values,
            sparsity_values=sparsity_values,
            sparsity_percentage=sparsity_percentages
        )
        
        return results
    
    def compare_algorithms(self, 
                          results1: ExperimentResults,
                          results2: ExperimentResults,
                          alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compara resultados de dois algoritmos
        
        Args:
            results1: Resultados do primeiro algoritmo
            results2: Resultados do segundo algoritmo
            alpha: Nível de significância
            
        Returns:
            Resultado da comparação
        """
        comparison = {}
        
        # Compara IGD
        igd_test = self.statistical_tests.wilcoxon_rank_sum_test(
            results1.igd_values, results2.igd_values, alpha
        )
        comparison['igd'] = igd_test
        
        # Compara Hypervolume
        if results1.hypervolume_values and results2.hypervolume_values:
            hv_test = self.statistical_tests.wilcoxon_rank_sum_test(
                results2.hypervolume_values, results1.hypervolume_values, alpha  # Inverte para maior ser melhor
            )
            comparison['hypervolume'] = hv_test
        
        # Compara Esparsidade
        if results1.sparsity_values and results2.sparsity_values:
            sparsity_test = self.statistical_tests.wilcoxon_rank_sum_test(
                results1.sparsity_values, results2.sparsity_values, alpha
            )
            comparison['sparsity'] = sparsity_test
        
        return comparison
    
    def generate_summary_table(self, 
                              results_list: List[ExperimentResults],
                              metrics: List[str] = ['igd', 'sparsity_percentage']) -> str:
        """
        Gera tabela resumo dos resultados
        
        Args:
            results_list: Lista de resultados
            metrics: Métricas a incluir na tabela
            
        Returns:
            Tabela em formato string
        """
        if not results_list:
            return "Nenhum resultado fornecido"
        
        # Cabeçalho
        header = f"{'Problema':<15} {'Algoritmo':<15} {'D':<5} {'M':<3}"
        for metric in metrics:
            header += f" {metric.upper():<12}"
        header += "\n" + "-" * len(header)
        
        # Linhas da tabela
        lines = [header]
        
        for result in results_list:
            line = f"{result.problem_name:<15} {result.algorithm_name:<15} {result.dimension:<5} {result.num_objectives:<3}"
            
            for metric in metrics:
                metric_result = result.get_metric_summary(metric)
                if not np.isnan(metric_result.value):
                    line += f" {metric_result.value:<8.4f}±{metric_result.std_dev:<.2f}"
                else:
                    line += f" {'N/A':<12}"
            
            lines.append(line)
        
        return "\n".join(lines)


# Função utilitária para análise rápida
def quick_analysis(population_objectives: List[np.ndarray],
                  true_pareto_front: np.ndarray,
                  population_masks: List[np.ndarray] = None) -> Dict[str, float]:
    """
    Análise rápida de uma única execução
    
    Args:
        population_objectives: Objetivos da população final
        true_pareto_front: Frente de Pareto verdadeira
        population_masks: Máscaras da população (opcional)
        
    Returns:
        Dicionário com métricas principais
    """
    metrics = {}
    
    # Métricas de qualidade
    quality_metrics = QualityMetrics()
    metrics['igd'] = quality_metrics.calculate_igd(population_objectives, true_pareto_front)
    metrics['hypervolume'] = quality_metrics.calculate_hypervolume(population_objectives)
    metrics['spread'] = quality_metrics.calculate_spread(population_objectives)
    
    # Métricas de esparsidade
    if population_masks is not None:
        sparsity_metrics = SparsityMetrics()
        sparsity_results = sparsity_metrics.calculate_sparsity(population_masks)
        metrics.update(sparsity_results)
    
    return metrics 