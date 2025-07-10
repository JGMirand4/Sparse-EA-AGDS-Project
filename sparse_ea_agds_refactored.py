"""
SparseEA-AGDS Refatorado
=======================

Implementação refatorada do algoritmo SparseEA-AGDS seguindo boas práticas
de engenharia de software para pesquisa.
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
import time
import warnings

from problems import Problem, create_problem
from config import ExperimentConfig, AlgorithmConfig
from metrics import QualityMetrics, SparsityMetrics, ExperimentResults


@dataclass
class Individual:
    """Representa um indivíduo na população"""
    
    dec: np.ndarray = field(default_factory=lambda: np.array([]))  # Variáveis de decisão
    mask: np.ndarray = field(default_factory=lambda: np.array([]))  # Máscara binária
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))  # Valores dos objetivos
    rank: int = float('inf')  # Rank de não-dominância
    crowding_distance: float = 0.0  # Distância de crowding
    
    @property
    def solution(self) -> np.ndarray:
        """Retorna a solução X = dec * mask"""
        return self.dec * self.mask
    
    def __post_init__(self):
        """Inicialização pós-criação"""
        if len(self.dec) > 0 and len(self.mask) == 0:
            self.mask = np.zeros_like(self.dec, dtype=int)


class SparseEAAGDS:
    """
    Implementação refatorada do algoritmo SparseEA-AGDS
    
    Características:
    - Usa interface Problem para problemas
    - Configuração externa via AlgorithmConfig
    - Controle de avaliações de função
    - Logging detalhado
    - Métricas integradas
    """
    
    def __init__(self, problem: Problem, config: AlgorithmConfig):
        """
        Inicializa o algoritmo
        
        Args:
            problem: Problema a ser resolvido
            config: Configuração do algoritmo
        """
        self.problem = problem
        self.config = config
        
        # Parâmetros do problema
        self.D = problem.dimension
        self.M = problem.num_objectives
        self.lower_bound, self.upper_bound = problem.get_bounds()
        
        # Parâmetros do algoritmo
        self.N = config.population_size
        self.max_fe = config.max_function_evaluations
        self.Pc0 = config.Pc0
        self.Pm0 = config.Pm0
        self.eta_c = config.eta_c
        self.eta_m = config.eta_m
        
        # Controle de execução
        self.current_generation = 0
        self.function_evaluations = 0
        self.start_time = None
        
        # Histórico para logging
        self.history = {
            'generations': [],
            'function_evaluations': [],
            'best_igd': [],
            'mean_sparsity': [],
            'execution_time': []
        }
        
        # Métricas
        self.quality_metrics = QualityMetrics()
        self.sparsity_metrics = SparsityMetrics()
        
        # Seed para reprodutibilidade
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
    
    def run(self, verbose: bool = True) -> Tuple[List[Individual], Dict[str, Any]]:
        """
        Executa o algoritmo SparseEA-AGDS
        
        Args:
            verbose: Se deve imprimir informações de progresso
            
        Returns:
            Tuple (população_final, informações_execução)
        """
        self.start_time = time.time()
        self.problem.reset_function_evaluations()
        
        if verbose:
            print(f"Iniciando SparseEA-AGDS para {self.problem.name}")
            print(f"Problema: D={self.D}, M={self.M}")
            print(f"Algoritmo: N={self.N}, maxFE={self.max_fe}")
        
        # Fase 1: Cálculo de pontuações iniciais
        if verbose:
            print("Calculando pontuações iniciais...")
        
        initial_scores = self._calculate_initial_scores()
        
        # Fase 2: Inicialização da população
        if verbose:
            print("Inicializando população...")
        
        population = self._initialize_population(initial_scores)
        
        # Atualiza histórico inicial
        self._update_history(population)
        
        # Fase 3: Loop evolucionário
        if verbose:
            print("Iniciando evolução...")
        
        while self.problem.function_evaluations < self.max_fe:
            # Operador genético adaptativo
            offspring_dec = self._adaptive_genetic_operator(population)
            
            # Mecanismo de pontuação dinâmica
            offspring_masks = self._dynamic_scoring_mechanism(population)
            
            # Cria população de filhos
            offspring = self._create_offspring(offspring_dec, offspring_masks)
            
            # Verifica se excedeu limite de avaliações
            if self.problem.function_evaluations >= self.max_fe:
                break
            
            # Seleção ambiental
            combined_population = population + offspring
            population = self._environmental_selection(combined_population)
            
            # Atualiza histórico
            self.current_generation += 1
            self._update_history(population)
            
            # Logging
            if verbose and self.current_generation % 10 == 0:
                elapsed_time = time.time() - self.start_time
                print(f"Geração {self.current_generation}: FE={self.problem.function_evaluations}, "
                      f"Tempo={elapsed_time:.2f}s")
        
        # Informações finais
        execution_info = {
            'generations': self.current_generation,
            'function_evaluations': self.problem.function_evaluations,
            'execution_time': time.time() - self.start_time,
            'history': self.history.copy()
        }
        
        if verbose:
            print(f"Otimização concluída!")
            print(f"Gerações: {execution_info['generations']}")
            print(f"Avaliações de função: {execution_info['function_evaluations']}")
            print(f"Tempo de execução: {execution_info['execution_time']:.2f}s")
        
        return population, execution_info
    
    def _calculate_initial_scores(self) -> np.ndarray:
        """Calcula pontuações iniciais das variáveis"""
        scores = np.zeros(self.D)
        
        for run in range(self.config.initial_scoring_runs):
            # Cria matriz D x D de variáveis reais aleatórias
            dec_matrix = np.random.uniform(self.lower_bound, self.upper_bound, (self.D, self.D))
            # Cria matriz identidade para mask
            mask_matrix = np.eye(self.D)
            
            # Cria população temporária G
            G = []
            for i in range(self.D):
                ind = Individual()
                ind.dec = dec_matrix[i]
                ind.mask = mask_matrix[i]
                ind.objectives = self.problem.evaluate(ind.solution)
                G.append(ind)
            
            # Ordena população G
            fronts = self._fast_non_dominated_sort(G)
            
            # Acumula pontuações
            for front_idx, front in enumerate(fronts):
                for ind_idx in front:
                    scores[ind_idx] += front_idx + 1
        
        return scores
    
    def _initialize_population(self, initial_scores: np.ndarray) -> List[Individual]:
        """Inicializa a população usando pontuações iniciais"""
        population = []
        
        for _ in range(self.N):
            ind = Individual()
            
            # Gera variáveis reais aleatórias
            ind.dec = np.random.uniform(self.lower_bound, self.upper_bound, self.D)
            
            # Inicializa mask com zeros
            ind.mask = np.zeros(self.D, dtype=int)
            
            # Seleciona posições para ativar no mask
            c = np.random.rand()  # Parâmetro de esparsidade
            num_active = int(c * self.D)
            
            if num_active > 0:
                active_positions = self._binary_tournament_selection(initial_scores, num_active)
                ind.mask[active_positions] = 1
            
            # Avalia o indivíduo
            ind.objectives = self.problem.evaluate(ind.solution)
            population.append(ind)
        
        return population
    
    def _adaptive_genetic_operator(self, population: List[Individual]) -> List[np.ndarray]:
        """Implementa operador genético adaptativo"""
        # Calcula ranks
        fronts = self._fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks) if ranks else 0
        
        offspring_dec = []
        
        for i in range(len(population)):
            # Calcula probabilidade de seleção
            Ps_i = (max_rank - ranks[i] + 1) / max_rank if max_rank > 0 else 1.0
            
            # Calcula probabilidades adaptativas
            Pc_i = self.Pc0 * Ps_i
            Pm_i = self.Pm0 * Ps_i
            
            # Operações genéticas
            child_dec = population[i].dec.copy()
            
            # Crossover
            if np.random.rand() < Pc_i:
                partner_idx = np.random.randint(0, len(population))
                child_dec, _ = self._simulated_binary_crossover(
                    population[i].dec, population[partner_idx].dec
                )
            
            # Mutação
            if np.random.rand() < Pm_i:
                child_dec = self._polynomial_mutation(child_dec)
            
            offspring_dec.append(child_dec)
        
        return offspring_dec
    
    def _dynamic_scoring_mechanism(self, population: List[Individual]) -> List[np.ndarray]:
        """Implementa mecanismo de pontuação dinâmica"""
        # Calcula ranks da população
        fronts = self._fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks) if ranks else 0
        
        # Calcula pontuação de camada
        Sr = np.array([max_rank - r + 1 for r in ranks])
        
        # Cria matriz de masks
        mask_matrix = np.array([ind.mask for ind in population])
        
        # Calcula pontuação ponderada
        sumS = Sr.T @ mask_matrix
        
        # Atualiza pontuação final da variável
        maxS = np.max(sumS) if len(sumS) > 0 else 1
        S = maxS - sumS + 1
        
        # Gera masks dos filhos
        offspring_masks = []
        for _ in range(len(population)):
            child_mask = np.zeros(self.D, dtype=int)
            
            # Seleciona dimensões baseado na pontuação S
            c = np.random.rand()
            num_active = int(c * self.D)
            
            if num_active > 0:
                active_positions = self._binary_tournament_selection(S, num_active)
                child_mask[active_positions] = 1
            
            offspring_masks.append(child_mask)
        
        return offspring_masks
    
    def _create_offspring(self, offspring_dec: List[np.ndarray], 
                         offspring_masks: List[np.ndarray]) -> List[Individual]:
        """Cria população de filhos"""
        offspring = []
        
        for i in range(len(offspring_dec)):
            child = Individual()
            child.dec = offspring_dec[i]
            child.mask = offspring_masks[i]
            child.objectives = self.problem.evaluate(child.solution)
            offspring.append(child)
        
        return offspring
    
    def _environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Seleção ambiental baseada em pontos de referência"""
        # Gera pontos de referência
        reference_points = self._generate_reference_points(self.M, self.config.reference_points_param)
        
        # Ordena população combinada
        fronts = self._fast_non_dominated_sort(combined_population)
        
        new_population = []
        front_idx = 0
        
        # Adiciona frentes completas
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.N:
            for ind_idx in fronts[front_idx]:
                new_population.append(combined_population[ind_idx])
            front_idx += 1
        
        # Seleção da última frente
        if front_idx < len(fronts) and len(new_population) < self.N:
            last_front = [combined_population[i] for i in fronts[front_idx]]
            remaining_slots = self.N - len(new_population)
            
            if remaining_slots == 1:
                new_population.append(last_front[0])
            else:
                # Associa aos pontos de referência
                associations, niche_counts = self._associate_to_reference_points(
                    new_population + last_front, reference_points
                )
                
                # Seleciona da última frente
                selected = self._select_from_last_front(
                    last_front, associations[len(new_population):], 
                    niche_counts, remaining_slots
                )
                new_population.extend(selected)
        
        return new_population[:self.N]
    
    def _select_from_last_front(self, last_front: List[Individual], 
                               associations: List[int], 
                               niche_counts: List[int], 
                               remaining_slots: int) -> List[Individual]:
        """Seleciona indivíduos da última frente"""
        selected = []
        current_niche_counts = niche_counts.copy()
        available_indices = list(range(len(last_front)))
        
        for _ in range(remaining_slots):
            if not available_indices:
                break
            
            # Encontra nicho com menor contagem
            min_niche_count = min(current_niche_counts)
            min_niche_indices = [i for i, count in enumerate(current_niche_counts) 
                               if count == min_niche_count]
            
            # Seleciona indivíduo do nicho menos povoado
            selected_idx = None
            for idx in available_indices:
                if associations[idx] in min_niche_indices:
                    selected_idx = idx
                    break
            
            if selected_idx is None:
                selected_idx = available_indices[0]
            
            selected.append(last_front[selected_idx])
            current_niche_counts[associations[selected_idx]] += 1
            available_indices.remove(selected_idx)
        
        return selected
    
    def _update_history(self, population: List[Individual]):
        """Atualiza histórico da execução"""
        # Calcula IGD se possível
        try:
            objectives = np.array([ind.objectives for ind in population])
            true_front = self.problem.get_true_pareto_front(1000)
            igd = self.quality_metrics.calculate_igd(objectives, true_front)
        except:
            igd = np.nan
        
        # Calcula esparsidade média
        try:
            masks = [ind.mask for ind in population]
            sparsity_metrics = self.sparsity_metrics.calculate_sparsity(masks)
            mean_sparsity = sparsity_metrics.get('mean_sparsity_percentage', 0)
        except:
            mean_sparsity = np.nan
        
        # Adiciona ao histórico
        self.history['generations'].append(self.current_generation)
        self.history['function_evaluations'].append(self.problem.function_evaluations)
        self.history['best_igd'].append(igd)
        self.history['mean_sparsity'].append(mean_sparsity)
        self.history['execution_time'].append(time.time() - self.start_time if self.start_time else 0)
    
    # Métodos auxiliares (versões simplificadas dos métodos originais)
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """Ordenação rápida não-dominada"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Verifica se ind1 domina ind2"""
        better_in_at_least_one = False
        for i in range(len(ind1.objectives)):
            if ind1.objectives[i] > ind2.objectives[i]:
                return False
            elif ind1.objectives[i] < ind2.objectives[i]:
                better_in_at_least_one = True
        return better_in_at_least_one
    
    def _binary_tournament_selection(self, scores: np.ndarray, k: int) -> List[int]:
        """Seleção por torneio binário"""
        selected = []
        for _ in range(k):
            i, j = random.sample(range(len(scores)), 2)
            if scores[i] < scores[j]:
                selected.append(i)
            else:
                selected.append(j)
        return selected
    
    def _simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                
                child1[i] = np.clip(child1[i], self.lower_bound[i], self.upper_bound[i])
                child2[i] = np.clip(child2[i], self.lower_bound[i], self.upper_bound[i])
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Mutação polinomial"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.rand() <= 1.0 / len(individual):
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                
                mutated[i] = individual[i] + delta * (self.upper_bound[i] - self.lower_bound[i])
                mutated[i] = np.clip(mutated[i], self.lower_bound[i], self.upper_bound[i])
        
        return mutated
    
    def _generate_reference_points(self, M: int, p: int) -> np.ndarray:
        """Gera pontos de referência usando método Das-Dennis"""
        def generate_recursive(M, p, current_point, points):
            if M == 1:
                current_point[0] = p
                points.append(current_point.copy())
                return
            
            for i in range(p + 1):
                current_point[M - 1] = i
                generate_recursive(M - 1, p - i, current_point, points)
        
        points = []
        generate_recursive(M, p, np.zeros(M), points)
        reference_points = np.array(points) / p
        
        return reference_points
    
    def _associate_to_reference_points(self, population: List[Individual], 
                                     reference_points: np.ndarray) -> Tuple[List[int], List[int]]:
        """Associa indivíduos aos pontos de referência"""
        objectives = np.array([ind.objectives for ind in population])
        
        # Normaliza objetivos
        min_obj = np.min(objectives, axis=0)
        max_obj = np.max(objectives, axis=0)
        normalized_obj = (objectives - min_obj) / (max_obj - min_obj + 1e-10)
        
        # Calcula distâncias
        distances = cdist(normalized_obj, reference_points)
        associations = np.argmin(distances, axis=1)
        
        # Conta nichos
        niche_counts = np.bincount(associations, minlength=len(reference_points))
        
        return associations.tolist(), niche_counts.tolist()


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Executa um único experimento com múltiplas execuções
    
    Args:
        config: Configuração do experimento
        
    Returns:
        Resultados do experimento
    """
    # Cria problema
    problem = create_problem(config.problem.name, config.problem.dimension, config.problem.num_objectives)
    
    # Armazena resultados
    all_populations = []
    all_objectives = []
    all_execution_info = []
    
    print(f"Executando experimento: {config.experiment_id}")
    print(f"Problema: {config.problem.name} (D={config.problem.dimension}, M={config.problem.num_objectives})")
    print(f"Algoritmo: N={config.algorithm.population_size}, maxFE={config.algorithm.max_function_evaluations}")
    print(f"Execuções: {config.num_runs}")
    
    for run_idx in range(config.num_runs):
        print(f"\nExecutando run {run_idx + 1}/{config.num_runs}...")
        
        # Usa seed específica para esta execução
        run_config = config.algorithm
        run_config.random_seed = config.random_seeds[run_idx]
        
        # Executa algoritmo
        algorithm = SparseEAAGDS(problem, run_config)
        population, exec_info = algorithm.run(verbose=config.verbose)
        
        # Armazena resultados
        all_populations.append(population)
        all_objectives.append(np.array([ind.objectives for ind in population]))
        all_execution_info.append(exec_info)
    
    # Analisa resultados
    print("\nAnalisando resultados...")
    
    # Obtém frente de Pareto verdadeira
    true_pareto_front = problem.get_true_pareto_front(10000)
    
    # Cria objeto de resultados
    from metrics import ResultsAnalyzer
    analyzer = ResultsAnalyzer()
    
    problem_info = {
        'name': config.problem.name,
        'algorithm': 'SparseEA-AGDS',
        'dimension': config.problem.dimension,
        'num_objectives': config.problem.num_objectives
    }
    
    results = analyzer.analyze_experiment(
        all_populations, all_objectives, true_pareto_front, problem_info
    )
    
    # Adiciona informações de execução
    results.function_evaluations = [info['function_evaluations'] for info in all_execution_info]
    results.execution_times = [info['execution_time'] for info in all_execution_info]
    
    return results


# Exemplo de uso
if __name__ == "__main__":
    from config import create_test_config
    
    # Cria configuração de teste
    config = create_test_config()
    
    # Executa experimento
    results = run_single_experiment(config)
    
    # Mostra resultados
    print("\n" + "="*50)
    print("RESULTADOS DO EXPERIMENTO")
    print("="*50)
    
    igd_summary = results.get_metric_summary('igd')
    sparsity_summary = results.get_metric_summary('sparsity_percentage')
    
    print(f"IGD: {igd_summary.value:.4f} ± {igd_summary.std_dev:.4f}")
    print(f"Esparsidade: {sparsity_summary.value:.2f}% ± {sparsity_summary.std_dev:.2f}%")
    
    if results.execution_times:
        time_summary = results.get_metric_summary('execution_times')
        print(f"Tempo de execução: {time_summary.value:.2f}s ± {time_summary.std_dev:.2f}s")
    
    print("\nExperimento concluído com sucesso!") 