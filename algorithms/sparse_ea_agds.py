"""
SparseEA-AGDS: Sparse Evolutionary Algorithm with Adaptive Genetic operators 
and Dynamic Scoring mechanism

Refactored implementation following software engineering best practices
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist

# Import from new structure
import sys
sys.path.append('..')
from problems.base import Problem
from config.experiment_config import AlgorithmConfig
from metrics.quality_metrics import find_pareto_front


@dataclass
class Individual:
    """Represents an individual in the population"""
    dec: np.ndarray = field(default_factory=lambda: np.array([]))  # Real variables
    mask: np.ndarray = field(default_factory=lambda: np.array([]))  # Binary mask
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))  # Objective values
    rank: int = float('inf')  # Non-domination rank
    crowding_distance: float = 0.0  # Crowding distance
    
    @property
    def solution(self) -> np.ndarray:
        """Returns the actual solution X = dec * mask"""
        return self.dec * self.mask
    
    def __post_init__(self):
        if len(self.dec) > 0 and len(self.mask) == 0:
            self.mask = np.zeros_like(self.dec, dtype=int)


class SparseEAAGDS:
    """
    SparseEA-AGDS Algorithm Implementation
    
    Features:
    - Adaptive genetic operators
    - Dynamic scoring mechanism  
    - Reference point-based environmental selection
    - Function evaluation counting
    - Controlled randomness via seeds
    """
    
    def __init__(self, problem: Problem, config: AlgorithmConfig, seed: Optional[int] = None):
        """
        Initialize the SparseEA-AGDS algorithm
        
        Args:
            problem: Optimization problem instance
            config: Algorithm configuration
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.config = config
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Algorithm parameters
        self.population_size = config.population_size
        self.max_function_evaluations = config.max_function_evaluations
        self.Pc0 = config.Pc0
        self.Pm0 = config.Pm0
        self.eta_c = config.eta_c
        self.eta_m = config.eta_m
        
        # Problem properties
        self.dimension = problem.dimension
        self.num_objectives = problem.num_objectives
        self.lower_bound, self.upper_bound = problem.get_bounds()
        
        # Runtime tracking
        self.generation = 0
        self.function_evaluations = 0
        self.convergence_history = []
        
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete SparseEA-AGDS algorithm
        
        Returns:
            Dictionary containing final population and run statistics
        """
        # Reset counters
        self.generation = 0
        self.function_evaluations = 0
        self.convergence_history = []
        self.problem.reset_evaluation_counter()
        
        # Phase 1: Calculate initial variable importance scores
        initial_scores = self._calculate_initial_scores()
        
        # Phase 2: Initialize population
        population = self._initialize_population(initial_scores)
        
        # Phase 3: Evolution loop
        while not self._termination_criterion():
            # Generate offspring using adaptive genetic operators
            offspring_dec = self._adaptive_genetic_operator(population)
            
            # Generate offspring masks using dynamic scoring
            offspring_masks = self._dynamic_scoring_mechanism(population)
            
            # Create offspring population
            offspring = self._create_offspring(offspring_dec, offspring_masks)
            
            # Environmental selection
            combined_population = population + offspring
            population = self._environmental_selection(combined_population)
            
            # Update generation counter
            self.generation += 1
            
            # Record convergence information
            self._record_convergence(population)
        
        # Extract final Pareto front
        objectives = np.array([ind.objectives for ind in population])
        pareto_front, pareto_indices = find_pareto_front(objectives)
        final_pareto_population = [population[i] for i in pareto_indices]
        
        return {
            'population': population,
            'pareto_front': pareto_front,
            'pareto_population': final_pareto_population,
            'generation': self.generation,
            'function_evaluations': self.problem.get_evaluation_count(),
            'convergence_history': self.convergence_history
        }
    
    def _termination_criterion(self) -> bool:
        """Check if algorithm should terminate"""
        return self.problem.get_evaluation_count() >= self.max_function_evaluations
    
    def _calculate_initial_scores(self) -> np.ndarray:
        """
        Calculate initial importance scores for each variable
        
        Based on Section 3.1 of the paper: Initial scoring mechanism
        """
        scores = np.zeros(self.dimension)
        num_runs = 10  # Number of runs for robust estimation
        
        for run in range(num_runs):
            # Create matrix of real variables (D x D)
            dec_matrix = np.random.uniform(
                self.lower_bound, self.upper_bound, (self.dimension, self.dimension)
            )
            
            # Create identity matrix for masks (D x D)
            mask_matrix = np.eye(self.dimension, dtype=int)
            
            # Create temporary population G
            G = []
            for i in range(self.dimension):
                ind = Individual()
                ind.dec = dec_matrix[i]
                ind.mask = mask_matrix[i]
                ind.objectives = self.problem.evaluate(ind.solution)
                G.append(ind)
            
            # Perform non-dominated sorting
            fronts = self._fast_non_dominated_sort(G)
            
            # Accumulate scores based on ranks
            for front_idx, front in enumerate(fronts):
                for ind_idx in front:
                    scores[ind_idx] += front_idx + 1
        
        return scores
    
    def _initialize_population(self, initial_scores: np.ndarray) -> List[Individual]:
        """
        Initialize population using initial variable importance scores
        
        Based on Section 3.2 of the paper
        """
        population = []
        
        for _ in range(self.population_size):
            ind = Individual()
            
            # Generate random real variables
            ind.dec = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
            
            # Initialize mask with zeros
            ind.mask = np.zeros(self.dimension, dtype=int)
            
            # Select active variables using binary tournament selection
            sparsity_factor = np.random.rand()  # Random sparsity level
            num_active = int(sparsity_factor * self.dimension)
            
            if num_active > 0:
                active_positions = self._binary_tournament_selection(initial_scores, num_active)
                ind.mask[active_positions] = 1
            
            # Evaluate the individual
            ind.objectives = self.problem.evaluate(ind.solution)
            population.append(ind)
        
        return population
    
    def _adaptive_genetic_operator(self, population: List[Individual]) -> List[np.ndarray]:
        """
        Adaptive genetic operator for generating offspring decision variables
        
        Based on Section 4.1 of the paper: Equations 5-7
        """
        # Calculate non-domination ranks
        fronts = self._fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks) if ranks else 0
        offspring_dec = []
        
        for i in range(len(population)):
            # Equation 5: Selection probability
            Ps_i = (max_rank - ranks[i] + 1) / max_rank if max_rank > 0 else 1.0
            
            # Equations 6-7: Adaptive probabilities
            Pc_i = self.Pc0 * Ps_i
            Pm_i = self.Pm0 * Ps_i
            
            # Start with parent's decision variables
            child_dec = population[i].dec.copy()
            
            # Crossover operation
            if np.random.rand() < Pc_i:
                partner_idx = np.random.randint(0, len(population))
                child_dec, _ = self._simulated_binary_crossover(
                    population[i].dec, population[partner_idx].dec
                )
            
            # Mutation operation
            if np.random.rand() < Pm_i:
                child_dec = self._polynomial_mutation(child_dec)
            
            offspring_dec.append(child_dec)
        
        return offspring_dec
    
    def _dynamic_scoring_mechanism(self, population: List[Individual]) -> List[np.ndarray]:
        """
        Dynamic scoring mechanism for generating offspring masks
        
        Based on Section 4.2 of the paper: Equations 8-10
        """
        # Calculate non-domination ranks
        fronts = self._fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks) if ranks else 0
        
        # Equation 8: Layer score
        Sr = np.array([max_rank - r + 1 for r in ranks])
        
        # Create mask matrix
        mask_matrix = np.array([ind.mask for ind in population])
        
        # Equation 9: Weighted score
        sumS = Sr.T @ mask_matrix
        
        # Equation 10: Final variable score
        maxS = np.max(sumS) if len(sumS) > 0 else 1
        S = maxS - sumS + 1
        
        # Generate offspring masks
        offspring_masks = []
        for _ in range(len(population)):
            child_mask = np.zeros(self.dimension, dtype=int)
            
            # Select variables based on updated scores
            sparsity_factor = np.random.rand()
            num_active = int(sparsity_factor * self.dimension)
            
            if num_active > 0:
                active_positions = self._binary_tournament_selection(S, num_active)
                child_mask[active_positions] = 1
            
            offspring_masks.append(child_mask)
        
        return offspring_masks
    
    def _create_offspring(self, offspring_dec: List[np.ndarray], 
                         offspring_masks: List[np.ndarray]) -> List[Individual]:
        """Create offspring population from decision variables and masks"""
        offspring = []
        
        for i in range(len(offspring_dec)):
            child = Individual()
            child.dec = offspring_dec[i]
            child.mask = offspring_masks[i]
            child.objectives = self.problem.evaluate(child.solution)
            offspring.append(child)
        
        return offspring
    
    def _environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """
        Reference point-based environmental selection
        
        Based on Section 4.3 of the paper
        """
        # Generate reference points
        reference_points = self._generate_reference_points()
        
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined_population)
        
        new_population = []
        front_idx = 0
        
        # Add complete fronts
        while (front_idx < len(fronts) and 
               len(new_population) + len(fronts[front_idx]) <= self.population_size):
            for ind_idx in fronts[front_idx]:
                new_population.append(combined_population[ind_idx])
            front_idx += 1
        
        # Handle last front if necessary
        if front_idx < len(fronts) and len(new_population) < self.population_size:
            last_front = [combined_population[i] for i in fronts[front_idx]]
            remaining_slots = self.population_size - len(new_population)
            
            selected = self._reference_point_selection(
                new_population + last_front, reference_points, remaining_slots
            )
            new_population.extend(selected)
        
        return new_population[:self.population_size]
    
    def _reference_point_selection(self, population: List[Individual], 
                                  reference_points: np.ndarray, 
                                  num_select: int) -> List[Individual]:
        """Select individuals based on reference points"""
        if num_select <= 0:
            return []
        
        # Get objectives
        objectives = np.array([ind.objectives for ind in population])
        
        # Normalize objectives
        min_obj = np.min(objectives, axis=0)
        max_obj = np.max(objectives, axis=0)
        range_obj = max_obj - min_obj
        range_obj[range_obj == 0] = 1  # Avoid division by zero
        normalized_obj = (objectives - min_obj) / range_obj
        
        # Associate individuals to reference points
        distances = cdist(normalized_obj, reference_points)
        associations = np.argmin(distances, axis=1)
        
        # Count niche populations
        niche_counts = np.bincount(associations, minlength=len(reference_points))
        
        # Select from last front
        selected = []
        available_indices = list(range(len(population)))
        
        for _ in range(num_select):
            if not available_indices:
                break
            
            # Find least crowded niche
            min_count = min(niche_counts[associations[i]] for i in available_indices)
            candidates = [i for i in available_indices 
                         if niche_counts[associations[i]] == min_count]
            
            # Select randomly from candidates
            selected_idx = random.choice(candidates)
            selected.append(population[selected_idx])
            
            # Update counts and remove selected
            niche_counts[associations[selected_idx]] += 1
            available_indices.remove(selected_idx)
        
        return selected
    
    def _generate_reference_points(self) -> np.ndarray:
        """Generate reference points using Das-Dennis method"""
        if self.num_objectives == 2:
            p = 12  # Number of divisions
        elif self.num_objectives == 3:
            p = 8
        else:
            p = max(3, 12 - self.num_objectives)
        
        def generate_recursive(M, p, current_point, points):
            if M == 1:
                current_point[0] = p
                points.append(current_point.copy())
                return
            
            for i in range(p + 1):
                current_point[M - 1] = i
                generate_recursive(M - 1, p - i, current_point, points)
        
        points = []
        generate_recursive(self.num_objectives, p, np.zeros(self.num_objectives), points)
        reference_points = np.array(points) / p
        
        return reference_points
    
    def _binary_tournament_selection(self, scores: np.ndarray, k: int) -> List[int]:
        """Binary tournament selection based on scores (lower is better)"""
        selected = []
        for _ in range(k):
            i, j = random.sample(range(len(scores)), 2)
            if scores[i] < scores[j]:
                selected.append(i)
            else:
                selected.append(j)
        return selected
    
    def _simulated_binary_crossover(self, parent1: np.ndarray, 
                                   parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)"""
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
                
                # Apply bounds
                child1[i] = np.clip(child1[i], self.lower_bound[i], self.upper_bound[i])
                child2[i] = np.clip(child2[i], self.lower_bound[i], self.upper_bound[i])
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
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
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """Fast non-dominated sorting algorithm"""
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
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2"""
        better_in_at_least_one = False
        for i in range(len(ind1.objectives)):
            if ind1.objectives[i] > ind2.objectives[i]:
                return False
            elif ind1.objectives[i] < ind2.objectives[i]:
                better_in_at_least_one = True
        return better_in_at_least_one
    
    def _record_convergence(self, population: List[Individual]):
        """Record convergence information for analysis"""
        objectives = np.array([ind.objectives for ind in population])
        sparsities = np.array([np.sum(ind.mask) for ind in population])
        
        convergence_info = {
            'generation': self.generation,
            'function_evaluations': self.problem.get_evaluation_count(),
            'mean_objectives': np.mean(objectives, axis=0).tolist(),
            'std_objectives': np.std(objectives, axis=0).tolist(),
            'mean_sparsity': float(np.mean(sparsities)),
            'std_sparsity': float(np.std(sparsities))
        }
        
        self.convergence_history.append(convergence_info) 