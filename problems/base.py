"""
Base classes for optimization problems in SparseEA-AGDS framework
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Problem(ABC):
    """Base abstract class for optimization problems"""
    
    def __init__(self, dimension: int, num_objectives: int, name: str = "Problem"):
        self.dimension = dimension
        self.num_objectives = num_objectives
        self.name = name
        self._function_evaluations = 0
        
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate a solution and return objective values
        
        Args:
            solution: Decision variable vector of length self.dimension
            
        Returns:
            Array of objective values of length self.num_objectives
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounds for decision variables
        
        Returns:
            Tuple (lower_bounds, upper_bounds) where each is an array of length self.dimension
        """
        pass
    
    @abstractmethod
    def get_true_pareto_front(self, num_points: int = 10000) -> Optional[np.ndarray]:
        """
        Get the true Pareto front for IGD calculation
        
        Args:
            num_points: Number of points to sample on the Pareto front
            
        Returns:
            Array of shape (num_points, num_objectives) with true Pareto front points,
            or None if not available
        """
        pass
    
    def reset_evaluation_counter(self):
        """Reset the function evaluation counter"""
        self._function_evaluations = 0
    
    def get_evaluation_count(self) -> int:
        """Get the current number of function evaluations"""
        return self._function_evaluations
    
    def _count_evaluation(self):
        """Internal method to increment evaluation counter"""
        self._function_evaluations += 1
    
    def batch_evaluate(self, solutions: np.ndarray) -> np.ndarray:
        """
        Evaluate multiple solutions at once
        
        Args:
            solutions: Array of shape (n_solutions, dimension)
            
        Returns:
            Array of shape (n_solutions, num_objectives)
        """
        results = []
        for solution in solutions:
            results.append(self.evaluate(solution))
        return np.array(results)
    
    def __str__(self):
        return f"{self.name}(D={self.dimension}, M={self.num_objectives})"
    
    def __repr__(self):
        return self.__str__()


class TestProblem(Problem):
    """Base class for test problems with common utilities"""
    
    def __init__(self, dimension: int, num_objectives: int, name: str):
        super().__init__(dimension, num_objectives, name)
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Default bounds [0, 1] for all variables"""
        lower = np.zeros(self.dimension)
        upper = np.ones(self.dimension)
        return lower, upper
    
    def generate_uniform_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """
        Generate uniformly distributed points on the Pareto front
        This is a default implementation that should be overridden for specific problems
        """
        if self.num_objectives == 2:
            # For 2 objectives, create a simple line
            x = np.linspace(0, 1, num_points)
            y = 1 - x
            return np.column_stack([x, y])
        else:
            # For multiple objectives, use simplex sampling
            points = np.random.random((num_points, self.num_objectives))
            # Normalize to sum to 1 (simplex)
            points = points / np.sum(points, axis=1, keepdims=True)
            return points 