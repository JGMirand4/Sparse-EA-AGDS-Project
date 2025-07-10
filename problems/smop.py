"""
SMOP (Sparse Multi-Objective Optimization Problems) benchmark suite
Based on the original SparseEA paper and SparseEA-AGDS article
"""

import numpy as np
from typing import Tuple, Optional
from .base import TestProblem


class SMOP1(TestProblem):
    """
    SMOP1: Basic bi-objective sparse problem
    f1 = sum(x^2)
    f2 = sum((x-1)^2)
    
    Optimal Pareto front: x* can have many zero elements
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 2):
        super().__init__(dimension, num_objectives, "SMOP1")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        f1 = np.sum(solution**2)
        f2 = np.sum((solution - 1)**2)
        
        return np.array([f1, f2])
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """
        True Pareto front for SMOP1
        The optimal solutions lie on x_i ∈ [0, 1] for active variables
        """
        # Generate points on the line connecting (0, D) and (D, 0)
        t = np.linspace(0, 1, num_points)
        f1 = t * self.dimension
        f2 = (1 - t) * self.dimension
        
        return np.column_stack([f1, f2])


class SMOP2(TestProblem):
    """
    SMOP2: Modified SMOP1 with different offset
    f1 = sum(x^2)  
    f2 = sum((x-2)^2)
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 2):
        super().__init__(dimension, num_objectives, "SMOP2")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        f1 = np.sum(solution**2)
        f2 = np.sum((solution - 2)**2)
        
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """SMOP2 typically uses bounds [0, 2]"""
        lower = np.zeros(self.dimension)
        upper = np.full(self.dimension, 2.0)
        return lower, upper
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        # The optimal solutions are on the line from (0, 4*D) to (4*D, 0)
        t = np.linspace(0, 1, num_points)
        f1 = 4 * t * self.dimension
        f2 = 4 * (1 - t) * self.dimension
        
        return np.column_stack([f1, f2])


class SMOP3(TestProblem):
    """
    SMOP3: Tri-objective sparse problem
    f1 = sum(x^2)
    f2 = sum((x-1)^2) 
    f3 = sum((x-0.5)^2)
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 3):
        super().__init__(dimension, num_objectives, "SMOP3")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        f1 = np.sum(solution**2)
        f2 = np.sum((solution - 1)**2)
        f3 = np.sum((solution - 0.5)**2)
        
        return np.array([f1, f2, f3])
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Generate points on the 3D Pareto front using uniform sampling"""
        # Use Dirichlet distribution to generate points on simplex
        alpha = np.ones(3)
        points = np.random.dirichlet(alpha, num_points)
        
        # Scale by dimension to get realistic values
        pareto_front = points * self.dimension
        
        return pareto_front


class SMOP4(TestProblem):
    """
    SMOP4: Non-convex sparse problem
    f1 = sum(x^2) + sin(sum(x))
    f2 = sum((x-1)^2) + cos(sum(x))
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 2):
        super().__init__(dimension, num_objectives, "SMOP4")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        sum_x = np.sum(solution)
        f1 = np.sum(solution**2) + np.sin(sum_x)
        f2 = np.sum((solution - 1)**2) + np.cos(sum_x)
        
        return np.array([f1, f2])
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Approximate Pareto front for SMOP4 (non-convex)"""
        # Generate diverse solutions and compute their objectives
        solutions = []
        objectives = []
        
        # Sample uniformly in decision space
        for _ in range(num_points * 10):  # Oversample and filter
            x = np.random.uniform(0, 1, self.dimension)
            # Make sparse by randomly setting some variables to 0
            mask = np.random.random(self.dimension) < 0.1  # 10% sparsity
            x = x * mask
            
            obj = self.evaluate(x)
            solutions.append(x)
            objectives.append(obj)
        
        objectives = np.array(objectives)
        
        # Find non-dominated solutions
        is_pareto = self._find_pareto_front(objectives)
        pareto_objectives = objectives[is_pareto]
        
        # Return the requested number of points
        if len(pareto_objectives) > num_points:
            indices = np.random.choice(len(pareto_objectives), num_points, replace=False)
            return pareto_objectives[indices]
        else:
            return pareto_objectives
    
    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Find non-dominated solutions"""
        is_pareto = np.ones(len(objectives), dtype=bool)
        
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if i != j and np.all(obj_j <= obj_i) and np.any(obj_j < obj_i):
                    is_pareto[i] = False
                    break
        
        return is_pareto


class SMOP5(TestProblem):
    """
    SMOP5: Many-objective sparse problem (5 objectives)
    f_i = sum((x - shift_i)^2) for i = 1, ..., 5
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 5):
        super().__init__(dimension, num_objectives, "SMOP5")
        # Define shifts for each objective
        self.shifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        objectives = []
        for shift in self.shifts:
            f = np.sum((solution - shift)**2)
            objectives.append(f)
        
        return np.array(objectives)
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Generate approximate Pareto front using uniform sampling on simplex"""
        # Use Dirichlet distribution for many-objective problems
        alpha = np.ones(self.num_objectives)
        points = np.random.dirichlet(alpha, num_points)
        
        # Scale appropriately
        pareto_front = points * self.dimension
        
        return pareto_front


class SMOP6(TestProblem):
    """
    SMOP6: Deceptive sparse problem with local optima
    f1 = sum(x^2) + sum(sin(10*pi*x))
    f2 = sum((x-1)^2) + sum(cos(10*pi*x))
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 2):
        super().__init__(dimension, num_objectives, "SMOP6")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        f1 = np.sum(solution**2) + np.sum(np.sin(10 * np.pi * solution))
        f2 = np.sum((solution - 1)**2) + np.sum(np.cos(10 * np.pi * solution))
        
        return np.array([f1, f2])
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Approximate Pareto front for deceptive problem"""
        return self._approximate_pareto_front(num_points)
    
    def _approximate_pareto_front(self, num_points: int) -> np.ndarray:
        """Generate approximate Pareto front by sampling and filtering"""
        objectives = []
        
        # Sample more points to get good approximation
        for _ in range(num_points * 20):
            # Generate sparse solutions
            x = np.random.uniform(0, 1, self.dimension)
            # Apply sparsity
            sparsity = np.random.uniform(0.05, 0.2)  # 5-20% of variables active
            num_active = int(sparsity * self.dimension)
            if num_active > 0:
                active_indices = np.random.choice(self.dimension, num_active, replace=False)
                mask = np.zeros(self.dimension)
                mask[active_indices] = 1
                x = x * mask
            
            obj = self.evaluate(x)
            objectives.append(obj)
        
        objectives = np.array(objectives)
        
        # Find non-dominated solutions
        is_pareto = self._find_pareto_front(objectives)
        pareto_objectives = objectives[is_pareto]
        
        # Return requested number of points
        if len(pareto_objectives) > num_points:
            indices = np.random.choice(len(pareto_objectives), num_points, replace=False)
            return pareto_objectives[indices]
        else:
            return pareto_objectives
    
    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Find non-dominated solutions"""
        is_pareto = np.ones(len(objectives), dtype=bool)
        
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if i != j and np.all(obj_j <= obj_i) and np.any(obj_j < obj_i):
                    is_pareto[i] = False
                    break
        
        return is_pareto


class SMOP7(TestProblem):
    """
    SMOP7: High-dimensional many-objective problem
    f_i = sum((x - c_i)^2) + noise_i for i = 1, ..., M
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 8):
        super().__init__(dimension, num_objectives, "SMOP7")
        # Define centers for each objective
        self.centers = np.linspace(0, 1, num_objectives)
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        objectives = []
        for center in self.centers:
            f = np.sum((solution - center)**2)
            # Add small random noise to increase difficulty
            noise = 0.01 * np.random.normal()
            objectives.append(f + noise)
        
        return np.array(objectives)
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Generate approximate Pareto front for many-objective problem"""
        # Use uniform sampling on (M-1)-simplex
        alpha = np.ones(self.num_objectives)
        points = np.random.dirichlet(alpha, num_points)
        
        # Scale by dimension
        pareto_front = points * self.dimension
        
        return pareto_front


class SMOP8(TestProblem):
    """
    SMOP8: Mixed variable types and constraints
    f1 = sum(x^2) + constraint_penalty
    f2 = sum((x-1)^2) + constraint_penalty
    """
    
    def __init__(self, dimension: int = 100, num_objectives: int = 2):
        super().__init__(dimension, num_objectives, "SMOP8")
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        self._count_evaluation()
        
        # Basic objectives
        f1 = np.sum(solution**2)
        f2 = np.sum((solution - 1)**2)
        
        # Add constraint penalty
        penalty = self._constraint_penalty(solution)
        
        return np.array([f1 + penalty, f2 + penalty])
    
    def _constraint_penalty(self, solution: np.ndarray) -> float:
        """Add penalty for constraint violation"""
        # Example constraint: sum of active variables should be <= D/2
        num_active = np.sum(solution > 1e-6)
        max_active = self.dimension // 2
        
        if num_active > max_active:
            penalty = 1000 * (num_active - max_active)
        else:
            penalty = 0
        
        return penalty
    
    def get_true_pareto_front(self, num_points: int = 10000) -> np.ndarray:
        """Approximate Pareto front considering constraints"""
        objectives = []
        
        # Generate feasible solutions
        for _ in range(num_points * 10):
            # Generate sparse solution respecting constraints
            max_active = self.dimension // 2
            num_active = np.random.randint(1, max_active + 1)
            
            solution = np.zeros(self.dimension)
            active_indices = np.random.choice(self.dimension, num_active, replace=False)
            solution[active_indices] = np.random.uniform(0, 1, num_active)
            
            obj = self.evaluate(solution)
            objectives.append(obj)
        
        objectives = np.array(objectives)
        
        # Find non-dominated solutions
        is_pareto = self._find_pareto_front(objectives)
        pareto_objectives = objectives[is_pareto]
        
        # Return requested number of points
        if len(pareto_objectives) > num_points:
            indices = np.random.choice(len(pareto_objectives), num_points, replace=False)
            return pareto_objectives[indices]
        else:
            return pareto_objectives
    
    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Find non-dominated solutions"""
        is_pareto = np.ones(len(objectives), dtype=bool)
        
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if i != j and np.all(obj_j <= obj_i) and np.any(obj_j < obj_i):
                    is_pareto[i] = False
                    break
        
        return is_pareto


# Factory function to create problems
def create_smop_problem(problem_name: str, dimension: int, num_objectives: int) -> TestProblem:
    """
    Factory function to create SMOP problems
    
    Args:
        problem_name: Name of the problem (e.g., "SMOP1", "SMOP2", etc.)
        dimension: Problem dimension
        num_objectives: Number of objectives
        
    Returns:
        Instantiated problem object
    """
    problem_classes = {
        "SMOP1": SMOP1,
        "SMOP2": SMOP2, 
        "SMOP3": SMOP3,
        "SMOP4": SMOP4,
        "SMOP5": SMOP5,
        "SMOP6": SMOP6,
        "SMOP7": SMOP7,
        "SMOP8": SMOP8
    }
    
    if problem_name not in problem_classes:
        raise ValueError(f"Unknown problem: {problem_name}. Available: {list(problem_classes.keys())}")
    
    problem_class = problem_classes[problem_name]
    return problem_class(dimension, num_objectives) 