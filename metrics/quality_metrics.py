"""
Quality metrics for multi-objective optimization
Implements IGD, GD, Hypervolume, and other standard metrics
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
import warnings


class QualityMetrics:
    """Collection of quality metrics for multi-objective optimization"""
    
    @staticmethod
    def igd(pareto_front: np.ndarray, true_pareto_front: np.ndarray) -> float:
        """
        Calculate Inverted Generational Distance (IGD)
        
        IGD measures the average distance from true Pareto front points 
        to the closest points in the obtained Pareto front.
        Lower values are better.
        
        Args:
            pareto_front: Obtained Pareto front (n_points, n_objectives)
            true_pareto_front: True Pareto front (m_points, n_objectives)
            
        Returns:
            IGD value (float)
        """
        if len(pareto_front) == 0:
            return float('inf')
        
        if len(true_pareto_front) == 0:
            return float('inf')
        
        # Calculate distances from each true Pareto point to obtained front
        distances = cdist(true_pareto_front, pareto_front, metric='euclidean')
        
        # Find minimum distance for each true Pareto point
        min_distances = np.min(distances, axis=1)
        
        # Return average minimum distance
        return np.mean(min_distances)
    
    @staticmethod
    def gd(pareto_front: np.ndarray, true_pareto_front: np.ndarray) -> float:
        """
        Calculate Generational Distance (GD)
        
        GD measures the average distance from obtained Pareto front points
        to the closest points in the true Pareto front.
        Lower values are better.
        
        Args:
            pareto_front: Obtained Pareto front (n_points, n_objectives)
            true_pareto_front: True Pareto front (m_points, n_objectives)
            
        Returns:
            GD value (float)
        """
        if len(pareto_front) == 0:
            return float('inf')
        
        if len(true_pareto_front) == 0:
            return float('inf')
        
        # Calculate distances from each obtained point to true Pareto front
        distances = cdist(pareto_front, true_pareto_front, metric='euclidean')
        
        # Find minimum distance for each obtained point
        min_distances = np.min(distances, axis=1)
        
        # Return average minimum distance
        return np.mean(min_distances)
    
    @staticmethod
    def hypervolume_2d(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Calculate 2D Hypervolume using the standard algorithm
        
        Args:
            pareto_front: Obtained Pareto front (n_points, 2)
            reference_point: Reference point (2,)
            
        Returns:
            Hypervolume value (float)
        """
        if len(pareto_front) == 0:
            return 0.0
        
        if pareto_front.shape[1] != 2:
            raise ValueError("This implementation only supports 2D objectives")
        
        # Sort points by first objective
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        # Calculate hypervolume
        hv = 0.0
        prev_x = reference_point[0]
        
        for point in sorted_front:
            if point[0] > prev_x:  # Only consider non-dominated points
                width = point[0] - prev_x
                height = reference_point[1] - point[1]
                if height > 0:  # Only positive contributions
                    hv += width * height
                prev_x = point[0]
        
        return hv
    
    @staticmethod
    def spacing(pareto_front: np.ndarray) -> float:
        """
        Calculate Spacing metric - measures uniformity of distribution
        
        Args:
            pareto_front: Obtained Pareto front (n_points, n_objectives)
            
        Returns:
            Spacing value (float) - lower is better
        """
        if len(pareto_front) <= 1:
            return 0.0
        
        # Calculate distances between all pairs of points
        distances = cdist(pareto_front, pareto_front, metric='euclidean')
        
        # For each point, find distance to nearest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Calculate mean and standard deviation
        mean_distance = np.mean(min_distances)
        spacing_value = np.sqrt(np.mean((min_distances - mean_distance)**2))
        
        return spacing_value
    
    @staticmethod
    def calculate_sparsity_metrics(population: List) -> dict:
        """
        Calculate sparsity-related metrics for sparse optimization
        
        Args:
            population: List of individuals with .mask attribute
            
        Returns:
            Dictionary with sparsity metrics
        """
        if not population:
            return {
                'mean_sparsity': 0.0,
                'std_sparsity': 0.0,
                'min_sparsity': 0,
                'max_sparsity': 0,
                'sparsity_ratio': 0.0
            }
        
        # Extract sparsity information
        sparsities = []
        total_variables = 0
        
        for ind in population:
            if hasattr(ind, 'mask'):
                sparsity = np.sum(ind.mask)
                sparsities.append(sparsity)
                total_variables = len(ind.mask)
            elif hasattr(ind, 'solution'):
                # Count non-zero elements
                sparsity = np.sum(np.abs(ind.solution) > 1e-10)
                sparsities.append(sparsity)
                total_variables = len(ind.solution)
        
        sparsities = np.array(sparsities)
        
        return {
            'mean_sparsity': float(np.mean(sparsities)),
            'std_sparsity': float(np.std(sparsities)),
            'min_sparsity': int(np.min(sparsities)),
            'max_sparsity': int(np.max(sparsities)),
            'sparsity_ratio': float(np.mean(sparsities) / total_variables) if total_variables > 0 else 0.0
        }


class StatisticalTests:
    """Statistical tests for comparing algorithm performance"""
    
    @staticmethod
    def wilcoxon_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Perform Wilcoxon rank-sum test
        
        Args:
            data1: Results from algorithm 1 (n_runs,)
            data2: Results from algorithm 2 (n_runs,)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        try:
            statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
            
            # Determine significance symbol
            if p_value < alpha:
                if np.median(data1) < np.median(data2):
                    symbol = '+'  # Algorithm 1 is significantly better
                else:
                    symbol = '-'  # Algorithm 1 is significantly worse
            else:
                symbol = '='  # No significant difference
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha,
                'symbol': symbol,
                'alpha': alpha
            }
        
        except Exception as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'symbol': '=',
                'alpha': alpha
            }
    
    @staticmethod
    def compare_algorithms(results1: List[dict], results2: List[dict], 
                          metric_name: str = 'igd') -> dict:
        """
        Compare two algorithms using statistical tests
        
        Args:
            results1: Results from algorithm 1 (list of result dictionaries)
            results2: Results from algorithm 2 (list of result dictionaries)
            metric_name: Name of the metric to compare
            
        Returns:
            Comparison results
        """
        # Extract metric values
        values1 = np.array([r.get(metric_name, np.inf) for r in results1])
        values2 = np.array([r.get(metric_name, np.inf) for r in results2])
        
        # Remove invalid values
        valid_mask1 = np.isfinite(values1)
        valid_mask2 = np.isfinite(values2)
        values1 = values1[valid_mask1]
        values2 = values2[valid_mask2]
        
        if len(values1) == 0 or len(values2) == 0:
            return {
                'mean1': np.nan,
                'std1': np.nan,
                'mean2': np.nan,
                'std2': np.nan,
                'test_result': {'symbol': '='}
            }
        
        # Calculate statistics
        mean1, std1 = np.mean(values1), np.std(values1)
        mean2, std2 = np.mean(values2), np.std(values2)
        
        # Perform statistical test
        test_result = StatisticalTests.wilcoxon_test(values1, values2)
        
        return {
            'mean1': mean1,
            'std1': std1,
            'mean2': mean2,
            'std2': std2,
            'test_result': test_result
        }


class MetricsCalculator:
    """Helper class to calculate all metrics for a given result"""
    
    def __init__(self, true_pareto_front: Optional[np.ndarray] = None):
        self.true_pareto_front = true_pareto_front
    
    def calculate_all_metrics(self, pareto_front: np.ndarray, 
                             population: List = None,
                             reference_point: Optional[np.ndarray] = None) -> dict:
        """
        Calculate all available metrics for a given result
        
        Args:
            pareto_front: Obtained Pareto front
            population: Full population (for sparsity metrics)
            reference_point: Reference point for hypervolume
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Quality metrics (require true Pareto front)
        if self.true_pareto_front is not None:
            metrics['igd'] = QualityMetrics.igd(pareto_front, self.true_pareto_front)
            metrics['gd'] = QualityMetrics.gd(pareto_front, self.true_pareto_front)
        
        # Diversity metrics
        metrics['spacing'] = QualityMetrics.spacing(pareto_front)
        
        # Hypervolume (for 2D problems)
        if pareto_front.shape[1] == 2 and reference_point is not None:
            metrics['hypervolume'] = QualityMetrics.hypervolume_2d(pareto_front, reference_point)
        
        # Basic statistics
        metrics['num_solutions'] = len(pareto_front)
        if len(pareto_front) > 0:
            metrics['mean_objectives'] = np.mean(pareto_front, axis=0).tolist()
            metrics['std_objectives'] = np.std(pareto_front, axis=0).tolist()
            metrics['min_objectives'] = np.min(pareto_front, axis=0).tolist()
            metrics['max_objectives'] = np.max(pareto_front, axis=0).tolist()
        
        # Sparsity metrics
        if population is not None:
            sparsity_metrics = QualityMetrics.calculate_sparsity_metrics(population)
            metrics.update(sparsity_metrics)
        
        return metrics
    
    def set_true_pareto_front(self, true_pareto_front: np.ndarray):
        """Set the true Pareto front for quality metrics"""
        self.true_pareto_front = true_pareto_front


def find_pareto_front(objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the non-dominated solutions (Pareto front) from a set of objectives
    
    Args:
        objectives: Array of objective values (n_points, n_objectives)
        
    Returns:
        Tuple of (pareto_front_objectives, pareto_indices)
    """
    n_points = len(objectives)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if j dominates i
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_pareto[i] = False
                    break
    
    pareto_indices = np.where(is_pareto)[0]
    pareto_front = objectives[pareto_indices]
    
    return pareto_front, pareto_indices


def normalize_objectives(objectives: np.ndarray, 
                        reference_min: Optional[np.ndarray] = None,
                        reference_max: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize objectives to [0, 1] range
    
    Args:
        objectives: Objective values to normalize (n_points, n_objectives)
        reference_min: Reference minimum values (if None, use min of objectives)
        reference_max: Reference maximum values (if None, use max of objectives)
        
    Returns:
        Normalized objectives
    """
    if reference_min is None:
        reference_min = np.min(objectives, axis=0)
    if reference_max is None:
        reference_max = np.max(objectives, axis=0)
    
    # Avoid division by zero
    range_vals = reference_max - reference_min
    range_vals[range_vals == 0] = 1.0
    
    normalized = (objectives - reference_min) / range_vals
    
    return normalized 