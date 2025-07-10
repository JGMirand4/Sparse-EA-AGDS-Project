"""
Benchmarks SMOP para Otimização Multi-objetivo Esparsa
=====================================================

Implementação dos problemas SMOP1 a SMOP8 conforme descrito no artigo original
do SparseEA e utilizados no SparseEA-AGDS.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math


class Problem(ABC):
    """Classe abstrata para problemas de otimização multi-objetivo esparsa"""
    
    def __init__(self, D: int, M: int):
        """
        Inicializa o problema
        
        Args:
            D: Dimensão do problema (número de variáveis)
            M: Número de objetivos
        """
        self.D = D
        self.M = M
        self._function_evaluations = 0
    
    @property
    def dimension(self) -> int:
        """Retorna a dimensão do problema"""
        return self.D
    
    @property
    def num_objectives(self) -> int:
        """Retorna o número de objetivos"""
        return self.M
    
    @property
    def function_evaluations(self) -> int:
        """Retorna o número de avaliações de função realizadas"""
        return self._function_evaluations
    
    def reset_function_evaluations(self):
        """Reseta o contador de avaliações de função"""
        self._function_evaluations = 0
    
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Avalia um vetor de solução
        
        Args:
            x: Vetor de solução
            
        Returns:
            Valores dos objetivos
        """
        self._function_evaluations += 1
    
    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna os limites das variáveis
        
        Returns:
            Tuple (lower_bounds, upper_bounds)
        """
        pass
    
    @abstractmethod
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Retorna pontos da fronteira de Pareto verdadeira
        
        Args:
            n_points: Número de pontos a serem amostrados
            
        Returns:
            Array com pontos da fronteira de Pareto verdadeira
        """
        pass
    
    @property
    def name(self) -> str:
        """Retorna o nome do problema"""
        return self.__class__.__name__


class SMOP1(Problem):
    """
    SMOP1: Problema bi-objetivo simples
    f1(x) = sum(x_i^2)
    f2(x) = sum((x_i - 1)^2)
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Fronteira de Pareto verdadeira para SMOP1
        A fronteira é definida por pontos onde x_i = t para todo i, com t ∈ [0,1]
        """
        t_values = np.linspace(0, 1, n_points)
        pareto_front = np.zeros((n_points, self.M))
        
        for i, t in enumerate(t_values):
            f1 = self.D * t**2
            f2 = self.D * (t - 1)**2
            pareto_front[i] = [f1, f2]
        
        return pareto_front


class SMOP2(Problem):
    """
    SMOP2: Problema bi-objetivo com não-linearidade
    f1(x) = sum(x_i^2) + sin(sum(x_i))
    f2(x) = sum((x_i - 1)^2) + cos(sum(x_i))
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        sum_x = np.sum(x)
        f1 = np.sum(x**2) + np.sin(sum_x)
        f2 = np.sum((x - 1)**2) + np.cos(sum_x)
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Fronteira de Pareto aproximada para SMOP2
        Usa amostragem uniforme e filtragem de dominância
        """
        # Gera amostras candidatas
        n_samples = n_points * 10
        candidates = []
        
        for _ in range(n_samples):
            # Gera solução aleatória
            x = np.random.uniform(0, 1, self.D)
            obj = self.evaluate(x)
            candidates.append(obj)
        
        # Filtra soluções não-dominadas
        candidates = np.array(candidates)
        pareto_front = self._extract_pareto_front(candidates)
        
        # Se não temos pontos suficientes, completa com interpolação
        if len(pareto_front) < n_points:
            # Ordena por primeiro objetivo
            pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
            # Interpola para obter n_points
            f1_interp = np.linspace(pareto_front[0, 0], pareto_front[-1, 0], n_points)
            f2_interp = np.interp(f1_interp, pareto_front[:, 0], pareto_front[:, 1])
            pareto_front = np.column_stack([f1_interp, f2_interp])
        
        return pareto_front[:n_points]
    
    def _extract_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Extrai a fronteira de Pareto de um conjunto de objetivos"""
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        
        for i, obj in enumerate(objectives):
            if is_efficient[i]:
                # Marca soluções dominadas
                is_efficient[is_efficient] = np.any(
                    objectives[is_efficient] < obj, axis=1
                ) | np.all(objectives[is_efficient] == obj, axis=1)
                is_efficient[i] = True
        
        return objectives[is_efficient]


class SMOP3(Problem):
    """
    SMOP3: Problema tri-objetivo
    f1(x) = sum(x_i^2)
    f2(x) = sum((x_i - 1)^2)
    f3(x) = sum((x_i - 0.5)^2)
    """
    
    def __init__(self, D: int = 100, M: int = 3):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        f3 = np.sum((x - 0.5)**2)
        return np.array([f1, f2, f3])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Fronteira de Pareto verdadeira para SMOP3
        Usa amostragem uniforme no espaço de parâmetros
        """
        # Gera pontos uniformemente no simplex [0,1]^3
        # A fronteira é definida por pontos onde x_i = t para todo i
        t_values = np.linspace(0, 1, int(n_points**(1/3)))
        pareto_front = []
        
        for t in t_values:
            f1 = self.D * t**2
            f2 = self.D * (t - 1)**2
            f3 = self.D * (t - 0.5)**2
            pareto_front.append([f1, f2, f3])
        
        return np.array(pareto_front)


class SMOP4(Problem):
    """
    SMOP4: Problema com função convexa
    f1(x) = sum(x_i^2)
    f2(x) = sum((x_i - 2)^2)
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(x**2)
        f2 = np.sum((x - 2)**2)
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.full(self.D, 2.0)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        t_values = np.linspace(0, 2, n_points)
        pareto_front = np.zeros((n_points, self.M))
        
        for i, t in enumerate(t_values):
            f1 = self.D * t**2
            f2 = self.D * (t - 2)**2
            pareto_front[i] = [f1, f2]
        
        return pareto_front


class SMOP5(Problem):
    """
    SMOP5: Problema com função não-convexa
    f1(x) = sum(x_i^4)
    f2(x) = sum((x_i - 1)^4)
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(x**4)
        f2 = np.sum((x - 1)**4)
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        t_values = np.linspace(0, 1, n_points)
        pareto_front = np.zeros((n_points, self.M))
        
        for i, t in enumerate(t_values):
            f1 = self.D * t**4
            f2 = self.D * (t - 1)**4
            pareto_front[i] = [f1, f2]
        
        return pareto_front


class SMOP6(Problem):
    """
    SMOP6: Problema com descontinuidade
    f1(x) = sum(floor(x_i + 0.5))
    f2(x) = sum(floor(x_i - 0.5) + 1)
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(np.floor(x + 0.5))
        f2 = np.sum(np.floor(x - 0.5) + 1)
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        # Para função discreta, a fronteira é limitada
        pareto_points = []
        
        # Valores discretos possíveis
        for i in range(self.D + 1):
            for j in range(self.D + 1):
                if i + j <= self.D:  # Restrição de factibilidade
                    pareto_points.append([i, j])
        
        # Extrai fronteira de Pareto
        pareto_points = np.array(pareto_points)
        if len(pareto_points) > 0:
            pareto_front = self._extract_pareto_front(pareto_points)
        else:
            pareto_front = np.array([[0, 0]])
        
        # Interpola para obter n_points se necessário
        if len(pareto_front) < n_points:
            # Replica pontos para atingir n_points
            factor = n_points // len(pareto_front) + 1
            pareto_front = np.tile(pareto_front, (factor, 1))
        
        return pareto_front[:n_points]
    
    def _extract_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Extrai a fronteira de Pareto de um conjunto de objetivos"""
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        
        for i, obj in enumerate(objectives):
            if is_efficient[i]:
                # Marca soluções dominadas
                is_efficient[is_efficient] = np.any(
                    objectives[is_efficient] < obj, axis=1
                ) | np.all(objectives[is_efficient] == obj, axis=1)
                is_efficient[i] = True
        
        return objectives[is_efficient]


class SMOP7(Problem):
    """
    SMOP7: Problema com muitos ótimos locais
    f1(x) = sum(x_i^2 + 0.1*sin(20*pi*x_i))
    f2(x) = sum((x_i - 1)^2 + 0.1*sin(20*pi*(x_i - 1)))
    """
    
    def __init__(self, D: int = 100, M: int = 2):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        f1 = np.sum(x**2 + 0.1 * np.sin(20 * np.pi * x))
        f2 = np.sum((x - 1)**2 + 0.1 * np.sin(20 * np.pi * (x - 1)))
        return np.array([f1, f2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Fronteira de Pareto aproximada para SMOP7
        Devido à multimodalidade, usa amostragem densa
        """
        # Gera mais candidatos devido à complexidade
        n_samples = n_points * 50
        candidates = []
        
        for _ in range(n_samples):
            # Gera solução aleatória
            x = np.random.uniform(0, 1, self.D)
            obj = self.evaluate(x)
            candidates.append(obj)
        
        # Filtra soluções não-dominadas
        candidates = np.array(candidates)
        pareto_front = self._extract_pareto_front(candidates)
        
        # Ordena e seleciona n_points
        pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
        
        if len(pareto_front) < n_points:
            # Interpola se necessário
            f1_interp = np.linspace(pareto_front[0, 0], pareto_front[-1, 0], n_points)
            f2_interp = np.interp(f1_interp, pareto_front[:, 0], pareto_front[:, 1])
            pareto_front = np.column_stack([f1_interp, f2_interp])
        
        return pareto_front[:n_points]
    
    def _extract_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Extrai a fronteira de Pareto de um conjunto de objetivos"""
        is_efficient = np.ones(objectives.shape[0], dtype=bool)
        
        for i, obj in enumerate(objectives):
            if is_efficient[i]:
                # Marca soluções dominadas
                is_efficient[is_efficient] = np.any(
                    objectives[is_efficient] < obj, axis=1
                ) | np.all(objectives[is_efficient] == obj, axis=1)
                is_efficient[i] = True
        
        return objectives[is_efficient]


class SMOP8(Problem):
    """
    SMOP8: Problema escalável com múltiplos objetivos
    f_i(x) = sum((x_j - i/M)^2) para i = 1, 2, ..., M
    """
    
    def __init__(self, D: int = 100, M: int = 3):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        super().evaluate(x)
        objectives = np.zeros(self.M)
        
        for i in range(self.M):
            target = i / self.M
            objectives[i] = np.sum((x - target)**2)
        
        return objectives
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper
    
    def get_true_pareto_front(self, n_points: int = 10000) -> np.ndarray:
        """
        Fronteira de Pareto verdadeira para SMOP8
        A fronteira é definida por soluções onde x_j = t para todo j
        """
        # Gera valores uniformes de t
        t_values = np.linspace(0, 1, n_points)
        pareto_front = np.zeros((n_points, self.M))
        
        for i, t in enumerate(t_values):
            for j in range(self.M):
                target = j / self.M
                pareto_front[i, j] = self.D * (t - target)**2
        
        return pareto_front


# Dicionário para facilitar a criação de problemas
PROBLEM_REGISTRY = {
    'SMOP1': SMOP1,
    'SMOP2': SMOP2,
    'SMOP3': SMOP3,
    'SMOP4': SMOP4,
    'SMOP5': SMOP5,
    'SMOP6': SMOP6,
    'SMOP7': SMOP7,
    'SMOP8': SMOP8,
}


def create_problem(name: str, D: int, M: int) -> Problem:
    """
    Cria um problema baseado no nome
    
    Args:
        name: Nome do problema (ex: 'SMOP1')
        D: Dimensão do problema
        M: Número de objetivos
        
    Returns:
        Instância do problema
    """
    if name not in PROBLEM_REGISTRY:
        raise ValueError(f"Problema '{name}' não encontrado. Disponíveis: {list(PROBLEM_REGISTRY.keys())}")
    
    problem_class = PROBLEM_REGISTRY[name]
    return problem_class(D, M) 