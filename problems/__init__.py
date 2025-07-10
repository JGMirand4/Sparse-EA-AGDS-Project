"""
Optimization problems package for SparseEA-AGDS
"""

from .base import Problem, TestProblem
from .smop import (
    SMOP1, SMOP2, SMOP3, SMOP4, SMOP5, SMOP6, SMOP7, SMOP8,
    create_smop_problem
)

__all__ = [
    'Problem', 'TestProblem',
    'SMOP1', 'SMOP2', 'SMOP3', 'SMOP4', 'SMOP5', 'SMOP6', 'SMOP7', 'SMOP8',
    'create_smop_problem'
] 