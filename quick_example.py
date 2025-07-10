#!/usr/bin/env python3
"""
Exemplo Rápido do SparseEA-AGDS
===============================

Este exemplo demonstra como usar o algoritmo SparseEA-AGDS de forma rápida e direta.
"""

import numpy as np
from sparse_ea_agds import SparseEAAGDS, SMOP1

def main():
    """Exemplo básico de uso do SparseEA-AGDS"""
    
    print("=== SparseEA-AGDS: Exemplo Rápido ===\n")
    
    # 1. Cria um problema de otimização
    print("1. Criando problema de teste (SMOP1)...")
    problem = SMOP1(D=10, M=2)  # 10 dimensões, 2 objetivos
    print(f"   - Dimensões: {problem.dimension}")
    print(f"   - Objetivos: {problem.num_objectives}")
    print(f"   - Limites: {problem.bounds}")
    
    # 2. Configura o algoritmo
    print("\n2. Configurando algoritmo...")
    algorithm = SparseEAAGDS(
        problem=problem,
        population_size=30,     # População pequena para execução rápida
        max_generations=50,     # Poucas gerações para teste
        Pc0=0.9,               # Probabilidade de crossover
        Pm0=0.1                # Probabilidade de mutação
    )
    
    # 3. Executa a otimização
    print("\n3. Executando otimização...")
    final_population = algorithm.run()
    
    # 4. Analisa os resultados
    print("\n4. Analisando resultados...")
    
    # Extrai dados da população final
    objectives = np.array([ind.objectives for ind in final_population])
    sparsity = np.array([np.sum(ind.mask) for ind in final_population])
    
    print(f"   - População final: {len(final_population)} indivíduos")
    print(f"   - Esparsidade média: {np.mean(sparsity):.2f}/{problem.dimension}")
    print(f"   - Esparsidade mínima: {np.min(sparsity)}")
    print(f"   - Esparsidade máxima: {np.max(sparsity)}")
    
    # Mostra os 5 melhores indivíduos
    print("\n5. Melhores soluções encontradas:")
    
    # Ordena por primeiro objetivo
    sorted_indices = np.argsort(objectives[:, 0])
    
    for i in range(min(5, len(final_population))):
        idx = sorted_indices[i]
        ind = final_population[idx]
        print(f"   Solução {i+1}:")
        print(f"     Objetivos: [{ind.objectives[0]:.4f}, {ind.objectives[1]:.4f}]")
        print(f"     Esparsidade: {np.sum(ind.mask)}/{len(ind.mask)}")
        print(f"     Variáveis ativas: {np.where(ind.mask == 1)[0].tolist()}")
        print(f"     Valores: {ind.solution[ind.mask == 1]}")
        print()
    
    # Encontra soluções não-dominadas
    non_dominated = find_non_dominated_solutions(final_population)
    print(f"6. Soluções não-dominadas: {len(non_dominated)}")
    
    if len(non_dominated) > 0:
        print("   Frente de Pareto:")
        for i, ind in enumerate(non_dominated[:3]):  # Mostra até 3
            print(f"     {i+1}. Objetivos: [{ind.objectives[0]:.4f}, {ind.objectives[1]:.4f}], "
                  f"Esparsidade: {np.sum(ind.mask)}")
    
    print("\n=== Exemplo concluído com sucesso! ===")
    return final_population


def find_non_dominated_solutions(population):
    """Encontra as soluções não-dominadas na população"""
    non_dominated = []
    
    for i, ind1 in enumerate(population):
        is_dominated = False
        for j, ind2 in enumerate(population):
            if i != j and dominates(ind2, ind1):
                is_dominated = True
                break
        
        if not is_dominated:
            non_dominated.append(ind1)
    
    return non_dominated


def dominates(ind1, ind2):
    """Verifica se ind1 domina ind2"""
    better_in_at_least_one = False
    for i in range(len(ind1.objectives)):
        if ind1.objectives[i] > ind2.objectives[i]:
            return False
        elif ind1.objectives[i] < ind2.objectives[i]:
            better_in_at_least_one = True
    return better_in_at_least_one


if __name__ == "__main__":
    main() 