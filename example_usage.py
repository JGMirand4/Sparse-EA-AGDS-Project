import numpy as np
import matplotlib.pyplot as plt
from sparse_ea_agds import SparseEAAGDS, Problem, Individual, SMOP1
from typing import List, Tuple
import time


class SMOP2(Problem):
    """Implementação do problema SMOP2 para teste"""
    
    def __init__(self, D: int = 20, M: int = 2):
        self.D = D
        self.M = M
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        SMOP2: Um problema mais complexo com múltiplos ótimos
        f1 = sum(x^2) + sin(sum(x))
        f2 = sum((x-0.5)^2) + cos(sum(x))
        """
        f1 = np.sum(x**2) + np.sin(np.sum(x))
        f2 = np.sum((x - 0.5)**2) + np.cos(np.sum(x))
        return np.array([f1, f2])
    
    @property
    def dimension(self) -> int:
        return self.D
    
    @property
    def num_objectives(self) -> int:
        return self.M
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper


class SMOP3(Problem):
    """Implementação do problema SMOP3 com 3 objetivos"""
    
    def __init__(self, D: int = 15, M: int = 3):
        self.D = D
        self.M = M
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        SMOP3: Problema com 3 objetivos
        f1 = sum(x^2)
        f2 = sum((x-1)^2)
        f3 = sum((x-0.5)^2)
        """
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        f3 = np.sum((x - 0.5)**2)
        return np.array([f1, f2, f3])
    
    @property
    def dimension(self) -> int:
        return self.D
    
    @property
    def num_objectives(self) -> int:
        return self.M
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper


def analyze_population(population: List[Individual], problem_name: str):
    """Analisa e exibe estatísticas da população final"""
    print(f"\n=== Análise da População Final - {problem_name} ===")
    
    # Extrai objetivos
    objectives = np.array([ind.objectives for ind in population])
    sparsity = np.array([np.sum(ind.mask) for ind in population])
    
    print(f"Número de indivíduos: {len(population)}")
    print(f"Número de objetivos: {len(objectives[0])}")
    print(f"Dimensão do problema: {len(population[0].dec)}")
    
    # Estatísticas de esparsidade
    print(f"\nEstatísticas de Esparsidade:")
    print(f"  Esparsidade média: {np.mean(sparsity):.2f}")
    print(f"  Esparsidade mínima: {np.min(sparsity)}")
    print(f"  Esparsidade máxima: {np.max(sparsity)}")
    print(f"  Desvio padrão: {np.std(sparsity):.2f}")
    
    # Estatísticas dos objetivos
    print(f"\nEstatísticas dos Objetivos:")
    for i in range(len(objectives[0])):
        print(f"  Objetivo {i+1}:")
        print(f"    Média: {np.mean(objectives[:, i]):.4f}")
        print(f"    Mínimo: {np.min(objectives[:, i]):.4f}")
        print(f"    Máximo: {np.max(objectives[:, i]):.4f}")
        print(f"    Desvio padrão: {np.std(objectives[:, i]):.4f}")
    
    # Encontra soluções não-dominadas
    non_dominated = find_non_dominated_solutions(population)
    print(f"\nSoluções não-dominadas encontradas: {len(non_dominated)}")
    
    return objectives, sparsity, non_dominated


def find_non_dominated_solutions(population: List[Individual]) -> List[Individual]:
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


def dominates(ind1: Individual, ind2: Individual) -> bool:
    """Verifica se ind1 domina ind2"""
    better_in_at_least_one = False
    for i in range(len(ind1.objectives)):
        if ind1.objectives[i] > ind2.objectives[i]:
            return False
        elif ind1.objectives[i] < ind2.objectives[i]:
            better_in_at_least_one = True
    return better_in_at_least_one


def plot_results(results: dict, save_plots: bool = True):
    """Plota os resultados dos diferentes problemas"""
    
    for problem_name, data in results.items():
        objectives = data['objectives']
        sparsity = data['sparsity']
        non_dominated = data['non_dominated']
        
        if len(objectives[0]) == 2:
            # Plot para 2 objetivos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Gráfico do espaço de objetivos
            ax1.scatter(objectives[:, 0], objectives[:, 1], 
                       c=sparsity, cmap='viridis', alpha=0.6, s=50)
            ax1.set_xlabel('Objetivo 1')
            ax1.set_ylabel('Objetivo 2')
            ax1.set_title(f'{problem_name} - Espaço de Objetivos')
            ax1.grid(True, alpha=0.3)
            
            # Destaca soluções não-dominadas
            if non_dominated:
                nd_objectives = np.array([ind.objectives for ind in non_dominated])
                ax1.scatter(nd_objectives[:, 0], nd_objectives[:, 1], 
                           c='red', marker='x', s=100, linewidths=2, 
                           label='Não-dominadas')
                ax1.legend()
            
            # Colorbar para esparsidade
            cbar = plt.colorbar(ax1.collections[0], ax=ax1)
            cbar.set_label('Esparsidade')
            
            # Histograma de esparsidade
            ax2.hist(sparsity, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Esparsidade')
            ax2.set_ylabel('Frequência')
            ax2.set_title(f'{problem_name} - Distribuição de Esparsidade')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'{problem_name.lower()}_results.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        elif len(objectives[0]) == 3:
            # Plot para 3 objetivos
            fig = plt.figure(figsize=(12, 5))
            
            # Gráfico 3D do espaço de objetivos
            ax1 = fig.add_subplot(121, projection='3d')
            scatter = ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                                c=sparsity, cmap='viridis', alpha=0.6, s=50)
            ax1.set_xlabel('Objetivo 1')
            ax1.set_ylabel('Objetivo 2')
            ax1.set_zlabel('Objetivo 3')
            ax1.set_title(f'{problem_name} - Espaço de Objetivos 3D')
            
            # Destaca soluções não-dominadas
            if non_dominated:
                nd_objectives = np.array([ind.objectives for ind in non_dominated])
                ax1.scatter(nd_objectives[:, 0], nd_objectives[:, 1], nd_objectives[:, 2], 
                           c='red', marker='x', s=100, linewidths=2)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5)
            cbar.set_label('Esparsidade')
            
            # Histograma de esparsidade
            ax2 = fig.add_subplot(122)
            ax2.hist(sparsity, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Esparsidade')
            ax2.set_ylabel('Frequência')
            ax2.set_title(f'{problem_name} - Distribuição de Esparsidade')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'{problem_name.lower()}_results.png', dpi=300, bbox_inches='tight')
            plt.show()


def run_comparative_study():
    """Executa um estudo comparativo em diferentes problemas"""
    
    print("=== Estudo Comparativo do SparseEA-AGDS ===\n")
    
    # Define problemas para teste
    problems = {
        'SMOP1': SMOP1(D=10, M=2),
        'SMOP2': SMOP2(D=15, M=2),
        'SMOP3': SMOP3(D=12, M=3)
    }
    
    # Parâmetros do algoritmo
    algorithm_params = {
        'population_size': 50,
        'max_generations': 300,
        'Pc0': 0.9,
        'Pm0': 0.1
    }
    
    results = {}
    
    for problem_name, problem in problems.items():
        print(f"\n--- Executando {problem_name} ---")
        print(f"Dimensão: {problem.dimension}")
        print(f"Objetivos: {problem.num_objectives}")
        
        # Executa algoritmo
        start_time = time.time()
        algorithm = SparseEAAGDS(problem=problem, **algorithm_params)
        final_population = algorithm.run()
        execution_time = time.time() - start_time
        
        print(f"Tempo de execução: {execution_time:.2f} segundos")
        
        # Analisa resultados
        objectives, sparsity, non_dominated = analyze_population(final_population, problem_name)
        
        results[problem_name] = {
            'objectives': objectives,
            'sparsity': sparsity,
            'non_dominated': non_dominated,
            'execution_time': execution_time,
            'final_population': final_population
        }
    
    # Plota resultados
    plot_results(results)
    
    # Resumo comparativo
    print("\n=== Resumo Comparativo ===")
    print(f"{'Problema':<10} {'Tempo(s)':<10} {'Não-Dom.':<10} {'Espars.Média':<12} {'Melhor_F1':<12}")
    print("-" * 60)
    
    for problem_name, data in results.items():
        best_f1 = np.min(data['objectives'][:, 0])
        avg_sparsity = np.mean(data['sparsity'])
        print(f"{problem_name:<10} {data['execution_time']:<10.2f} {len(data['non_dominated']):<10} "
              f"{avg_sparsity:<12.2f} {best_f1:<12.4f}")
    
    return results


def demonstrate_parameter_sensitivity():
    """Demonstra a sensibilidade do algoritmo aos parâmetros"""
    
    print("\n=== Análise de Sensibilidade dos Parâmetros ===")
    
    problem = SMOP1(D=10, M=2)
    base_params = {
        'problem': problem,
        'population_size': 50,
        'max_generations': 200
    }
    
    # Testa diferentes valores de Pc0 e Pm0
    param_combinations = [
        {'Pc0': 0.7, 'Pm0': 0.05, 'name': 'Baixo'},
        {'Pc0': 0.9, 'Pm0': 0.1, 'name': 'Médio'},
        {'Pc0': 0.95, 'Pm0': 0.2, 'name': 'Alto'}
    ]
    
    print(f"{'Configuração':<12} {'Tempo(s)':<10} {'Não-Dom.':<10} {'Espars.Média':<12} {'Melhor_F1':<12}")
    print("-" * 60)
    
    for params in param_combinations:
        config_name = params.pop('name')
        
        algorithm = SparseEAAGDS(**base_params, **params)
        
        start_time = time.time()
        final_population = algorithm.run()
        execution_time = time.time() - start_time
        
        objectives = np.array([ind.objectives for ind in final_population])
        sparsity = np.array([np.sum(ind.mask) for ind in final_population])
        non_dominated = find_non_dominated_solutions(final_population)
        
        best_f1 = np.min(objectives[:, 0])
        avg_sparsity = np.mean(sparsity)
        
        print(f"{config_name:<12} {execution_time:<10.2f} {len(non_dominated):<10} "
              f"{avg_sparsity:<12.2f} {best_f1:<12.4f}")


def main():
    """Função principal que executa todos os exemplos"""
    
    print("SparseEA-AGDS - Demonstração de Uso")
    print("=" * 50)
    
    # Exemplo básico
    print("\n1. Exemplo Básico - SMOP1")
    problem = SMOP1(D=10, M=2)
    algorithm = SparseEAAGDS(
        problem=problem,
        population_size=30,
        max_generations=100
    )
    
    final_population = algorithm.run()
    analyze_population(final_population, "SMOP1 - Exemplo Básico")
    
    # Estudo comparativo
    print("\n2. Estudo Comparativo")
    comparative_results = run_comparative_study()
    
    # Análise de sensibilidade
    print("\n3. Análise de Sensibilidade")
    demonstrate_parameter_sensitivity()
    
    print("\n=== Demonstração Concluída ===")
    print("Arquivos de gráficos foram salvos como PNG.")
    print("Para problemas customizados, herde da classe Problem e implemente os métodos abstratos.")


if __name__ == "__main__":
    main() 