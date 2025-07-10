import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from scipy.spatial.distance import cdist


@dataclass
class Individual:
    """Estrutura de um indivíduo na população"""
    dec: np.ndarray = field(default_factory=lambda: np.array([]))  # Variáveis reais
    mask: np.ndarray = field(default_factory=lambda: np.array([]))  # Máscara binária
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))  # Valores dos objetivos
    rank: int = float('inf')  # Rank não-dominado
    crowding_distance: float = 0.0  # Distância de crowding
    
    @property
    def solution(self) -> np.ndarray:
        """Retorna a solução X = dec * mask"""
        return self.dec * self.mask
    
    def __post_init__(self):
        if len(self.dec) > 0 and len(self.mask) == 0:
            self.mask = np.zeros_like(self.dec, dtype=int)


class Problem(ABC):
    """Classe abstrata para definir problemas de otimização"""
    
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Avalia um indivíduo e retorna os valores dos objetivos"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Retorna a dimensão do problema"""
        pass
    
    @property
    @abstractmethod
    def num_objectives(self) -> int:
        """Retorna o número de objetivos"""
        pass
    
    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna os limites inferior e superior das variáveis"""
        pass


class SMOP1(Problem):
    """Implementação do problema SMOP1 para teste"""
    
    def __init__(self, D: int = 10, M: int = 2):
        self.D = D
        self.M = M
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Avalia SMOP1: minimizar f1 = sum(x^2) e f2 = sum((x-1)^2)"""
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
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


class SparseEAAGDS:
    """Implementação do algoritmo SparseEA-AGDS"""
    
    def __init__(self, 
                 problem: Problem,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 Pc0: float = 0.9,
                 Pm0: float = 0.1,
                 eta_c: float = 20.0,
                 eta_m: float = 20.0):
        self.problem = problem
        self.N = population_size
        self.max_generations = max_generations
        self.Pc0 = Pc0
        self.Pm0 = Pm0
        self.eta_c = eta_c  # Parâmetro do SBX
        self.eta_m = eta_m  # Parâmetro da mutação polinomial
        self.D = problem.dimension
        self.M = problem.num_objectives
        self.lower_bound, self.upper_bound = problem.bounds
        
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """Implementa a ordenação rápida não-dominada"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(population[j], population[i]):
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
        
        return fronts[:-1]  # Remove a última frente vazia
    
    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Verifica se ind1 domina ind2"""
        better_in_at_least_one = False
        for i in range(len(ind1.objectives)):
            if ind1.objectives[i] > ind2.objectives[i]:
                return False
            elif ind1.objectives[i] < ind2.objectives[i]:
                better_in_at_least_one = True
        return better_in_at_least_one
    
    def binary_tournament_selection(self, scores: np.ndarray, k: int) -> List[int]:
        """Seleção por torneio binário baseada nas pontuações"""
        selected = []
        for _ in range(k):
            # Escolhe duas dimensões aleatórias
            i, j = random.sample(range(len(scores)), 2)
            # A menor pontuação vence (mais importante)
            if scores[i] < scores[j]:
                selected.append(i)
            else:
                selected.append(j)
        return selected
    
    def calculate_initial_scores(self) -> np.ndarray:
        """Calcula as pontuações iniciais das variáveis"""
        # Cria matriz D x D de variáveis reais aleatórias
        dec_matrix = np.random.rand(self.D, self.D)
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
        fronts = self.fast_non_dominated_sort(G)
        
        # Calcula pontuações iniciais
        scores = np.zeros(self.D)
        num_runs = 10  # Número de repetições para acumular pontuações
        
        for run in range(num_runs):
            # Recria G com novos valores aleatórios
            dec_matrix = np.random.rand(self.D, self.D)
            for i in range(self.D):
                G[i].dec = dec_matrix[i]
                G[i].objectives = self.problem.evaluate(G[i].solution)
            
            # Reordena
            fronts = self.fast_non_dominated_sort(G)
            
            # Acumula pontuações (rank da frente)
            for front_idx, front in enumerate(fronts):
                for ind_idx in front:
                    scores[ind_idx] += front_idx + 1
        
        return scores
    
    def initialize_population(self, initial_scores: np.ndarray) -> List[Individual]:
        """Inicializa a população usando as pontuações iniciais"""
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
                active_positions = self.binary_tournament_selection(initial_scores, num_active)
                ind.mask[active_positions] = 1
            
            # Avalia o indivíduo
            ind.objectives = self.problem.evaluate(ind.solution)
            population.append(ind)
        
        return population
    
    def simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implementa o Simulated Binary Crossover (SBX)"""
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
                
                # Aplica limites
                child1[i] = np.clip(child1[i], self.lower_bound[i], self.upper_bound[i])
                child2[i] = np.clip(child2[i], self.lower_bound[i], self.upper_bound[i])
        
        return child1, child2
    
    def polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Implementa a mutação polinomial"""
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
    
    def adaptive_genetic_operator(self, population: List[Individual]) -> List[np.ndarray]:
        """Implementa o operador genético adaptativo"""
        # Calcula ranks
        fronts = self.fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks)
        
        offspring_dec = []
        
        for i in range(len(population)):
            # Calcula probabilidade de seleção (Equação 5)
            Ps_i = (max_rank - ranks[i] + 1) / max_rank if max_rank > 0 else 1.0
            
            # Calcula probabilidades adaptativas (Equações 6 e 7)
            Pc_i = self.Pc0 * Ps_i
            Pm_i = self.Pm0 * Ps_i
            
            # Operações genéticas
            child_dec = population[i].dec.copy()
            
            # Crossover
            if np.random.rand() < Pc_i:
                partner_idx = np.random.randint(0, len(population))
                child_dec, _ = self.simulated_binary_crossover(
                    population[i].dec, population[partner_idx].dec
                )
            
            # Mutação
            if np.random.rand() < Pm_i:
                child_dec = self.polynomial_mutation(child_dec)
            
            offspring_dec.append(child_dec)
        
        return offspring_dec
    
    def dynamic_scoring_mechanism(self, population: List[Individual]) -> List[np.ndarray]:
        """Implementa o mecanismo de pontuação dinâmica"""
        # Calcula ranks da população
        fronts = self.fast_non_dominated_sort(population)
        ranks = [0] * len(population)
        for front_idx, front in enumerate(fronts):
            for ind_idx in front:
                ranks[ind_idx] = front_idx
        
        max_rank = max(ranks)
        
        # Calcula pontuação de camada (Equação 8)
        Sr = np.array([max_rank - r + 1 for r in ranks])
        
        # Cria matriz de masks
        mask_matrix = np.array([ind.mask for ind in population])
        
        # Calcula pontuação ponderada (Equação 9)
        sumS = Sr.T @ mask_matrix
        
        # Atualiza pontuação final da variável (Equação 10)
        maxS = np.max(sumS)
        S = maxS - sumS + 1
        
        # Gera masks dos filhos usando seleção por torneio
        offspring_masks = []
        for _ in range(len(population)):
            # Crossover de mask
            parent1_idx = np.random.randint(0, len(population))
            parent2_idx = np.random.randint(0, len(population))
            
            child_mask = np.zeros(self.D, dtype=int)
            
            # Seleciona dimensões baseado na pontuação S
            c = np.random.rand()
            num_active = int(c * self.D)
            if num_active > 0:
                active_positions = self.binary_tournament_selection(S, num_active)
                child_mask[active_positions] = 1
            
            offspring_masks.append(child_mask)
        
        return offspring_masks
    
    def generate_reference_points(self, M: int, p: int) -> np.ndarray:
        """Gera pontos de referência usando o método Das-Dennis"""
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
    
    def associate_to_reference_points(self, population: List[Individual], 
                                    reference_points: np.ndarray) -> Tuple[List[int], List[int]]:
        """Associa indivíduos aos pontos de referência mais próximos"""
        objectives = np.array([ind.objectives for ind in population])
        
        # Normaliza objetivos (assumindo minimização)
        min_obj = np.min(objectives, axis=0)
        max_obj = np.max(objectives, axis=0)
        normalized_obj = (objectives - min_obj) / (max_obj - min_obj + 1e-10)
        
        # Calcula distâncias aos pontos de referência
        distances = cdist(normalized_obj, reference_points)
        associations = np.argmin(distances, axis=1)
        
        # Conta nichos
        niche_counts = np.bincount(associations, minlength=len(reference_points))
        
        return associations.tolist(), niche_counts.tolist()
    
    def environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Seleção ambiental baseada em pontos de referência"""
        # Gera pontos de referência
        p = 12  # Parâmetro para geração de pontos
        reference_points = self.generate_reference_points(self.M, p)
        
        # Ordena população combinada
        fronts = self.fast_non_dominated_sort(combined_population)
        
        new_population = []
        front_idx = 0
        
        # Adiciona frentes completas
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.N:
            for ind_idx in fronts[front_idx]:
                new_population.append(combined_population[ind_idx])
            front_idx += 1
        
        # Se ainda há espaço, seleciona da próxima frente
        if front_idx < len(fronts) and len(new_population) < self.N:
            last_front = [combined_population[i] for i in fronts[front_idx]]
            remaining_slots = self.N - len(new_population)
            
            # Se há apenas uma solução restante, adiciona diretamente
            if remaining_slots == 1:
                new_population.append(last_front[0])
            else:
                # Associa aos pontos de referência
                associations, niche_counts = self.associate_to_reference_points(
                    new_population + last_front, reference_points
                )
                
                # Seleciona indivíduos da última frente
                current_niche_counts = niche_counts.copy()
                last_front_associations = associations[len(new_population):]
                
                selected_count = 0
                available_indices = list(range(len(last_front)))
                
                while selected_count < remaining_slots and available_indices:
                    # Encontra nicho com menor contagem
                    if len(current_niche_counts) > 0:
                        min_niche_count = min(current_niche_counts)
                        min_niche_indices = [i for i, count in enumerate(current_niche_counts) 
                                           if count == min_niche_count]
                    else:
                        min_niche_indices = list(range(len(reference_points)))
                    
                    # Seleciona indivíduo do nicho menos povoado
                    selected_in_round = False
                    for i in available_indices[:]:
                        assoc = last_front_associations[i]
                        if assoc in min_niche_indices:
                            new_population.append(last_front[i])
                            current_niche_counts[assoc] += 1
                            selected_count += 1
                            available_indices.remove(i)
                            selected_in_round = True
                            break
                    
                    # Se não conseguiu selecionar nenhum, pega o primeiro disponível
                    if not selected_in_round and available_indices:
                        new_population.append(last_front[available_indices[0]])
                        selected_count += 1
                        available_indices.remove(available_indices[0])
                    
                    if selected_count >= remaining_slots:
                        break
        
        return new_population[:self.N]
    
    def run(self) -> List[Individual]:
        """Executa o algoritmo SparseEA-AGDS"""
        print("Iniciando SparseEA-AGDS...")
        
        # Inicialização (Linhas 1-11 do Algoritmo 3)
        print("Calculando pontuações iniciais...")
        initial_scores = self.calculate_initial_scores()
        
        print("Inicializando população...")
        population = self.initialize_population(initial_scores)
        
        print(f"População inicial criada com {len(population)} indivíduos")
        
        # Loop evolucionário (Linha 12)
        for generation in range(self.max_generations):
            if generation % 100 == 0:
                print(f"Geração {generation}/{self.max_generations}")
            
            # Gera O.dec (Linha 13)
            offspring_dec = self.adaptive_genetic_operator(population)
            
            # Gera O.mask (Linha 14)
            offspring_masks = self.dynamic_scoring_mechanism(population)
            
            # Cria população de filhos O (Linha 15)
            offspring = []
            for i in range(len(offspring_dec)):
                child = Individual()
                child.dec = offspring_dec[i]
                child.mask = offspring_masks[i]
                child.objectives = self.problem.evaluate(child.solution)
                offspring.append(child)
            
            # Seleção ambiental (Linha 16)
            combined_population = population + offspring
            population = self.environmental_selection(combined_population)
        
        print("Otimização concluída!")
        return population


# Exemplo de uso
if __name__ == "__main__":
    # Cria problema de teste
    problem = SMOP1(D=10, M=2)
    
    # Cria algoritmo
    algorithm = SparseEAAGDS(
        problem=problem,
        population_size=50,
        max_generations=500,
        Pc0=0.9,
        Pm0=0.1
    )
    
    # Executa algoritmo
    final_population = algorithm.run()
    
    # Mostra resultados
    print("\nMelhores soluções encontradas:")
    for i, ind in enumerate(final_population[:5]):
        print(f"Indivíduo {i+1}:")
        print(f"  Objetivos: {ind.objectives}")
        print(f"  Esparsidade: {np.sum(ind.mask)}/{len(ind.mask)}")
        print(f"  Solução: {ind.solution}")
        print() 