# SparseEA-AGDS: Implementação do Algoritmo Evolutivo Esparso com Seleção Adaptativa

Esta implementação fornece uma versão completa do algoritmo **SparseEA-AGDS** (Sparse Evolutionary Algorithm with Adaptive Genetic operators and Dynamic Scoring mechanism) para otimização multi-objetivo esparsa.

## Características Principais

- **Operador Genético Adaptativo**: Probabilidades de crossover e mutação que se adaptam com base na qualidade das soluções
- **Mecanismo de Pontuação Dinâmica**: Atualização dinâmica da importância das variáveis a cada geração
- **Seleção Ambiental por Pontos de Referência**: Seleção baseada em pontos de referência para manter diversidade
- **Suporte a Múltiplos Objetivos**: Funciona com problemas de 2 ou mais objetivos
- **Esparsidade Controlada**: Automaticamente encontra soluções com poucas variáveis ativas

## Estrutura dos Arquivos

```
sparse-ea-agds-project/
├── sparse_ea_agds.py       # Implementação principal do algoritmo
├── example_usage.py        # Exemplos de uso e análise
├── requirements.txt        # Dependências necessárias
└── README.md              # Esta documentação
```

## Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso Básico

### Exemplo Simples

```python
from sparse_ea_agds import SparseEAAGDS, SMOP1

# Cria um problema de teste
problem = SMOP1(D=10, M=2)  # 10 dimensões, 2 objetivos

# Configura o algoritmo
algorithm = SparseEAAGDS(
    problem=problem,
    population_size=50,
    max_generations=300,
    Pc0=0.9,  # Probabilidade base de crossover
    Pm0=0.1   # Probabilidade base de mutação
)

# Executa a otimização
final_population = algorithm.run()

# Analisa os resultados
for ind in final_population[:5]:
    print(f"Objetivos: {ind.objectives}")
    print(f"Esparsidade: {np.sum(ind.mask)}/{len(ind.mask)}")
    print(f"Solução: {ind.solution}")
```

### Criando Problemas Customizados

```python
from sparse_ea_agds import Problem
import numpy as np

class MeuProblema(Problem):
    def __init__(self, D=20, M=2):
        self.D = D
        self.M = M
    
    def evaluate(self, x):
        # Implemente sua função de avaliação aqui
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        return np.array([f1, f2])
    
    @property
    def dimension(self):
        return self.D
    
    @property
    def num_objectives(self):
        return self.M
    
    @property
    def bounds(self):
        lower = np.zeros(self.D)
        upper = np.ones(self.D)
        return lower, upper

# Use seu problema customizado
problem = MeuProblema(D=15, M=2)
algorithm = SparseEAAGDS(problem=problem)
resultado = algorithm.run()
```

## Exemplos Avançados

Execute o arquivo `example_usage.py` para ver exemplos avançados incluindo:

```bash
python example_usage.py
```

- **Estudo Comparativo**: Testa o algoritmo em diferentes problemas
- **Análise de Sensibilidade**: Mostra como diferentes parâmetros afetam o desempenho
- **Visualização de Resultados**: Gera gráficos dos resultados
- **Análise de Esparsidade**: Mostra estatísticas sobre a esparsidade das soluções

## Parâmetros do Algoritmo

| Parâmetro | Descrição | Valor Padrão |
|-----------|-----------|--------------|
| `population_size` | Tamanho da população | 100 |
| `max_generations` | Número máximo de gerações | 1000 |
| `Pc0` | Probabilidade base de crossover | 0.9 |
| `Pm0` | Probabilidade base de mutação | 0.1 |
| `eta_c` | Parâmetro do crossover SBX | 20.0 |
| `eta_m` | Parâmetro da mutação polinomial | 20.0 |

## Componentes Principais

### 1. Classe Individual

Representa uma solução no algoritmo:

```python
@dataclass
class Individual:
    dec: np.ndarray      # Variáveis de decisão reais
    mask: np.ndarray     # Máscara binária (esparsidade)
    objectives: np.ndarray  # Valores dos objetivos
    rank: int            # Rank de não-dominância
    
    @property
    def solution(self):
        return self.dec * self.mask  # Solução final
```

### 2. Operador Genético Adaptativo

- Calcula probabilidades adaptativas baseadas no rank das soluções
- Usa **Simulated Binary Crossover (SBX)** para crossover
- Usa **Mutação Polinomial** para mutação
- Equações implementadas:
  - $P_{s,i} = \frac{maxr - r_i + 1}{maxr}$ (Probabilidade de seleção)
  - $P_{c,i} = P_{c0} \times P_{s,i}$ (Probabilidade de crossover)
  - $P_{m,i} = P_{m0} \times P_{s,i}$ (Probabilidade de mutação)

### 3. Mecanismo de Pontuação Dinâmica

- Atualiza a importância das variáveis a cada geração
- Usa informações da população atual para calcular pontuações
- Implementa as equações:
  - $S_{i_r} = maxr - r_i + 1$ (Pontuação da camada)
  - $SumS = S_r^T \times mask$ (Pontuação ponderada)
  - $S_d = maxS - sumS_d + 1$ (Pontuação final)

### 4. Seleção Ambiental por Pontos de Referência

- Usa pontos de referência gerados pelo método Das-Dennis
- Mantém diversidade na frente de Pareto
- Combina ordenação não-dominada com seleção por nichos

## Problemas de Teste Incluídos

### SMOP1 (Bi-objetivo)
- f1 = Σ(x²)
- f2 = Σ((x-1)²)

### SMOP2 (Bi-objetivo com não-linearidade)
- f1 = Σ(x²) + sin(Σ(x))
- f2 = Σ((x-0.5)²) + cos(Σ(x))

### SMOP3 (Tri-objetivo)
- f1 = Σ(x²)
- f2 = Σ((x-1)²)
- f3 = Σ((x-0.5)²)

## Análise de Resultados

O algoritmo produz:

1. **População Final**: Lista de soluções otimizadas
2. **Frente de Pareto**: Soluções não-dominadas
3. **Estatísticas de Esparsidade**: Número de variáveis ativas
4. **Valores dos Objetivos**: Performance nas funções objetivo

### Métricas de Avaliação

- **Número de soluções não-dominadas**
- **Esparsidade média** (número de variáveis ativas)
- **Convergência dos objetivos**
- **Tempo de execução**

## Personalização

### Modificando Operadores Genéticos

```python
# Sobrescreva métodos específicos
class MeuSparseEA(SparseEAAGDS):
    def simulated_binary_crossover(self, parent1, parent2):
        # Sua implementação de crossover
        pass
    
    def polynomial_mutation(self, individual):
        # Sua implementação de mutação
        pass
```

### Adicionando Novos Critérios de Seleção

```python
def environmental_selection(self, combined_population):
    # Sua lógica de seleção ambiental
    pass
```

## Dicas de Uso

1. **Ajuste do Tamanho da População**: Populações maiores (100-200) para problemas mais complexos
2. **Número de Gerações**: Aumente para problemas de alta dimensionalidade
3. **Parâmetros Pc0 e Pm0**: Valores altos (0.9, 0.1) funcionam bem na maioria dos casos
4. **Problemas de Alta Dimensionalidade**: Considere ajustar os parâmetros eta_c e eta_m

## Limitações

- Requer que as funções objetivo sejam avaliáveis numericamente
- Melhor performance em problemas onde a esparsidade é desejável
- Pode ser computacionalmente intensivo para problemas de muitas dimensões

## Referências

Esta implementação é baseada no artigo sobre SparseEA-AGDS que introduce:

1. Operadores genéticos adaptativos
2. Mecanismo de pontuação dinâmica
3. Seleção ambiental baseada em pontos de referência

Para detalhes teóricos, consulte o artigo original.

## Contribuições

Para melhorias ou correções:

1. Identifique o problema
2. Implemente a solução
3. Teste com os exemplos fornecidos
4. Documente as mudanças

## Licença

Esta implementação é fornecida para fins educacionais e de pesquisa. 