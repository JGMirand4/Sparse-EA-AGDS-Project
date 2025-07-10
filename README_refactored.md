# SparseEA-AGDS - Implementação Refatorada

Uma implementação robusta e reproduzível do algoritmo **SparseEA-AGDS** (Sparse Evolutionary Algorithm with Adaptive Genetic Operator and Dynamic Scoring) para otimização multi-objetivo esparsa.

## 🚀 **Melhorias da Versão Refatorada**

### **1. Arquitetura Modular**
- **Separação de responsabilidades**: Problema, Algoritmo, Configuração e Métricas
- **Interface abstrata**: Fácil adição de novos problemas de benchmark
- **Configuração externa**: Parâmetros gerenciados via arquivos JSON/YAML
- **Análise integrada**: Métricas e testes estatísticos automatizados

### **2. Reprodutibilidade Total**
- **Controle de seeds**: Cada execução usa seed específica
- **Parâmetros exatos**: Configuração idêntica ao artigo original
- **Critério de parada**: Baseado em avaliações de função (maxFE = 100×D)
- **Testes estatísticos**: Wilcoxon rank-sum com correção para múltiplas comparações

### **3. Facilidade de Uso**
- **Script automatizado**: Execução de todos os experimentos com um comando
- **Execução paralela**: Suporte a múltiplos cores para acelerar experimentos
- **Tabelas automáticas**: Geração de resultados em CSV, Excel e LaTeX
- **Análise completa**: IGD, esparsidade, testes estatísticos

## 📁 **Estrutura dos Arquivos**

```
sparse-ea-agds-project/
├── problems.py                    # Benchmarks SMOP1-SMOP8
├── config.py                      # Sistema de configuração
├── metrics.py                     # Métricas de avaliação
├── sparse_ea_agds_refactored.py   # Algoritmo principal
├── run_experiments.py             # Script de execução
├── requirements.txt               # Dependências
├── README_refactored.md           # Este arquivo
└── experiment_results/            # Resultados dos experimentos
    ├── configs/                   # Configurações salvas
    ├── raw_results/              # Resultados brutos
    ├── tables/                   # Tabelas resumo
    └── plots/                    # Gráficos (futuramente)
```

## ⚙️ **Instalação**

```bash
# Clone o repositório
git clone <repository-url>
cd sparse-ea-agds-project

# Instale as dependências
pip install -r requirements.txt

# Teste a instalação
python run_experiments.py
```

## 🎯 **Uso Rápido**

### **Teste Rápido**
```python
from run_experiments import ExperimentRunner

runner = ExperimentRunner()
result = runner.run_quick_test()
```

### **Experimentos Completos**
```bash
# Executa todos os benchmarks do artigo
python run_experiments.py

# Escolha opção 2 para experimentos completos
# Escolha opção 1 para teste rápido
```

### **Uso Programático**
```python
from config import ConfigManager
from sparse_ea_agds_refactored import run_single_experiment

# Cria configuração personalizada
config = ConfigManager.create_default_config("SMOP1", D=100, M=2)
config.algorithm.population_size = 50
config.num_runs = 10

# Executa experimento
results = run_single_experiment(config)

# Analisa resultados
print(f"IGD médio: {results.get_metric_summary('igd').value:.4f}")
print(f"Esparsidade: {results.get_metric_summary('sparsity_percentage').value:.1f}%")
```

## 🔬 **Experimentos do Artigo**

### **Configuração Exata**
Os experimentos seguem **exatamente** os parâmetros do artigo:

| Parâmetro | Valor |
|-----------|-------|
| População (N) | 100 |
| Avaliações máximas | 100×D |
| Probabilidade de crossover base (Pc₀) | 1.0 |
| Probabilidade de mutação base (Pm₀) | 1/D |
| Parâmetro SBX (η_c) | 20 |
| Parâmetro mutação (η_m) | 20 |
| Execuções independentes | 30 |

### **Problemas de Benchmark**
- **SMOP1-SMOP8**: Implementação completa de todos os problemas
- **Dimensões**: 100, 500, 1000 variáveis
- **Objetivos**: 2, 3, 5, 8, 10, 15 (conforme aplicável)

### **Métricas de Avaliação**
- **IGD (Inverted Generational Distance)**: Qualidade da fronteira de Pareto
- **Esparsidade**: Porcentagem de variáveis não-zero
- **Testes estatísticos**: Wilcoxon rank-sum (p < 0.05)

## 📊 **Análise de Resultados**

### **Tabelas Automáticas**
O sistema gera automaticamente:

1. **Resumo Geral**: IGD e esparsidade por problema
2. **Análise por Dimensão**: Desempenho vs. dimensionalidade
3. **Análise por Objetivos**: Desempenho vs. número de objetivos
4. **Comparação Estatística**: Significância das diferenças

### **Exemplo de Saída**
```
Problem         D     M    IGD_mean    IGD_std    Sparsity_mean  Sparsity_std
SMOP1         100     2      0.0234     0.0045           15.2%       3.1%
SMOP1         500     2      0.0456     0.0078           12.8%       2.9%
SMOP1        1000     2      0.0678     0.0089           11.4%       2.5%
```

## 🛠️ **Configuração Avançada**

### **Arquivo de Configuração (JSON)**
```json
{
  "problem": {
    "name": "SMOP1",
    "dimension": 100,
    "num_objectives": 2
  },
  "algorithm": {
    "population_size": 100,
    "max_function_evaluations": 10000,
    "Pc0": 1.0,
    "Pm0": 0.01,
    "eta_c": 20.0,
    "eta_m": 20.0,
    "random_seed": 42
  },
  "num_runs": 30,
  "save_results": true,
  "verbose": true
}
```

### **Criação de Novos Problemas**
```python
from problems import Problem

class MyProblem(Problem):
    def __init__(self, D: int, M: int):
        super().__init__(D, M)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Implementa funções objetivo
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        return np.array([f1, f2])
    
    def get_bounds(self):
        return np.zeros(self.D), np.ones(self.D)
    
    def get_true_pareto_front(self, n_points=10000):
        # Implementa fronteira de Pareto verdadeira
        pass
```

## 🔄 **Execução Paralela**

### **Configuração de Paralelização**
```python
runner = ExperimentRunner()

# Executa 4 experimentos simultaneamente
results = runner.run_benchmark_experiments(
    parallel=True,
    max_workers=4
)
```

### **Otimização de Performance**
- **Processamento paralelo**: Múltiplos experimentos simultâneos
- **Gerenciamento de memória**: Resultados salvos incrementalmente
- **Controle de recursos**: Número configurável de workers

## 📈 **Validação dos Resultados**

### **Comparação com o Artigo**
Os resultados devem ser **estatisticamente equivalentes** aos do artigo original:

1. **IGD**: Valores próximos às tabelas publicadas
2. **Esparsidade**: Porcentagem de variáveis ativas similar
3. **Significância**: Testes estatísticos confirmam melhorias

### **Exemplo de Validação**
```python
# Carrega resultados
results = runner.load_results()

# Compara com baseline
baseline_results = load_baseline_results()  # Implementar
comparison = runner.generate_comparison_tables(results, baseline_results)

# Verifica significância
significant_improvements = comparison[comparison['Statistical_Test'] == '+']
print(f"Melhorias significativas: {len(significant_improvements)}")
```

## 🐛 **Solução de Problemas**

### **Problemas Comuns**

1. **Erro de memória**: Reduza `max_workers` ou `population_size`
2. **Convergência lenta**: Verifique parâmetros do algoritmo
3. **Resultados inconsistentes**: Confirme controle de seeds

### **Debug e Logging**
```python
# Ativa logging detalhado
config.verbose = True
config.log_frequency = 50  # Log a cada 50 avaliações

# Salva populações para análise
config.save_populations = True
```

## 📚 **Referências**

- **Artigo Original**: SparseEA-AGDS for Sparse Multi-objective Optimization
- **Baseline**: SparseEA (Referência [7] no artigo)
- **Problemas**: Benchmarks SMOP1-SMOP8

## 🤝 **Contribuição**

Para contribuir com o projeto:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature
3. **Implemente** seguindo os padrões do código
4. **Teste** com os benchmarks existentes
5. **Submeta** um pull request

## 📄 **Licença**

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para detalhes.

---

## 🎓 **Citação**

Se você usar este código em sua pesquisa, por favor cite:

```bibtex
@article{sparseea_agds_2024,
  title={SparseEA-AGDS: Implementação Refatorada para Otimização Multi-objetivo Esparsa},
  author={Seu Nome},
  year={2024},
  url={https://github.com/seu-usuario/sparse-ea-agds}
}
```

---

**Implementação robusta, reproduzível e pronta para pesquisa científica!** 🚀 