#!/usr/bin/env python3
"""
Demonstração Rápida do SparseEA-AGDS Refatorado
==============================================

Este script demonstra como usar a nova estrutura refatorada do SparseEA-AGDS
com configuração externa, análise automática e comparação de resultados.
"""

import numpy as np
from pathlib import Path

# Imports do projeto
from problems import create_problem, SMOP1, SMOP2, SMOP3
from config import ConfigManager, ExperimentConfig
from sparse_ea_agds_refactored import SparseEAAGDS, run_single_experiment
from metrics import QualityMetrics, SparsityMetrics, quick_analysis
from run_experiments import ExperimentRunner


def demo_basic_usage():
    """Demonstra uso básico do algoritmo"""
    print("="*60)
    print("DEMO 1: USO BÁSICO DO ALGORITMO")
    print("="*60)
    
    # Cria problema
    problem = create_problem("SMOP1", D=10, M=2)
    print(f"Problema criado: {problem.name} (D={problem.D}, M={problem.M})")
    
    # Cria configuração
    config = ConfigManager.create_default_config("SMOP1", D=10, M=2)
    config.algorithm.population_size = 20
    config.algorithm.max_function_evaluations = 2000
    config.num_runs = 3
    
    print(f"Configuração: N={config.algorithm.population_size}, maxFE={config.algorithm.max_function_evaluations}")
    
    # Executa algoritmo
    algorithm = SparseEAAGDS(problem, config.algorithm)
    population, exec_info = algorithm.run(verbose=True)
    
    # Analisa resultados
    objectives = np.array([ind.objectives for ind in population])
    masks = [ind.mask for ind in population]
    
    print(f"\nResultados:")
    print(f"Gerações: {exec_info['generations']}")
    print(f"Avaliações: {exec_info['function_evaluations']}")
    print(f"Tempo: {exec_info['execution_time']:.2f}s")
    print(f"Tamanho da população final: {len(population)}")
    
    # Análise rápida
    true_front = problem.get_true_pareto_front(1000)
    analysis = quick_analysis(objectives, true_front, masks)
    
    print(f"\nMétricas:")
    print(f"IGD: {analysis.get('igd', 'N/A'):.4f}")
    print(f"Esparsidade média: {analysis.get('mean_sparsity_percentage', 'N/A'):.1f}%")
    print(f"Hypervolume: {analysis.get('hypervolume', 'N/A'):.4f}")


def demo_experiment_config():
    """Demonstra sistema de configuração"""
    print("\n" + "="*60)
    print("DEMO 2: SISTEMA DE CONFIGURAÇÃO")
    print("="*60)
    
    # Cria configuração personalizada
    config = ConfigManager.create_default_config("SMOP2", D=20, M=2)
    config.algorithm.population_size = 30
    config.algorithm.max_function_evaluations = 3000
    config.num_runs = 5
    config.verbose = True
    
    print("Configuração criada programaticamente")
    print(f"Problema: {config.problem.name}")
    print(f"Dimensão: {config.problem.dimension}")
    print(f"Objetivos: {config.problem.num_objectives}")
    
    # Salva configuração
    config_dir = Path("demo_configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "demo_config.json"
    ConfigManager.save_to_file(config, config_file)
    print(f"Configuração salva em: {config_file}")
    
    # Carrega configuração
    loaded_config = ConfigManager.load_from_file(config_file)
    print(f"Configuração carregada: {loaded_config.experiment_id}")
    
    # Executa experimento
    print("\nExecutando experimento...")
    results = run_single_experiment(loaded_config)
    
    # Mostra resultados
    print("\nResultados do experimento:")
    igd_summary = results.get_metric_summary('igd')
    sparsity_summary = results.get_metric_summary('sparsity_percentage')
    
    print(f"IGD: {igd_summary.value:.4f} ± {igd_summary.std_dev:.4f}")
    print(f"Esparsidade: {sparsity_summary.value:.1f}% ± {sparsity_summary.std_dev:.1f}%")
    
    if results.execution_times:
        time_summary = results.get_metric_summary('execution_times')
        print(f"Tempo médio: {time_summary.value:.2f}s ± {time_summary.std_dev:.2f}s")


def demo_multiple_problems():
    """Demonstra comparação entre problemas"""
    print("\n" + "="*60)
    print("DEMO 3: COMPARAÇÃO ENTRE PROBLEMAS")
    print("="*60)
    
    # Lista de problemas para testar
    problems_to_test = [
        ("SMOP1", 15, 2),
        ("SMOP2", 15, 2),
        ("SMOP3", 15, 3),
    ]
    
    results = []
    
    for problem_name, D, M in problems_to_test:
        print(f"\nTestando {problem_name} (D={D}, M={M})...")
        
        # Cria configuração
        config = ConfigManager.create_default_config(problem_name, D, M)
        config.algorithm.population_size = 25
        config.algorithm.max_function_evaluations = 2500
        config.num_runs = 3
        
        # Executa
        result = run_single_experiment(config)
        results.append(result)
        
        # Mostra resultado
        igd = result.get_metric_summary('igd')
        sparsity = result.get_metric_summary('sparsity_percentage')
        
        print(f"  IGD: {igd.value:.4f} ± {igd.std_dev:.4f}")
        print(f"  Esparsidade: {sparsity.value:.1f}% ± {sparsity.std_dev:.1f}%")
    
    # Tabela comparativa
    print("\nTabela Comparativa:")
    print("-" * 70)
    print(f"{'Problema':<10} {'D':<3} {'M':<3} {'IGD':<12} {'Esparsidade':<12}")
    print("-" * 70)
    
    for result in results:
        igd = result.get_metric_summary('igd')
        sparsity = result.get_metric_summary('sparsity_percentage')
        
        print(f"{result.problem_name:<10} {result.dimension:<3} {result.num_objectives:<3} "
              f"{igd.value:<8.4f}±{igd.std_dev:<3.3f} {sparsity.value:<8.1f}%±{sparsity.std_dev:<3.1f}")


def demo_runner_system():
    """Demonstra sistema de execução automatizada"""
    print("\n" + "="*60)
    print("DEMO 4: SISTEMA DE EXECUÇÃO AUTOMATIZADA")
    print("="*60)
    
    # Cria runner
    runner = ExperimentRunner(results_dir="demo_results")
    
    # Executa teste rápido
    print("Executando teste rápido...")
    result = runner.run_quick_test()
    
    # Mostra resultados
    print("\nResultados do teste rápido:")
    igd_summary = result.get_metric_summary('igd')
    sparsity_summary = result.get_metric_summary('sparsity_percentage')
    
    print(f"IGD: {igd_summary.value:.4f} ± {igd_summary.std_dev:.4f}")
    print(f"Esparsidade: {sparsity_summary.value:.1f}% ± {sparsity_summary.std_dev:.1f}%")
    
    # Cria alguns experimentos personalizados
    print("\nCriando experimentos personalizados...")
    
    custom_configs = []
    for problem in ["SMOP1", "SMOP2"]:
        config = ConfigManager.create_default_config(problem, D=10, M=2)
        config.algorithm.population_size = 20
        config.algorithm.max_function_evaluations = 2000
        config.num_runs = 3
        custom_configs.append(config)
    
    # Executa experimentos
    print("Executando experimentos personalizados...")
    results = runner.run_benchmark_experiments(custom_configs, parallel=False)
    
    # Gera tabelas
    print("Gerando tabelas...")
    tables = runner.generate_summary_tables(results)
    runner.save_tables(tables)
    
    # Mostra resumo
    runner.print_experiment_summary(results)
    
    print(f"\nResultados salvos em: {runner.results_dir}")


def demo_advanced_analysis():
    """Demonstra análise avançada de resultados"""
    print("\n" + "="*60)
    print("DEMO 5: ANÁLISE AVANÇADA")
    print("="*60)
    
    # Cria problema
    problem = SMOP1(D=12, M=2)
    
    # Executa algoritmo
    from config import AlgorithmConfig
    config = AlgorithmConfig()
    config.population_size = 25
    config.max_function_evaluations = 2500
    config.random_seed = 123
    
    algorithm = SparseEAAGDS(problem, config)
    population, exec_info = algorithm.run(verbose=False)
    
    # Análise detalhada
    print("Análise detalhada da população final:")
    
    # Extrai dados
    objectives = np.array([ind.objectives for ind in population])
    masks = [ind.mask for ind in population]
    solutions = [ind.solution for ind in population]
    
    print(f"Tamanho da população: {len(population)}")
    print(f"Dimensão das soluções: {len(solutions[0])}")
    
    # Métricas de qualidade
    quality_metrics = QualityMetrics()
    true_front = problem.get_true_pareto_front(1000)
    
    igd = quality_metrics.calculate_igd(objectives, true_front)
    gd = quality_metrics.calculate_gd(objectives, true_front)
    hv = quality_metrics.calculate_hypervolume(objectives)
    spread = quality_metrics.calculate_spread(objectives)
    
    print(f"\nMétricas de qualidade:")
    print(f"  IGD: {igd:.6f}")
    print(f"  GD: {gd:.6f}")
    print(f"  Hypervolume: {hv:.6f}")
    print(f"  Spread: {spread:.6f}")
    
    # Métricas de esparsidade
    sparsity_metrics = SparsityMetrics()
    sparsity_results = sparsity_metrics.calculate_sparsity(masks)
    
    print(f"\nMétricas de esparsidade:")
    print(f"  Esparsidade média: {sparsity_results['mean_sparsity']:.2f}")
    print(f"  Esparsidade mediana: {sparsity_results['median_sparsity']:.2f}")
    print(f"  Percentual médio: {sparsity_results['mean_sparsity_percentage']:.1f}%")
    print(f"  Variação: {sparsity_results['min_sparsity']:.0f} - {sparsity_results['max_sparsity']:.0f}")
    
    # Frequência de uso das variáveis
    var_frequency = sparsity_metrics.calculate_variable_frequency(masks)
    print(f"\nFrequência de uso das variáveis:")
    print(f"  Mais usada: {np.max(var_frequency):.2f}")
    print(f"  Menos usada: {np.min(var_frequency):.2f}")
    print(f"  Média: {np.mean(var_frequency):.2f}")
    
    # Diversidade
    diversity = sparsity_metrics.calculate_diversity_index(masks)
    print(f"  Índice de diversidade: {diversity:.2f}")
    
    # Histórico de execução
    print(f"\nHistórico de execução:")
    history = exec_info['history']
    print(f"  Gerações executadas: {len(history['generations'])}")
    print(f"  Avaliações de função: {history['function_evaluations'][-1]}")
    print(f"  Tempo total: {history['execution_time'][-1]:.2f}s")
    
    if history['best_igd']:
        valid_igd = [x for x in history['best_igd'] if not np.isnan(x)]
        if valid_igd:
            print(f"  IGD inicial: {valid_igd[0]:.6f}")
            print(f"  IGD final: {valid_igd[-1]:.6f}")
            print(f"  Melhoria: {(valid_igd[0] - valid_igd[-1]) / valid_igd[0] * 100:.1f}%")


def main():
    """Função principal da demonstração"""
    print("DEMONSTRAÇÃO DO SPARSEEA-AGDS REFATORADO")
    print("========================================")
    
    try:
        # Executa todas as demonstrações
        demo_basic_usage()
        demo_experiment_config()
        demo_multiple_problems()
        demo_runner_system()
        demo_advanced_analysis()
        
        print("\n" + "="*60)
        print("TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!")
        print("="*60)
        
        print("\nPróximos passos:")
        print("1. Execute 'python run_experiments.py' para experimentos completos")
        print("2. Veja o README_refactored.md para documentação completa")
        print("3. Customize os problemas e configurações conforme necessário")
        print("4. Use os resultados para validar contra o artigo original")
        
    except Exception as e:
        print(f"\nERRO na demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 