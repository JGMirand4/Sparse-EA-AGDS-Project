"""
Script de Execução de Experimentos
==================================

Script automatizado para executar todos os experimentos do artigo SparseEA-AGDS
e gerar tabelas comparativas com os resultados.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Imports do projeto
from config import (
    ExperimentConfig, ConfigManager, 
    create_benchmark_configs, create_test_config
)
from sparse_ea_agds_refactored import run_single_experiment
from metrics import ResultsAnalyzer, ExperimentResults, StatisticalTests


class ExperimentRunner:
    """Gerenciador de execução de experimentos"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        """
        Inicializa o runner
        
        Args:
            results_dir: Diretório onde salvar resultados
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdiretórios
        self.configs_dir = self.results_dir / "configs"
        self.raw_results_dir = self.results_dir / "raw_results"
        self.tables_dir = self.results_dir / "tables"
        self.plots_dir = self.results_dir / "plots"
        
        for dir_path in [self.configs_dir, self.raw_results_dir, self.tables_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Analisador de resultados
        self.analyzer = ResultsAnalyzer()
        self.statistical_tests = StatisticalTests()
    
    def run_benchmark_experiments(self, 
                                 experiment_configs: List[ExperimentConfig] = None,
                                 parallel: bool = True,
                                 max_workers: int = 4) -> List[ExperimentResults]:
        """
        Executa experimentos de benchmark
        
        Args:
            experiment_configs: Lista de configurações (padrão: todos os benchmarks)
            parallel: Se deve executar em paralelo
            max_workers: Número máximo de workers paralelos
            
        Returns:
            Lista de resultados
        """
        if experiment_configs is None:
            experiment_configs = create_benchmark_configs()
        
        print(f"Executando {len(experiment_configs)} experimentos...")
        
        # Salva configurações
        for i, config in enumerate(experiment_configs):
            config_file = self.configs_dir / f"experiment_{i:03d}.json"
            ConfigManager.save_to_file(config, config_file)
        
        # Executa experimentos
        all_results = []
        
        if parallel and len(experiment_configs) > 1:
            # Execução paralela
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submete todos os experimentos
                future_to_config = {
                    executor.submit(run_single_experiment, config): (i, config)
                    for i, config in enumerate(experiment_configs)
                }
                
                # Coleta resultados
                for future in as_completed(future_to_config):
                    exp_idx, config = future_to_config[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # Salva resultado individual
                        result_file = self.raw_results_dir / f"result_{exp_idx:03d}.pkl"
                        with open(result_file, 'wb') as f:
                            pickle.dump(result, f)
                        
                        print(f"Experimento {exp_idx + 1} concluído: {config.experiment_id}")
                        
                    except Exception as e:
                        print(f"Erro no experimento {exp_idx + 1}: {e}")
        
        else:
            # Execução sequencial
            for i, config in enumerate(experiment_configs):
                try:
                    print(f"\nExecutando experimento {i + 1}/{len(experiment_configs)}")
                    result = run_single_experiment(config)
                    all_results.append(result)
                    
                    # Salva resultado individual
                    result_file = self.raw_results_dir / f"result_{i:03d}.pkl"
                    with open(result_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    print(f"Experimento {i + 1} concluído: {config.experiment_id}")
                    
                except Exception as e:
                    print(f"Erro no experimento {i + 1}: {e}")
        
        # Salva todos os resultados
        all_results_file = self.results_dir / "all_results.pkl"
        with open(all_results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\nTodos os experimentos concluídos! Resultados salvos em: {self.results_dir}")
        
        return all_results
    
    def load_results(self, results_file: str = None) -> List[ExperimentResults]:
        """
        Carrega resultados salvos
        
        Args:
            results_file: Arquivo específico (padrão: all_results.pkl)
            
        Returns:
            Lista de resultados
        """
        if results_file is None:
            results_file = self.results_dir / "all_results.pkl"
        else:
            results_file = Path(results_file)
        
        if not results_file.exists():
            raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_file}")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def generate_summary_tables(self, results: List[ExperimentResults]) -> Dict[str, pd.DataFrame]:
        """
        Gera tabelas resumo dos resultados
        
        Args:
            results: Lista de resultados
            
        Returns:
            Dicionário com DataFrames das tabelas
        """
        tables = {}
        
        # Tabela 1: Resultados gerais de IGD
        table_data = []
        for result in results:
            igd_summary = result.get_metric_summary('igd')
            sparsity_summary = result.get_metric_summary('sparsity_percentage')
            
            table_data.append({
                'Problem': result.problem_name,
                'D': result.dimension,
                'M': result.num_objectives,
                'IGD_mean': igd_summary.value,
                'IGD_std': igd_summary.std_dev,
                'Sparsity_mean': sparsity_summary.value,
                'Sparsity_std': sparsity_summary.std_dev,
                'Runs': result.num_runs
            })
        
        tables['igd_summary'] = pd.DataFrame(table_data)
        
        # Tabela 2: Resultados por dimensão
        dimension_table = []
        for dimension in [100, 500, 1000]:
            dim_results = [r for r in results if r.dimension == dimension]
            if dim_results:
                igd_values = []
                sparsity_values = []
                
                for result in dim_results:
                    igd_values.extend(result.igd_values)
                    if result.sparsity_percentage:
                        sparsity_values.extend(result.sparsity_percentage)
                
                dimension_table.append({
                    'Dimension': dimension,
                    'Problems': len(dim_results),
                    'IGD_mean': np.mean(igd_values),
                    'IGD_std': np.std(igd_values),
                    'Sparsity_mean': np.mean(sparsity_values) if sparsity_values else 0,
                    'Sparsity_std': np.std(sparsity_values) if sparsity_values else 0
                })
        
        tables['dimension_summary'] = pd.DataFrame(dimension_table)
        
        # Tabela 3: Resultados por número de objetivos
        objective_table = []
        for num_objectives in [2, 3, 5, 8, 10, 15]:
            obj_results = [r for r in results if r.num_objectives == num_objectives]
            if obj_results:
                igd_values = []
                sparsity_values = []
                
                for result in obj_results:
                    igd_values.extend(result.igd_values)
                    if result.sparsity_percentage:
                        sparsity_values.extend(result.sparsity_percentage)
                
                objective_table.append({
                    'Objectives': num_objectives,
                    'Problems': len(obj_results),
                    'IGD_mean': np.mean(igd_values),
                    'IGD_std': np.std(igd_values),
                    'Sparsity_mean': np.mean(sparsity_values) if sparsity_values else 0,
                    'Sparsity_std': np.std(sparsity_values) if sparsity_values else 0
                })
        
        tables['objectives_summary'] = pd.DataFrame(objective_table)
        
        return tables
    
    def generate_comparison_tables(self, 
                                  results1: List[ExperimentResults],
                                  results2: List[ExperimentResults],
                                  algorithm1_name: str = "SparseEA-AGDS",
                                  algorithm2_name: str = "SparseEA") -> pd.DataFrame:
        """
        Gera tabela de comparação entre algoritmos
        
        Args:
            results1: Resultados do primeiro algoritmo
            results2: Resultados do segundo algoritmo
            algorithm1_name: Nome do primeiro algoritmo
            algorithm2_name: Nome do segundo algoritmo
            
        Returns:
            DataFrame com resultados da comparação
        """
        comparison_data = []
        
        # Agrupa resultados por problema
        results1_by_problem = {f"{r.problem_name}_D{r.dimension}_M{r.num_objectives}": r for r in results1}
        results2_by_problem = {f"{r.problem_name}_D{r.dimension}_M{r.num_objectives}": r for r in results2}
        
        # Compara problemas em comum
        common_problems = set(results1_by_problem.keys()) & set(results2_by_problem.keys())
        
        for problem_key in sorted(common_problems):
            result1 = results1_by_problem[problem_key]
            result2 = results2_by_problem[problem_key]
            
            # Testa significância estatística
            comparison = self.analyzer.compare_algorithms(result1, result2)
            igd_test = comparison.get('igd', {})
            
            comparison_data.append({
                'Problem': result1.problem_name,
                'D': result1.dimension,
                'M': result1.num_objectives,
                f'{algorithm1_name}_IGD': f"{result1.get_metric_summary('igd').value:.4f}±{result1.get_metric_summary('igd').std_dev:.4f}",
                f'{algorithm2_name}_IGD': f"{result2.get_metric_summary('igd').value:.4f}±{result2.get_metric_summary('igd').std_dev:.4f}",
                'Statistical_Test': igd_test.get('result', '?'),
                'P_Value': igd_test.get('p_value', np.nan)
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_tables(self, tables: Dict[str, pd.DataFrame]):
        """
        Salva tabelas em arquivos
        
        Args:
            tables: Dicionário com DataFrames
        """
        for table_name, df in tables.items():
            # Salva CSV
            csv_file = self.tables_dir / f"{table_name}.csv"
            df.to_csv(csv_file, index=False)
            
            # Salva Excel
            excel_file = self.tables_dir / f"{table_name}.xlsx"
            df.to_excel(excel_file, index=False)
            
            # Salva LaTeX
            latex_file = self.tables_dir / f"{table_name}.tex"
            df.to_latex(latex_file, index=False)
        
        print(f"Tabelas salvas em: {self.tables_dir}")
    
    def run_quick_test(self) -> ExperimentResults:
        """
        Executa um teste rápido para verificar se tudo está funcionando
        
        Returns:
            Resultados do teste
        """
        print("Executando teste rápido...")
        
        # Cria configuração de teste
        test_config = create_test_config()
        
        # Executa
        result = run_single_experiment(test_config)
        
        # Salva resultado
        test_result_file = self.results_dir / "quick_test_result.pkl"
        with open(test_result_file, 'wb') as f:
            pickle.dump(result, f)
        
        print("Teste rápido concluído!")
        return result
    
    def print_experiment_summary(self, results: List[ExperimentResults]):
        """
        Imprime resumo dos experimentos
        
        Args:
            results: Lista de resultados
        """
        print("\n" + "="*60)
        print("RESUMO DOS EXPERIMENTOS")
        print("="*60)
        
        print(f"Total de experimentos: {len(results)}")
        
        # Agrupa por problema
        problems = {}
        for result in results:
            problem_name = result.problem_name
            if problem_name not in problems:
                problems[problem_name] = []
            problems[problem_name].append(result)
        
        print(f"Problemas testados: {len(problems)}")
        
        for problem_name, problem_results in problems.items():
            print(f"\n{problem_name}:")
            for result in problem_results:
                igd_summary = result.get_metric_summary('igd')
                sparsity_summary = result.get_metric_summary('sparsity_percentage')
                
                print(f"  D={result.dimension}, M={result.num_objectives}: "
                      f"IGD={igd_summary.value:.4f}±{igd_summary.std_dev:.4f}, "
                      f"Sparsity={sparsity_summary.value:.1f}%±{sparsity_summary.std_dev:.1f}%")
        
        # Estatísticas gerais
        all_igd = []
        all_sparsity = []
        
        for result in results:
            all_igd.extend(result.igd_values)
            if result.sparsity_percentage:
                all_sparsity.extend(result.sparsity_percentage)
        
        print(f"\nEstatísticas gerais:")
        print(f"IGD médio: {np.mean(all_igd):.4f} ± {np.std(all_igd):.4f}")
        print(f"Esparsidade média: {np.mean(all_sparsity):.1f}% ± {np.std(all_sparsity):.1f}%")
        
        print("="*60)


def main():
    """Função principal do script"""
    print("="*60)
    print("EXECUÇÃO DE EXPERIMENTOS SPARSEEA-AGDS")
    print("="*60)
    
    # Cria runner
    runner = ExperimentRunner()
    
    # Opções do usuário
    print("\nOpções disponíveis:")
    print("1. Teste rápido")
    print("2. Executar todos os experimentos de benchmark")
    print("3. Executar experimentos específicos")
    print("4. Carregar e analisar resultados existentes")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == "1":
        # Teste rápido
        result = runner.run_quick_test()
        
        # Mostra resultados
        print("\nResultados do teste rápido:")
        igd_summary = result.get_metric_summary('igd')
        sparsity_summary = result.get_metric_summary('sparsity_percentage')
        
        print(f"IGD: {igd_summary.value:.4f} ± {igd_summary.std_dev:.4f}")
        print(f"Esparsidade: {sparsity_summary.value:.1f}% ± {sparsity_summary.std_dev:.1f}%")
    
    elif choice == "2":
        # Todos os experimentos
        print("\nConfigurando experimentos de benchmark...")
        
        # Pergunta sobre paralelização
        parallel = input("Executar em paralelo? (s/n): ").strip().lower() == 's'
        max_workers = 4
        
        if parallel:
            try:
                max_workers = int(input("Número de workers paralelos (padrão: 4): ") or "4")
            except ValueError:
                max_workers = 4
        
        # Executa experimentos
        experiment_configs = create_benchmark_configs()
        results = runner.run_benchmark_experiments(
            experiment_configs, parallel=parallel, max_workers=max_workers
        )
        
        # Análise dos resultados
        print("\nGerando tabelas de resultados...")
        tables = runner.generate_summary_tables(results)
        runner.save_tables(tables)
        
        # Mostra resumo
        runner.print_experiment_summary(results)
    
    elif choice == "3":
        # Experimentos específicos
        print("\nConfigurando experimentos específicos...")
        
        # Permite seleção de problemas
        available_problems = ['SMOP1', 'SMOP2', 'SMOP3', 'SMOP4', 'SMOP5', 'SMOP6', 'SMOP7', 'SMOP8']
        print(f"Problemas disponíveis: {', '.join(available_problems)}")
        
        selected_problems = input("Digite os problemas (separados por vírgula): ").strip().split(',')
        selected_problems = [p.strip() for p in selected_problems if p.strip() in available_problems]
        
        if not selected_problems:
            print("Nenhum problema válido selecionado.")
            return
        
        # Cria configurações específicas
        specific_configs = []
        for problem in selected_problems:
            for D in [100, 500, 1000]:
                for M in [2, 3]:
                    from config import ConfigManager
                    config = ConfigManager.create_default_config(problem, D, M)
                    specific_configs.append(config)
        
        # Executa
        results = runner.run_benchmark_experiments(specific_configs)
        
        # Análise
        tables = runner.generate_summary_tables(results)
        runner.save_tables(tables)
        runner.print_experiment_summary(results)
    
    elif choice == "4":
        # Carregar resultados existentes
        try:
            results = runner.load_results()
            
            print(f"\nResultados carregados: {len(results)} experimentos")
            
            # Gera tabelas
            tables = runner.generate_summary_tables(results)
            runner.save_tables(tables)
            
            # Mostra resumo
            runner.print_experiment_summary(results)
            
        except FileNotFoundError:
            print("Nenhum resultado encontrado. Execute os experimentos primeiro.")
    
    else:
        print("Opção inválida.")
    
    print("\nScript concluído!")


if __name__ == "__main__":
    main() 