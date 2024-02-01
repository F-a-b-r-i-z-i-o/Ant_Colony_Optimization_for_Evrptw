import os
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
import sys 

# Make sure you adjust the sys.path to include the directory of evrptw_config and macs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evrptw_config import EvrptwGraph
from acsd import AntColonySystem
import ast 

def run_macs_for_alpha_beta_q0_rho(file_path, param_combinations):
    results = []
    for alpha, beta, q0, rho in param_combinations:
        graph = EvrptwGraph(file_path, rho=rho)
        macs = AntColonySystem(graph, ants_num=10, max_iter=1, alpha=alpha, beta=beta, q0=q0, k1=0, k2=0, apply_local_search=False)
        final_cost, _, _, penalty, _, _, _ = macs._ACS_DIST_G()

        has_zero_penalty = all(p == 0 for p in penalty) if isinstance(penalty, list) else penalty == 0

        results.append({
            'file': os.path.basename(file_path),
            'alpha': alpha,
            'beta': beta,
            'q0': q0,
            'rho': rho,
            'cost': final_cost,
            'penalty': penalty,
            'has_zero_penalty': has_zero_penalty
        })
    return results


def find_best_params(results_df):
    zero_penalty_df = results_df[results_df['has_zero_penalty']]
    if not zero_penalty_df.empty:
        return zero_penalty_df.sort_values(by='cost').head(1)[['alpha', 'beta', 'q0', 'rho']]

    aggregated = results_df.groupby(['alpha', 'beta', 'q0', 'rho']).agg({'penalty_sum': 'mean', 'cost': 'mean'}).reset_index()
    return aggregated.sort_values(by=['penalty_sum', 'cost']).head(1)[['alpha', 'beta', 'q0', 'rho']]



def print_and_save_best_combinations_for_alpha_beta_q0(csv_file, num_top_combinations=10, output_file='result_grid_search_best_alpha_beta_q0.csv'):
    df = pd.read_csv(csv_file)

    df['penalty'] = df['penalty'].apply(ast.literal_eval)
    df['penalty_sum'] = df['penalty'].apply(sum)

    top_combinations_list = []

    for _, group_df in df.groupby('file'):
        top_combinations = group_df.sort_values(by=['penalty_sum', 'cost']).head(num_top_combinations)
        top_combinations_list.append(top_combinations)

    top_combinations_df = pd.concat(top_combinations_list)

    print("Top 10 combinations for each instance (minimizing penalty and cost):")
    print(top_combinations_df[['file', 'alpha', 'beta', 'q0', 'cost', 'penalty_sum']])

    top_combinations_df.to_csv(output_file, index=False)


def tuning_params(directory_path):
    abs_directory_path = os.path.abspath(directory_path)

    
    alpha_values = np.arange(1, 11)
    beta_values = np.arange(1, 11)
    q0_values = np.arange(0.1, 1.1, 0.1)
    rho_values = np.arange(0.1, 0.6, 0.1)  

    num_cores = 5
    param_combinations = [(alpha, beta, q0, rho)
                          for alpha in alpha_values
                          for beta in beta_values
                          for q0 in q0_values
                          for rho in rho_values]

    chunks = [param_combinations[i::num_cores] for i in range(num_cores)]

    client = Client(n_workers=num_cores, threads_per_worker=1)
    futures = []
    for filename in os.listdir(abs_directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(abs_directory_path, filename)
            for chunk in chunks:
                future = client.submit(run_macs_for_alpha_beta_q0_rho, file_path, chunk)
                futures.append(future)

    results = []
    for future in as_completed(futures):
        results.extend(future.result())

    client.close()

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Check if 'penalty' is a list and calculate 'penalty_sum' accordingly
    if results_df['penalty'].apply(lambda x: isinstance(x, list)).all():
        results_df['penalty_sum'] = results_df['penalty'].apply(sum)
    else:
        results_df['penalty_sum'] = results_df['penalty']

    results_df.to_csv('grid_search_results_for_params.csv', index=False)

    print_and_save_best_combinations_for_alpha_beta_q0("grid_search_results_for_params.csv")

    best_params = find_best_params(results_df)
    print("Best parameters (alpha, beta, q0):")
    print(best_params)

    best_params.to_csv('best_parameters.csv', index=False)

if __name__ == "__main__":
    tuning_params("../test_instances")
