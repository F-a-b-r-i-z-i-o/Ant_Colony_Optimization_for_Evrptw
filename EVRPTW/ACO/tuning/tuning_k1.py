import os
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
import sys
# Add the directory containing evrptw_config and macs to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evrptw_config import EvrptwGraph
from acsd import AntColonySystem
import ast

def run_macs_for_k(file_path, k1_combinations):
    results = []
    for k1 in k1_combinations:
        graph = EvrptwGraph(file_path)
        macs = AntColonySystem(graph, ants_num=10, max_iter=3, alpha=1, beta=2, q0=0.9, k1=k1)
        final_cost, _, _, penalty, _, _, _ = macs._ACS_DIST_G()

        has_zero_penalty = all(p == 0 for p in penalty)

        results.append({
            'file': os.path.basename(file_path),
            'k1': k1,
            'cost': final_cost,
            'penalty': penalty,
            'has_zero_penalty': has_zero_penalty
        })
    return results


def find_best_k1(results_df):
    zero_penalty_df = results_df[results_df['has_zero_penalty']]
    if not zero_penalty_df.empty:
        sorted_df = zero_penalty_df.sort_values(by='cost')
    else:
        sorted_df = results_df.sort_values(by=['penalty_sum', 'cost'])

    best_k1 = sorted_df.head(1)
    return best_k1[['k1']]


def print_and_save_best_combinations_for_k(csv_file, num_top_combinations=10, output_file='result_grid_search_best_for_k.csv'):
    df = pd.read_csv(csv_file)
    
    df['penalty'] = df['penalty'].apply(lambda x: [float(i) for i in ast.literal_eval(x)])
    df['penalty_sum'] = df['penalty'].apply(sum)
    df['has_zero_penalty'] = df['penalty'].apply(lambda x: all(p == 0 for p in x))

    top_combinations_list = []

    for _, group_df in df.groupby('file'):
        zero_penalty_df = group_df[group_df['has_zero_penalty']]
        if not zero_penalty_df.empty:
            
            top_zero_penalty_combinations = zero_penalty_df.sort_values(by='cost').head(num_top_combinations)
            top_combinations_list.append(top_zero_penalty_combinations)
        else:
            
            top_combinations = group_df.sort_values(by=['penalty_sum', 'cost']).head(num_top_combinations)
            top_combinations_list.append(top_combinations)

    
    top_combinations_df = pd.concat(top_combinations_list)
    top_combinations_df.to_csv(output_file, index=False)



def tuning(directory_path):
    abs_directory_path = os.path.abspath(directory_path)
    k1_values = np.arange(150, 300)

    num_cores = 6
    k1_combinations = [k1 for k1 in k1_values]
    chunks = [k1_combinations[i::num_cores] for i in range(num_cores)]
    client = Client(n_workers=num_cores, threads_per_worker=1)

    futures = []
    for filename in os.listdir(abs_directory_path):
        if filename.endswith(".txt"): 
            file_path = os.path.join(abs_directory_path, filename)
            for chunk in chunks:
                future = client.submit(run_macs_for_k, file_path, chunk)
                futures.append(future)

    results = []
    for future in as_completed(futures):
        results.extend(future.result())

    client.close()

    results_df = pd.DataFrame(results)

    results_df['penalty'] = results_df['penalty'].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
        if isinstance(x, str) else [x]
    )
    results_df['penalty_sum'] = results_df['penalty'].apply(sum)

    results_df['has_zero_penalty'] = results_df['penalty'].apply(lambda x: all(p == 0 for p in x))

    results_df.to_csv('grid_search_results_for_k.csv', index=False)

    best_k1 = find_best_k1(results_df)
    best_k1.to_csv('best_value_k1.csv', index=False)

    print_and_save_best_combinations_for_k('grid_search_results_for_k.csv')


if __name__ == "__main__":
    tuning("../penality_instances")

