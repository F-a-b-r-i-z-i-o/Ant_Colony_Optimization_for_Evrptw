import os
import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed
from acsd import AntColonySystem
from evrptw_config import EvrptwGraph
import pandas as pd
import yaml
import csv


def load_configs(general_config_file, aco_config_file) -> dict:
    configs = {}
    # Load general configuration
    with open(general_config_file, "r") as file:
        configs["general"] = yaml.safe_load(file)
    # Load ACO-specific configuration
    with open(aco_config_file, "r") as file:
        configs["aco"] = yaml.safe_load(file)["aco_config"]
    return configs


def run_macs(
    run, file_path, max_iter, ants_num, alpha, beta, q0, rho, k1, k2, apply_local_search
) -> tuple[list, list]:
    graph = EvrptwGraph(file_path, rho)
    macs_instance = AntColonySystem(
        graph, ants_num, max_iter, alpha, beta, q0, k1, k2, apply_local_search
    )
    (
        _,
        c_test_history,
        time_history,
        penalty_history,
        improvement_iter_history,
        num_vehicles,
        improvement_path,
    ) = macs_instance._ACS_DIST_G(apply_local_search)

    path_details = []
    for idx, path in enumerate(improvement_path):
        node_types = [graph.nodes[idx].node_type for idx in path]
        # Convert path indices to coordinates
        coordinates = graph.get_coordinates_from_path(path)
        path_detail = {
            "run": run,
            "numer_of_improvement": idx,
            "path": path,
            "node_types": node_types,
            "coordinates": coordinates,
            "num_vehicles": num_vehicles,  # Assuming num_vehicles is a list with the count per iteration
        }
        path_details.append(path_detail)

    # List to store improvements data for each run
    improvements = []
    for idx, improvement_time in enumerate(improvement_iter_history):
        improvement_data = {
            "run": run,
            "improvement_iteration": improvement_time,
            "fitness_improvement": c_test_history[idx],
            "penalty": penalty_history[idx],
            "time": time_history[idx],
            "num_vehicles": num_vehicles,
            "path": improvement_path[idx],
        }
        improvements.append(improvement_data)

    return improvements, path_details


def save_results_to_csv(results, filename) -> None:
    df = pd.DataFrame(results)
    dask_df = dd.from_pandas(df, npartitions=1)
    dask_df.to_csv(filename, index=False, single_file=True)


def save_path_details_to_csv(path_details, filename) -> None:
    df = pd.DataFrame(path_details)
    df["coordinates"] = df["coordinates"].apply(
        lambda coords: ";".join([f"({x},{y})" for x, y in coords])
    )
    dask_df = dd.from_pandas(df, npartitions=1)
    dask_df.to_csv(filename, index=False, single_file=True)


def calculate_and_append_average(filename) -> None:
    df = pd.read_csv(filename)
    last_improvements = df.groupby("run").last()["fitness_improvement"]
    # Calculate the average of these last improvements
    avg_fitness = last_improvements.mean()

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Average Fitness of Last Iterations", avg_fitness])


if __name__ == "__main__":
    # Define directory paths
    directory_path = "paper_total_instances"
    abs_directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), directory_path
    )
    results_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )

    # Load configuration files
    general_config_file = "config/dask_config.yaml"
    aco_config_file = "config/aco_config.yaml"
    configs = load_configs(general_config_file, aco_config_file)

    # Extract ACO configuration settings
    aco_config = configs["aco"]
    num_runs = aco_config["num_runs"]
    ants_num = aco_config["ants_num"]
    max_iter = aco_config["max_iter"]
    alpha = aco_config["alpha"]
    beta = aco_config["beta"]
    q0 = aco_config["q0"]
    k1 = aco_config["k1"]
    k2 = aco_config["k2"]
    rho = aco_config["rho"]
    apply_local_search = aco_config["apply_local_search"]
    save_path_details = aco_config["save_path_details"]

    # Initialize Dask client with configuration
    dask_config = configs["general"]["dask_config"]
    client = Client(**dask_config)

    # Process each file in the directory
    for filename in sorted(os.listdir(abs_directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(abs_directory_path, filename)
            instance_folder = os.path.join(
                results_directory, filename.replace(".txt", "")
            )
            if not os.path.exists(instance_folder):
                os.makedirs(instance_folder)

            delayed_runs = []

            # Prepare run with MACS config
            for i in range(num_runs):
                delayed_run = delayed(run_macs)(
                    i,
                    file_path,
                    max_iter,
                    ants_num,
                    alpha,
                    beta,
                    q0,
                    rho,
                    k1,
                    k2,
                    apply_local_search,
                )
                delayed_runs.append(delayed_run)

            # Compute all results in parallel
            all_improvements, all_path_details = zip(*dask.compute(*delayed_runs))

            # Flatten the results for CSV and extract last improvements for average calculation
            combined_improvements = []
            for improvement in all_improvements:
                combined_improvements.extend(improvement)

            # Save all results to CSV and calculate average of last iterations
            csv_filename = os.path.join(instance_folder, "fitness_result.csv")
            save_results_to_csv(combined_improvements, csv_filename)
            calculate_and_append_average(csv_filename)
            print(f"File {csv_filename} updated.")

            # Save path details to CSV
            if save_path_details:
                combined_path_details = []
                for path_detail in all_path_details:
                    combined_path_details.extend(path_detail)
                path_details_filename = os.path.join(
                    instance_folder, "path_results.csv"
                )
                save_path_details_to_csv(combined_path_details, path_details_filename)
                print(f"File {path_details_filename} updated.")

    # Close the Dask client
    client.close()
