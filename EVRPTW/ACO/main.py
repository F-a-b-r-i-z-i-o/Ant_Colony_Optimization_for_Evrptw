import csv
import os
from multiprocessing import Pool
from macs import MultipleAntColonySystem
from evrptw_config import EvrptwGraph
import matplotlib.pyplot as plt


def run_macs(args):
    """
    Executes the Multiple Ant Colony System (MACS) algorithm for a given problem instance.

    Args:
    - args (tuple): A tuple containing the arguments needed for the MACS execution,
                    including iteration number, file path, and algorithm parameters.

    Returns:
    - tuple: A tuple containing the iteration number, the best C_test value found,
             and the history of C_test and time values.
    """
    iteration, file_path, max_iter, ants_num, alpha, beta, q0, k1, k2 = args
    graph = EvrptwGraph(file_path)
    macs_instance = MultipleAntColonySystem(graph, max_iter, ants_num, alpha, beta, q0, k1, k2)
    C_test, c_test_history, time_history = macs_instance.macs()
    return iteration, C_test, c_test_history, time_history

    

def plot_c_test(c_test_history, time_history, save_path=None):
    """
    Plots the history of the C_test value over time.

    Args:
    - c_test_history (list): A list of C_test values over time.
    - time_history (list): A list of time values corresponding to the C_test history.
    - save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, c_test_history, marker='o', color='b')
    plt.title("Variation of Dist over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Dist found")
    plt.grid(False)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def calculate_and_append_average(filename, results):
    """
    Calculates the average of the results and appends it to a CSV file.

    Args:
    - filename (str): Path to the CSV file where the results are stored.
    - results (list): A list of tuples containing execution numbers and their respective results.
    """
    if results:
        avg_result = sum(r[1] for r in results) / len(results)
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Average", avg_result])


if __name__ == "__main__":
    """
    Main execution block. Iterates over all instance files in a specified directory,
    runs the MACS algorithm for each instance, and stores the results.
    """
    directory_path = "test_instances"
    abs_directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_path)
    results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # Define algorithm parameters
    ants_num = 10
    max_iter = 400
    beta = 2
    alpha = 1
    q0 = 0.9
    num_runs = 15
    processes = 15
    k1 = 2
    k2 = 3

    for filename in sorted(os.listdir(abs_directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(abs_directory_path, filename)
            instance_folder = os.path.join(results_directory, filename.replace(".txt", ""))
            if not os.path.exists(instance_folder):
                os.makedirs(instance_folder)

            all_results = []
            args = [(i, file_path, max_iter, ants_num, alpha, beta, q0, k1, k2) for i in range(num_runs)]

            with Pool(processes=processes) as pool:
                for iteration, result, c_test_history, time_history in pool.imap_unordered(run_macs, args):
                    all_results.append((iteration + 1, result))
                    unique_graph_filename = os.path.join(instance_folder, f"graph_{iteration + 1}.png")
                    plot_c_test(c_test_history, time_history, save_path=unique_graph_filename)

            # Sort and write results to CSV
            csv_filename = os.path.join(instance_folder, "results.csv")
            with open(csv_filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Result"])
                sorted_results = sorted(all_results, key=lambda x: x[0])
                for execution_number, result in sorted_results:
                    writer.writerow([f"Execution{execution_number}", result])

            calculate_and_append_average(csv_filename, sorted_results)
            print(f"File {csv_filename} updated.")
