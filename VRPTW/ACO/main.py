import os
from vrptw_config import VrptwGraph
from aco import BasicACO

if __name__ == '__main__':
    # Define the relative base directory and file name
    relative_base_dir = 'solomon-100'
    file_name = 'c101.txt'

    # Get the absolute path of the base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, relative_base_dir)

    # Construct the absolute file path
    file_path = os.path.join(base_dir, file_name)

    # Parameters
    ants_num = 10
    max_iter = 100
    alpha = 1
    beta = 2
    q0 = 0.9

    # Initialize the graph and ACO algorithm
    graph = VrptwGraph(file_path)
    aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, alpha=alpha, beta=beta, q0=q0)

    # Run the ACO algorithm
    aco._basic_aco()

