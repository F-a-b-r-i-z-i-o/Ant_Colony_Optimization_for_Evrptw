import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import imageio.v2

def create_gif_for_path_evolution(directory, gif_name='path_evolution.gif', fps=1):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'path_results.csv':
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                graphs_directory = os.path.join(root, "plot_paths")
                if not os.path.exists(graphs_directory):
                    os.makedirs(graphs_directory)

                # Extract instance name from the directory or file path
                instance_name = os.path.basename(root)  

                images = []

                for (run, num_improvement), group_df in df.groupby(['run', 'numer_of_improvement']):
                    fig, ax = plt.subplots(figsize=(15, 10))

                    coordinates_list = []
                    node_types_list = []

                    for index, row in group_df.iterrows():
                        coordinates_str = "[" + row['coordinates'].replace(");(", "),(") + "]"
                        coordinates = ast.literal_eval(coordinates_str)
                        node_types = ast.literal_eval(row['node_types'])

                        coordinates_list.extend(coordinates)
                        node_types_list.extend(node_types)

                        for coord, node_type in zip(coordinates, node_types):
                            color = 'lightgrey'
                            if node_type == 'd':
                                color = 'blue'
                            elif node_type == 'f':
                                color = 'green'
                            elif node_type == 'c':
                                color = 'red'

                            ax.scatter(*coord, color=color, s=70)

                    if coordinates_list:
                        x_coords, y_coords = zip(*coordinates_list)
                        ax.plot(x_coords, y_coords, linestyle='--', color='lightgrey', linewidth=2, alpha=0.6)

                    # Include the instance name in the title
                    ax.set_title(f'{instance_name} - Run {run} - Improvement {num_improvement}')
                    ax.set_xlabel('Coordinate X')
                    ax.set_ylabel('Coordinate Y')
                    ax.grid(False)
                    
                    # Add legend
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', label='Depot (d)', markerfacecolor='blue', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Fuel Station (f)', markerfacecolor='green', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Customer (c)', markerfacecolor='red', markersize=10)
                    ]

                    ax.legend(handles=legend_elements)

                    img_path = os.path.join(graphs_directory, f'Run_{run}_Improvement_{num_improvement}.png')
                    plt.savefig(img_path)
                    plt.close(fig)

                    images.append(imageio.imread(img_path))

                gif_path = os.path.join(graphs_directory, gif_name)
                imageio.mimsave(gif_path, images, fps=fps)

                print(f"GIF created at {gif_path}")

if __name__ == "__main__": 
    results_directory = '../../results/'
    create_gif_for_path_evolution(results_directory, gif_name='path_evolution.gif', fps=1)
