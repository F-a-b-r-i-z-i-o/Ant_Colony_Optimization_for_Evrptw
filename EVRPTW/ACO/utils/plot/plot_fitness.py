import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fitness_improvement_per_run(directory):
    """
    Plots fitness improvement over iterations for each run in the CSV files in the given directory.
    Saves the plots in a directory named 'plot_fitness' within the same location as the CSV file.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('fitness_result.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                   
                with open(file_path, 'r') as f:
                    last_line = f.readlines()[-1]
                average_last_iterations = float(last_line.split(',')[-1])

                df = df.iloc[:-1]

                # Create folder for save graphs
                graphs_directory = os.path.join(root, "plot_fitness")
                if not os.path.exists(graphs_directory):
                    os.makedirs(graphs_directory)

                # Name folder fot graphs titile 
                graph_title = os.path.basename(root)

                for run, run_df in df.groupby('run'):
                    if not run_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(run_df['improvement_iteration'], run_df['fitness_improvement'], marker='o')
                        ax.set_title(f'Fitness Improvement - {graph_title} - Run {run}')
                        ax.set_xlabel('Improvement Iteration')
                        ax.set_ylabel('Fitness Improvement')
                        ax.grid(True)

                        fig.subplots_adjust(bottom=0.2)
                        fig.text(0.5, 0.02, f'Average Fitness of Last Iterations: {average_last_iterations:.2f}', 
                                 ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.7, "pad":5})

                        plt.savefig(os.path.join(graphs_directory, f'{file[:-4]}_Run_{run}.png'))
                        plt.close(fig)

if __name__ == "__main__": 
    results_directory = '../ACO/results'
    plot_fitness_improvement_per_run(results_directory)

