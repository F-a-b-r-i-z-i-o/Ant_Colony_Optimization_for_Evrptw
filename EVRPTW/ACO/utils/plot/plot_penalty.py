import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_penalty_improvement_per_run(directory):
    """
    Plots penalty over iterations for each run in the CSV files in the given directory, 
    if there is a non-zero penalty value. Saves the plots in a directory named 'plot_penalty' 
    within the same location as the CSV file. Uses red line for the plots.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('fitness_result.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

        
                # Remove last line and verify penalty values
                df = df.iloc[:-1]
                if df['penalty'].eq(0).all():
                    continue  

                # Create folder for save graphs 
                graphs_directory = os.path.join(root, "plot_penalty_fix")
                if not os.path.exists(graphs_directory):
                    os.makedirs(graphs_directory)

                graph_title = os.path.basename(root)

                for run, run_df in df.groupby('run'):
                    if not run_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(run_df['improvement_iteration'], run_df['penalty'], marker='o', color='red')  # Red line
                        ax.set_title(f'Penalty Improvement - {graph_title} - Run {run}')
                        ax.set_xlabel('Improvement Iteration')
                        ax.set_ylabel('Penalty')
                        ax.grid(True)

                        plt.savefig(os.path.join(graphs_directory, f'{file[:-4]}_Run_{run}.png'))
                        plt.close(fig)

if __name__ == "__main__": 
    results_directory = '../../results/'
    plot_penalty_improvement_per_run(results_directory)
