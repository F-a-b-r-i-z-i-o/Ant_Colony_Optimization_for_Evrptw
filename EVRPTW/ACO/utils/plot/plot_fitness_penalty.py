import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fitness_penalty_combined(directory):
    """
    Plots fitness and penalty over iterations for each run in the CSV files in the given directory,
    ensuring that the penalty axis is scaled independently from the fitness values.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('fitness_result.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Remove the last line and check for non-zero penalty values
                df = df.iloc[:-1]
                if df['penalty'].eq(0).all():
                    continue  # Skip the file if all penalty values are zero

                # Create a directory for saving the plots
                graphs_directory = os.path.join(root, "combined_plots")
                if not os.path.exists(graphs_directory):
                    os.makedirs(graphs_directory)

                graph_title = os.path.basename(root)

                for run, run_df in df.groupby('run'):
                    if not run_df.empty:
                        fig, ax1 = plt.subplots(figsize=(10, 6))

                        # Plot fitness on the primary y-axis
                        ax1.set_xlabel('Improvement Iteration')
                        ax1.set_ylabel('Fitness', color='tab:blue')
                        ax1.plot(run_df['improvement_iteration'], run_df['fitness_improvement'], 
                                 'o-', color='tab:blue', markersize=5, linewidth=2)
                        ax1.tick_params(axis='y', labelcolor='tab:blue')

                        # Set up the secondary y-axis for penalty with an independent scale
                        ax2 = ax1.twinx()
                        ax2.set_ylabel('Penalty', color='tab:red')
                        # Scale the penalty axis independently
                        ax2.set_ylim([0, max(run_df['penalty']) * 2.5])  # Give some headroom for the penalty values
                        ax2.plot(run_df['improvement_iteration'], run_df['penalty'],
                                 'o-', color='tab:red', markersize=5, linewidth=2)
                        ax2.tick_params(axis='y', labelcolor='tab:red')

                        plt.title(f'Fitness and Penalty Improvement - {graph_title} - Run {run}')
                        plt.grid(True)

                        plt.savefig(os.path.join(graphs_directory, f'{file[:-4]}_Run_{run}_Combined.png'))
                        plt.close(fig)


if __name__ == "__main__": 
    results_directory = '../../results/'
    plot_fitness_penalty_combined(results_directory)