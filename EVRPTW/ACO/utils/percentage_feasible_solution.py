import os
import pandas as pd

def calculate_feasibility_percentage(directory):
    """
    Calculate the feasibility percentage of solutions across all CSV files in the given directory.

    Args:
    - directory (str): The path to the directory containing subdirectories with CSV files.

    Returns:
    - list of dicts: A list containing dictionaries with filepath and feasibility percentage.
    """
    feasibility_results = []

    # Iterate through all subdirectories in the given directory
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('fitness_result.csv'):
                file_path = os.path.join(subdir, file)
                
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Check if the 'penalty' column exists in the DataFrame
                if 'penalty' in df.columns:
                    # Calculate the percentage of solutions with zero penalty
                    zero_penalty_count = df['penalty'].apply(lambda x: x == 0).sum()
                    total_count = len(df)
                    feasibility_percentage = (zero_penalty_count / total_count) * 100

                    # Add the result to the feasibility_results list
                    feasibility_results.append({
                        'filepath': file_path,
                        'feasibility_percentage': feasibility_percentage
                    })

    return feasibility_results

def write_results_to_csv(results, output_filename):
    """
    Write the feasibility results to a CSV file and append the overall average feasibility percentage.

    Args:
    - results (list of dicts): The feasibility results to write.
    - output_filename (str): The name of the output CSV file.
    """
    df = pd.DataFrame(results)

    average_feasibility = df['feasibility_percentage'].mean()
    
    df.to_csv(output_filename, index=False)
    
    with open(output_filename, 'a') as f:
        f.write(f"\nAverage Feasibility Percentage,{average_feasibility}")

if __name__ == "__main__":
    # Existing directory containing the results
    results_directory = "../results_k1_k2_no_ls/"
    
    # New directory for saving the summary file
    summary_directory = "../percentage_feasible_solution/"
    if not os.path.exists(summary_directory):
        os.makedirs(summary_directory)

    # Define the output file name in the new directory
    output_filename = os.path.join(summary_directory, "final_feasibility_results.csv")

    # Calculate the feasibility percentages
    results = calculate_feasibility_percentage(results_directory)

    # Write the results to the output file in the new directory
    write_results_to_csv(results, output_filename)
    print(f"Feasibility results written to {output_filename}.")
