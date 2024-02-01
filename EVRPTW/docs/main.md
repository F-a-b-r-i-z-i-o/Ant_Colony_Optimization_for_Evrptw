# Ant Colony System Simulation

This script implements a simulation for solving the Electric Vehicle Routing Problem with Time Windows (EVRPTW) using a Multiple Ant Colony System (MACS).

## Dependencies

- `numpy`
- `dask`
- `pandas`
- `yaml`
- `csv`
- `os`
- `time`

## Classes

- `AntColonySystem`: From the `acsd` module, simulates an ant colony system.
- `EvrptwGraph`: From the `evrptw_config` module, represents the EVRPTW graph.

## Functions

### `load_configs(general_config_file, aco_config_file)`

Loads configuration settings from YAML files.

- **Parameters:**
  - `general_config_file`: Path to the general configuration file.
  - `aco_config_file`: Path to the ACO-specific configuration file.
- **Returns:**
  - `dict`: Configuration data.

### `run_macs(run, file_path, max_iter, ants_num, alpha, beta, q0, k1)`

Runs a single instance of the MACS simulation.

- **Parameters:**
  - `run`: Run number.
  - `file_path`: Path to the data file.
  - `max_iter`: Maximum number of iterations.
  - `ants_num`: Number of ants.
  - `alpha`: Alpha parameter for MACS.
  - `beta`: Beta parameter for MACS.
  - `q0`: Q0 parameter for MACS.
  - `k1`: K1 parameter for MACS.
- **Returns:**
  - `tuple`: Improvements and path details.

### `save_results_to_csv(results, filename)`

Saves results to a CSV file using Dask.

- **Parameters:**
  - `results`: Results data to save.
  - `filename`: Output file name.

### `save_path_details_to_csv(path_details, filename)`

Saves path details to a CSV file using Dask.

- **Parameters:**
  - `path_details`: Path details data to save.
  - `filename`: Output file name.

### `calculate_and_append_average(filename)`

Calculates and appends the average of the last improvements to the CSV file.

- **Parameters:**
  - `filename`: The CSV file to update.

## Main Execution

- Configures and initializes the Dask client.
- Processes each `.txt` file in the specified directory for EVRPTW instances.
- Executes multiple runs of MACS for each instance.
- Saves results and path details to CSV files.
- Calculates and appends the average fitness of the last iterations.

## Usage

Run the script in a Python environment where all dependencies are installed. Ensure that the necessary configuration files and EVRPTW instance files are available in the specified directories.
