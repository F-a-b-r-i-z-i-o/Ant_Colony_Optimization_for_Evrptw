# Ant Colony Optimization for EVRPTW 🌐

Welcome Ant Colony Optimization (ACO) project, where advanced algorithms and electric vehicles meet to solve the Electric Vehicle Routing Problem with Time Windows (EVRPTW). 🚚⏱️

## Project Overview 📜

This repository contains a Python-based implementation of ACO algorithms, designed to optimize routing paths for electric vehicles considering specific time windows and recharging requirements.

## Documentation 📖

For more detailed information and usage instructions, check out our [documentation](/EVRPTW/docs/).

## Installation and Dependencies 🔧

Ensure Python is installed along with the following packages:
- `numpy`
- `dask`
- `pandas`
- `yaml`
- `csv`

**Steps for Installation**:

1. **Clone the Repository**: 
   - Use the command `git clone https://github.com/F-a-b-r-i-z-i-o/Ant_Colony_Optimization_for_Evrptw.git` to clone the EVRPTW repository to your local machine.
2. **Create and Activate Virtual Environment**:
   - Navigate to the EVRPTW directory (`cd EVRPTW`).
   - Create a virtual environment named `env` within this directory. Use the command `python3 -m venv env`.
   - Activate the virtual environment using the `script.sh` script with the command `. ./active_venv.sh`.
3. **Install Dependencies**:
   - Install required dependencies by executing `pip3 install -r requirements.txt`. This will install all the necessary packages as listed in the `requirements.txt` file.

By following these steps, you'll set up the necessary environment to run the Ant Colony Optimization simulations for EVRPTW.

## Results 📊

### CPLEX vs MACS vs ACSD BEST VALUES FOUND  

<div align="center">

   | **E-VRPTW**  | **CPLEX best** | **secs** | **MACS best** | **secs**  | **ACSD best** | **secs**  | **$\Delta\%$** |
   |----------|------------|------|-----------|-------|-----------|-------|------------|
   | C101-5   | 257.75     | 81   | 257.75    | 0.002 | N/A       | 0.0   | 0.0        |
   | C103-5   | 176.05     | 5    | 176.05    | 0.21  | N/A       | 0.0   | 0.0        |
   | **C206-5** | **242.55** | 518  | **242.55**| 0.008 | **242.55**| 0.47  | 0.0        |
   | **C208-5** | **158.48** | 15   | **158.48**| 0.0002| **158.48**| 0.037 | 0.0        |
   | R104-5   | 136.69     | 1    | 136.69    | 0.005 | 140.28    | 0.0351| 2.26       |
   | **R105-5** | **156.08** | 3    | **156.08** | 0.001 | **156.08** | 0.082 | 0.0        |
   | **R202-5** | **128.78** | 1    | **128.78** | 0.08  | **128.28** | 0.97  | 0.0        |
   | R203-5   | 179.06     | 5    | 179.06    | 1.11  | 197.77    | 1.79  | 10.44      |
   | **RC105-5**| **241.30** | 764  | **241.30** | 2.37  | **241.30** | 5.87  | 0.0        |
   | **RC108-5**| **253.93** | 311  | **253.93** | 0.002 | **253.93** | 0.057 | 0.0        |
   | RC204-5  | 176.39     | 54   | 176.39    | 0.001 | 179.81    | 0.089 | 1.93       |
   | RC208-5  | 167.98     | 21   | 167.98    | 0.003 | 181.67    | 0.034 | 8.15       |
   | C101-10  | 393.76     | 171  | 393.76    | 4.53  | N/A       | 0.0   | 0.0        |
   | C104-10  | 273.93     | 360  | 273.93    | 24.1  | N/A       | 0.0   | 0.0        |
   | **C202-10**| **304.06** | 300  | **304.06** | 2.85  | **304.06** | 4.90  | 0.0        |
   | C205-10  | 228.28     | 4    | 228.28    | 20.70 | 287.29    | 93.89 | 25.84      |
   | R102-10  | 249.19     | 389  | 249.19    | 1.57  | 268.87    | 20.87 | 7.89       |
   | **R103-10**| **207.05** | 119  | **207.05** | 13.50 | **207.05** | 35.68 | 0.0        |
   | R201-10  | 241.51     | 177  | 241.51    | 1.14  | 246.63    | 13.89 | 2.11       |
   | R203-10  | 218.21     | 573  | 218.21    | 15.45 | 302.78    | 56.89 | 38.75      |
   | **RC102-10**| **423.51** | 810  | **423.51** | 11.85 | **423.51** | 78.98 | 0.0        |
   | RC108-10 | 345.93     | 39   | 345.93    | 7.99  | 398.87    | 30.87 | 15.30      |
   | **RC201-10**| **412.86** | 7200 | **412.86** | 0.02  | **412.86** | 89.67 | 0.0        |
   | RC205-10 | 325.98     | 399  | 325.98    | 25.57 | N/A       | 0.0   | 0.0        |
   | C103-15  | 348.29     | 7200 | 348.29    | 24.36 | N/A       | 0.0   | 0.0        |
   | C106-15  | 275.13     | 17   | 275.13    | 21.88 | N/A       | 0.0   | 0.0        |
   | C202-15  | 383.62     | 7200 | 383.62    | 59.46 | 555.58    | 363.68| 44.82      |
   | C208-15  | 300.55     | 5060 | 300.55    | 44.1  | 398.63    | 237.87| 32.63      |
   | R102-15  | 413.93     | 7200 | 413.93    | 25.84 | 415.15    | 234.68| 0.29       |
   | R105-15  | 336.15     | 7200 | 336.15    | 13.42 | 403.20    | 398.78| 19.95      |
   | R202-15  | 358.00     | 7200 | 358.00    | 7.32  | 400.26    | 287.89| 11.80      |
   | R209-15  | 313.24     | 7200 | 313.24    | 9.01  | 444.78    | 345.89| 41.99      |
   | RC103-15 | 397.67     | 7200 | 397.67    | 24.52 | 500.12    | 456.82| 25.76      |
   | RC108-15 | 370.25     | 7200 | 370.25    | 26.96 | 467.86    | 554.89| 26.36      |
   | RC202-15 | 394.39     | 7200 | 394.39    | 73.38 | 467.56    | 556.98| 18.55      |
   | RC204-15 | 407.45     | 7200 | 382.22    | 15.51 | 409.89    | 666.89| 7.23       |

</div>

_Note: "N/A" indicates that the instance does not generate eligible solutions._

Other results are available [Results](/EVRPTW/results/)

## Path Visualization 🔄

![Path Evolution](/EVRPTW/gif/path_evolution.gif)

## Fitness Trend Graph 📉

![Fitness Trend](/EVRPTW/img/fitness_result_Run_2.png)

## Paper Citations 📄
- Michalis Mavrovouniotis. "A Multiple Ant Colony System for the Electric Vehicle Routing Problem with Time Windows". KIOS Research and Innovation Center of Excellence, Department of Electrical and Computer Engineering, University of Cyprus, Nicosia, Cyprus. [View Paper](https://ieeexplore.ieee.org/document/10022257).

## Contributing 🤝

Contributions to enhance the project are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

## License 📄

This project is released under [MIT License](/LICENSE).

---

*Enjoy 2F_*
