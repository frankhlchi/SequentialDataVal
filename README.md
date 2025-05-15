# Sequential Data Selection with Data Values Project

This project implements various data valuation methods and evaluates their performance on different datasets.

## Main Files

- `main.py`: The entry point of the project. It parses command-line arguments, loads the configuration, creates the experiment, and runs the data valuation methods. It also saves the results to CSV files.

- `utils/data.py`: Contains utility functions for data processing, model evaluation, and experiment setup. Key functions include:
  - `evaluate_model`: Evaluates the performance of a scikit-learn model.
  - `set_seed`: Sets the random seed for reproducibility.
  - `create_output_dir`: Creates the output directory for saving results.
  - `process_data_for_sklearn`: Prepares data for use with scikit-learn models.
  - `train_model_on_subset`: Trains a scikit-learn model on a subset of data.
  - `remove_points_one_by_one`: Evaluates model performance by removing data points one by one.

- `utils/plotting.py`: Contains functions for plotting the aggregated results with confidence intervals across multiple random seeds.

- `evaluators/`: A directory containing custom data valuation methods implemented as subclasses of `DataEvaluator`. These include:
  - `BipartiteMatchingEvaluator`: A data valuation method based on category-aware greedy bipartite matching.

## Usage

To run the project, use the `main.py` script with the desired command-line arguments. The script loads the configuration from `config/base_config.yaml` and `config/cifar_config.yaml` (for CIFAR-10 dataset), and saves the results in the `results/` directory.

## Configuration

The project uses YAML files for configuration. The config/base_config.yaml file contains the default settings, which can be overridden by dataset-specific configurations like config/cifar_config.yaml. The configuration files specify the dataset, model, and experiment settings.

## Results

The results of the data valuation experiments are saved in the results/ directory, organized by dataset and random seed. The addition_experiment_results.csv file in each seed directory contains the performance metrics for each data valuation method when data points are added one by one.

The utils/plotting.py script can be used to plot the aggregated results with confidence intervals across multiple random seeds. The plots are saved in the results/{dataset}/aggregate/ directory.

```

This README provides an overview of the main Python files, their functionalities, and how to use the project. It also explains the configuration and results structure. Feel free to modify and expand the README based on your specific project requirements.
