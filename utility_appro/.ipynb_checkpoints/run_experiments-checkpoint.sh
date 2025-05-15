#!/bin/bash
# Define all datasets and seeds
datasets=("fried" "electricity" "election" "nomao" "MiniBooNE" "digits" "2dplanes" "bbc-embeddings")
seeds=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
max_parallel=20  # Maximum parallel processes

# Define function to run a single experiment
run_experiment() {
   local seed=$1
   local dataset=$2
   local log_file="logs/${dataset}_seed_${seed}.log"
   echo "Running experiment with dataset $dataset, seed $seed"
   python src/main.py --dataset $dataset --seed $seed --num_samples 1000 --train_size 50 --valid_size 50 > $log_file 2>&1
}

# Remove existing directories if they exist and create new ones
rm -rf logs aggregate_results
mkdir -p logs
mkdir -p aggregate_results

# Create result directories for each dataset
for dataset in "${datasets[@]}"; do
   rm -rf "results/$dataset"
   mkdir -p "results/$dataset"
done

# Export the function for parallel execution
export -f run_experiment

# Run experiments for each dataset
for dataset in "${datasets[@]}"; do
   echo "Starting experiments for dataset: $dataset"
   
   # Create a temporary file with all seeds for current dataset
   temp_file=$(mktemp)
   for seed in "${seeds[@]}"; do
       echo "$seed $dataset" >> "$temp_file"
   done
   
   # Use xargs to run experiments in parallel
   cat "$temp_file" | xargs -P "$max_parallel" -n 2 bash -c 'run_experiment "$@"' _
   
   # Remove temporary file
   rm "$temp_file"
   
   # Generate aggregate results for current dataset
   echo "Generating aggregate results for dataset: $dataset"
   python src/utils/plotting.py --dataset "$dataset"
done

echo "All experiments completed!"