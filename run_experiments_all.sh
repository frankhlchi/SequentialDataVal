#!/bin/bash


datasets=("fried" "electricity" "election" "nomao" "MiniBooNE" "digits" "bbc-embeddings" "adult" "2dplanes" "pol")
seeds=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
max_parallel=20  


run_experiment() {
    local seed=$1
    local dataset=$2
    local log_file="logs/${dataset}_seed_${seed}.log"
    echo "Running experiment with dataset $dataset, seed $seed"
    python src/main.py --seed $seed --dataset $dataset > $log_file 2>&1
}


for dataset in "${datasets[@]}"; do
    results_dir="./results/$dataset"
    if [ -d "$results_dir" ]; then
        echo "Deleting existing results directory: $results_dir"
        rm -rf "$results_dir"
    fi
done


mkdir -p logs

# Export the function for parallel execution
export -f run_experiment

for dataset in "${datasets[@]}"; do
    echo "Starting experiments for dataset: $dataset"

    temp_file=$(mktemp)
    for seed in "${seeds[@]}"; do
        echo "$seed $dataset" >> "$temp_file"
    done
    

    cat "$temp_file" | xargs -P "$max_parallel" -n 2 bash -c 'run_experiment "$@"' _
    

    rm "$temp_file"
    

    echo "Generating aggregate results for dataset: $dataset"
    python src/utils/plotting.py --dataset "$dataset"
done

echo "All experiments completed!"