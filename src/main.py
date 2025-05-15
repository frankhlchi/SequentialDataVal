import numpy as np
import torch
import argparse
import yaml
import math
from pathlib import Path
from opendataval.dataloader import add_gauss_noise, mix_labels
from opendataval.metrics import Metrics
from opendataval.experiment import ExperimentMediator
from opendataval.dataval import (
    AME, DVRL, BetaShapley, DataBanzhaf, DataOob, DataShapley,
    InfluenceSubsample, KNNShapley, LavaEvaluator, DataEvaluator,
    LeaveOneOut, RandomEvaluator,  DataBanzhaf, DVRL
)
from opendataval.dataval.margcontrib import MonteCarloSampler

from evaluators import (
    BipartiteMatchingEvaluator,
    PredictionBasedMatchingEvaluator,
    LearnableEmbeddingMatchingEvaluator,
    WeightedBipartiteEvaluator,
    DualThresholdTripartiteEvaluator
)


from utils.plotting import plot_results_with_ci
from utils.data import set_seed, create_output_dir
from utils.data import remove_points_one_by_one 

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_PATH = PROJECT_ROOT / "config" / "base_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description='Run data valuation experiments')
    

    parser.add_argument('--dataset', type=str, default='fried', 
                        help='Dataset name (default: fried)')
    parser.add_argument('--train_count', type=int, default=None,
                        help='Number of training samples')
    parser.add_argument('--valid_count', type=int, default=None,
                        help='Number of validation samples')
    parser.add_argument('--test_count', type=int, default=None,
                        help='Number of test samples')
    

    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    

    parser.add_argument('--noise_rate', type=float, default=None,
                        help='Noise rate')
    parser.add_argument('--noise_mu', type=float, default=None,
                        help='Noise mean')
    parser.add_argument('--noise_sigma', type=float, default=None,
                        help='Noise standard deviation')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at {CONFIG_PATH}. "
            "Please make sure base_config.yaml exists in the config directory."
        )

    if args.dataset == 'cifar10-embeddings':
        CONFIG_PATH = PROJECT_ROOT / "config" / "cifar_config.yaml"

    print ('CONFIG_PATH',CONFIG_PATH)
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
            
    print ('config', config['experiment']['train_count'])
    print ('args.train_count', args.train_count)

    train_count = args.train_count or config['experiment']['train_count']
    valid_count = args.valid_count or config['experiment']['valid_count']
    test_count = args.test_count or config['experiment']['test_count']
    device = args.device or config['experiment']['device']
    model = args.model or config['model']['name']
    
    noise_kwargs = {
        'noise_rate': args.noise_rate or config['noise']['kwargs']['noise_rate'],
        'mu': args.noise_mu or config['noise']['kwargs']['mu'],
        'sigma': args.noise_sigma or config['noise']['kwargs']['sigma']
    }

    noise_kwargs = {'noise_rate': 0.0}

    
    set_seed(args.seed)
    print(f"Running experiment with seed={args.seed} on {device}")
    

    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=args.dataset,
        model_name=model,
        train_count=train_count,
        valid_count=valid_count, 
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        metric_name="accuracy",
        device=device
    )
    

    mc_sampler = MonteCarloSampler(
        mc_epochs=math.ceil(1000/train_count),
        min_cardinality=1,
        cache_name="cached",
        random_state=None
    )
    

    data_evaluators = [
        RandomEvaluator(),
        LeaveOneOut(),
        InfluenceSubsample(num_models=1000),
        KNNShapley(k_neighbors=valid_count),
        DataShapley(sampler=mc_sampler, cache_name=f"cached"),
        BetaShapley(sampler=mc_sampler, cache_name=f"cached"),
        DataBanzhaf(num_models=1000),
        AME(num_models=math.ceil(1000/4)),
        DVRL(rl_epochs=math.ceil(1000/32)),
        DataOob(num_models=1000),
        BipartiteMatchingEvaluator(n_samples=1000, random_state=args.seed),
        PredictionBasedMatchingEvaluator(num_samples=20, num_trials=50),
       DualThresholdTripartiteEvaluator(n_samples=1000, random_state=args.seed)
    ]
    

    print(f"Computing data values for seed {args.seed}...")
    exper_med = exper_med.compute_data_values(data_evaluators)
    

    output_dir = Path("results") / args.dataset / f"seed_{args.seed}"
    create_output_dir(output_dir)
    exper_med.set_output_directory(output_dir)
    

    print("Running removal experiment...")
    df_removal, _ = exper_med.plot(remove_points_one_by_one)
    df_removal['axis'] = (df_removal['axis'] * train_count).astype(int)
    df_removal.to_csv(output_dir / "addition_experiment_results.csv")
    
    print(f"Results saved to {output_dir}")
    
