import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
from scipy import stats, special
import warnings
from itertools import combinations
import tqdm
import multiprocessing as mp
from functools import partial
warnings.filterwarnings("ignore")
import math

def create_base_mean_vectors(base_distance=2):
    height = base_distance * np.sqrt(3)
    return [
        [-base_distance, -base_distance],
        [0, -base_distance + height],
        [base_distance, -base_distance]
    ]

def generate_gmm_data(sizes, means, covs):
    X, y = [], []
    for i, (mean, cov, size) in enumerate(zip(means, covs, sizes)):
        X_class = np.random.multivariate_normal(mean, cov, size)
        X.append(X_class)
        y.append(np.full(size, i))
    return np.vstack(X), np.hstack(y)

def propagate_features_gnn_style(X_original, X_current, y, proportion):
    if proportion == 0.0:
        return X_current.copy()
    X_new = np.zeros_like(X_current)
    for class_label in np.unique(y):
        mask = (y == class_label)
        global_indices = np.where(mask)[0]
        original_class_points = X_original[mask]
        current_class_points = X_current[mask]
        n_points = len(original_class_points)
        if proportion == 1.0:
            n_neighbors = n_points
        else:
            n_neighbors = max(1, min(n_points-1, int(proportion * n_points)))
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            distances[i] = np.linalg.norm(original_class_points - original_class_points[i], axis=1)
        for i in range(n_points):
            neighbor_indices = np.argsort(distances[i])[1:n_neighbors+1]
            global_i = global_indices[i]
            X_new[global_i] = np.mean(current_class_points[neighbor_indices], axis=0)
    return X_new

def compute_subset_utility(subset, X_train, y_train, X_val, y_val):
    try:
        model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
        model.fit(X_train[list(subset)], y_train[list(subset)])
        utility = model.score(X_val, y_val)
        return subset, utility
    except ValueError:
        unique_classes = np.unique(y_train[list(subset)])
        num_classes = len(unique_classes)
        if num_classes == 1:
            predictions = np.full_like(y_val, unique_classes[0])
            utility = np.mean(predictions == y_val)
            return subset, utility
        elif num_classes == 2:
            try:
                binary_model = LogisticRegression(max_iter=1000, random_state=42)
                binary_model.fit(X_train[list(subset)], y_train[list(subset)])
                utility = binary_model.score(X_val, y_val)
                return subset, utility
            except:
                majority_class = np.argmax(np.bincount(y_train[list(subset)]))
                predictions = np.full_like(y_val, majority_class)
                utility = np.mean(predictions == y_val)
                return subset, utility

def compute_all_subset_utilities_parallel(X_train, y_train, X_val, y_val, n_processes=None):
    n = len(X_train)
    all_subsets = []
    for size in range(1, n + 1):
        all_subsets.extend(combinations(range(n), size))
    compute_utility_partial = partial(
        compute_subset_utility,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(compute_utility_partial, all_subsets),
            total=len(all_subsets),
            desc="Computing subset utilities"
        ))
    return dict(results)

def compute_values(subset_utilities, n, method='shapley'):
    values = np.zeros(n)
    if method == 'beta':
        alpha = beta = 0.5
        p_k_list = [special.beta(k+beta, (n-k-1)+alpha)/special.beta(alpha,beta) for k in range(n)]
        total_list = [special.comb(n-1, k)*p_k_list[k] for k in range(n)] 
        sum_all = np.sum(total_list)
    for i in range(n):
        marginal_contributions = []
        subset_sizes = []
        for subset in subset_utilities:
            if i not in subset:
                new_subset = tuple(sorted(list(subset) + [i]))
                k = len(subset)
                margin = subset_utilities[new_subset]
                if k > 0:
                    margin -= subset_utilities[subset]
                marginal_contributions.append(margin)
                subset_sizes.append(k)
        if method == 'shapley':
            weights = [math.factorial(k) * math.factorial(n-k-1) / math.factorial(n) for k in subset_sizes]
        elif method == 'banzhaf':
            weights = [1/2**(n-1) for _ in subset_sizes]
        elif method == 'beta':
            weights = [(p_k_list[k]/sum_all) / special.comb(n-1, k) for k in subset_sizes]
        values[i] = np.sum([w * m for w, m in zip(weights, marginal_contributions)])
    return values

def run_single_experiment(seed, base_distance=2, variance=2, class_sizes=[5,5,5], val_sizes=[1000,1000,1000], test_sizes=[1000,1000,1000], proportions=[0.1,0.3,0.5,0.7,1.0]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_means = create_base_mean_vectors(base_distance)
    base_cov = [[variance, 0], [0, variance]]
    train_covs = [base_cov.copy() for _ in range(3)]
    X_train, y_train = generate_gmm_data(class_sizes, base_means, train_covs)
    X_val, y_val = generate_gmm_data(val_sizes, base_means, train_covs)
    X_test, y_test = generate_gmm_data(test_sizes, base_means, train_covs)
    X_original = X_train.copy()
    X_current = X_train.copy()
    X_train_history = [X_train.copy()]
    for proportion in proportions:
        X_current = propagate_features_gnn_style(X_original, X_current, y_train, proportion)
        X_train_history.append(X_current.copy())
    return X_train_history, y_train

BASE_DISTANCE = 2
VARIANCE = 2
PROPORTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_SIZES = [6, 6, 6]
all_X_histories = []
all_y_trains = []
for seed in range(20):
    X_history, y_train = run_single_experiment(seed, BASE_DISTANCE, VARIANCE, CLASS_SIZES)
    all_X_histories.append(X_history)
    all_y_trains.append(y_train)
