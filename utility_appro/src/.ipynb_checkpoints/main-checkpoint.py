import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
import random
import torch
import time
import argparse
import cvxpy as cp
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import DataFetcher
from sklearn.preprocessing import StandardScaler
from opendataval.dataval.api import DataEvaluator, ModelMixin
import scipy.stats as stats
from scipy.special import comb, beta
from torch.utils.data import Subset
import warnings
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

# Ensure we can import local modules
module_path = os.path.abspath(os.path.join(os.getcwd(), "src"))
if module_path not in sys.path:
   sys.path.append(module_path)

# Import local modules
from utils.data import process_data_for_sklearn

# Define evaluator class
class BaseDataEvaluator(DataEvaluator, ModelMixin):
   """Base class for all evaluator implementations"""
   
   def __init__(self, n_samples=50, random_state=None):
       self.n_samples = n_samples
       self.random_state = np.random.RandomState(random_state)
       
# Function to generate random subsets and calculate true utility
def generate_random_subsets(x_train, y_train, x_valid, y_valid, model, n_samples=300, random_state=None):
   """Generate random subsets and calculate their true utility"""
   if random_state is None:
       random_state = np.random.RandomState()
   elif isinstance(random_state, int):
       random_state = np.random.RandomState(random_state)
   
   # Generate random sampling ratios
   ratios = random_state.uniform(0.01, 0.99, n_samples)
   
   subsets = []  # Store all subsets
   accuracies = []  # Store corresponding accuracies
   subset_sizes = []  # Store subset sizes
   
   print("Generating random subsets and calculating true utilities...")
   for ratio in tqdm(ratios):
       size = max(1, int(ratio * len(x_train)))
       indices = random_state.choice(len(x_train), size=size, replace=False).tolist()
       subset_sizes.append(size)
       
       # Calculate true utility
       true_util = compute_true_utility(model, indices, x_train, y_train, x_valid, y_valid)
       
       subsets.append(indices)
       accuracies.append(true_util)
   
   return subsets, accuracies, subset_sizes

# Function to generate testing subsets using a different strategy
def generate_testing_subsets(x_train, y_train, x_valid, y_valid, model, n_samples=300, random_state=None):
   """Generate testing subsets based on random sizes from 1 to num training data points"""
   if random_state is None:
       random_state = np.random.RandomState()
   elif isinstance(random_state, int):
       random_state = np.random.RandomState(random_state)
   
   num_training_points = len(x_train)
   
   subsets = []  # Store all subsets
   accuracies = []  # Store corresponding accuracies
   subset_sizes = []  # Store subset sizes
   
   print("Generating testing subsets and calculating true utilities...")
   for _ in tqdm(range(n_samples)):
       # Randomly sample a size between 1 and number of training data points
       size = random_state.randint(1, num_training_points + 1)
       indices = random_state.choice(num_training_points, size=size, replace=False).tolist()
       subset_sizes.append(size)
       
       # Calculate true utility
       true_util = compute_true_utility(model, indices, x_train, y_train, x_valid, y_valid)
       
       subsets.append(indices)
       accuracies.append(true_util)
   
   return subsets, accuracies, subset_sizes

def compute_true_utility(model, subset_indices, x_train, y_train, x_valid, y_valid):
   """Calculate true utility function value (through model training and evaluation)"""
   if len(subset_indices) == 0:
       return 0.0
   
   try:
       # Create subset
       if isinstance(x_train, torch.Tensor):
           x_subset = x_train[subset_indices]
           y_subset = y_train[subset_indices]
       else:
           x_subset = np.array([x_train[i] for i in subset_indices])
           y_subset = np.array([y_train[i] for i in subset_indices])
       
       # Process data for sklearn
       x_subset_processed, y_subset_processed = process_data_for_sklearn(x_subset, y_subset)
       x_valid_processed, y_valid_processed = process_data_for_sklearn(x_valid, y_valid)
       
       # Check number of classes
       unique_classes = np.unique(y_subset_processed)
       num_classes = len(unique_classes)
       
       if num_classes == 1:
           # If only one class, predict all validation samples as that class
           majority_class = unique_classes[0]
           predictions = np.full_like(y_valid_processed, majority_class)
           accuracy = np.mean(predictions == y_valid_processed)
           return accuracy
       
       # Normal case: use scikit-learn's LogisticRegression
       scikit_model = LogisticRegression(max_iter=10000)
       scikit_model.fit(x_subset_processed, y_subset_processed)
       
       # Predict and evaluate
       y_pred = scikit_model.predict(x_valid_processed)
       accuracy = np.mean(y_pred == y_valid_processed)
       
       return accuracy
   except Exception as e:
       print(f"Error in compute_true_utility: {e}")
       
       # Try using majority class prediction
       try:
           if isinstance(y_train, torch.Tensor):
               y_subset = y_train[subset_indices].cpu().numpy()
           else:
               y_subset = np.array([y_train[i] for i in subset_indices])
               
           if isinstance(y_valid, torch.Tensor):
               y_valid_np = y_valid.cpu().numpy()
           else:
               y_valid_np = np.array(y_valid)
           
           # Process labels
           if len(y_subset.shape) > 1 and y_subset.shape[1] > 1:
               y_subset = np.argmax(y_subset, axis=1)
           if len(y_valid_np.shape) > 1 and y_valid_np.shape[1] > 1:
               y_valid_np = np.argmax(y_valid_np, axis=1)
           
           # Use majority class prediction
           majority_class = np.argmax(np.bincount(y_subset.astype(int)))
           predictions = np.full_like(y_valid_np, majority_class)
           accuracy = np.mean(predictions == y_valid_np)
           return accuracy
       except Exception as e2:
           print(f"Fallback error in compute_true_utility: {e2}")
           return 0.0  # Return default value

class SimpleLinear(BaseDataEvaluator):
    """Simple linear approximator without weighting or constraints"""
    
    def __init__(self, train_subsets=None, train_accuracies=None, n_samples=50, random_state=None):
        super().__init__(n_samples, random_state)
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        
    def train_data_values(self, *args, **kwargs):
        """Calculate data values using simple linear regression"""
        if self.train_subsets is None or self.train_accuracies is None:
            raise ValueError("Training subsets and accuracies must be provided")
        
        n_points = len(self.x_train)
        
        # Create design matrix
        X_design = np.zeros((len(self.train_subsets), n_points))
        for i, subset in enumerate(self.train_subsets):
            X_design[i, subset] = 1
            
        # Fit linear model without intercept (assumes U(empty) = 0)
        linear_model = LinearRegression(fit_intercept=False)
        linear_model.fit(X_design, self.train_accuracies)
        
        # Store data values
        self.data_values = linear_model.coef_
        
        # Debug output
        print(f"Linear - Coefficient range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
        print(f"Linear - Coefficient mean: {self.data_values.mean():.6f}")
        print(f"Linear - Coefficient sum: {self.data_values.sum():.6f}")
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values

class MLPRegressorEvaluator(BaseDataEvaluator):
    """MLP Regressor approximator without weighting or constraints"""
    
    def __init__(self, train_subsets=None, train_accuracies=None, n_samples=50, hidden_layer_sizes=(100,), 
                 activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', max_iter=200, random_state=None):
        super().__init__(n_samples, random_state)
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def train_data_values(self, *args, **kwargs):
        """Calculate data values using MLP Regressor"""
        if self.train_subsets is None or self.train_accuracies is None:
            raise ValueError("Training subsets and accuracies must be provided")
        
        n_points = len(self.x_train)
        
        # Create design matrix
        X_design = np.zeros((len(self.train_subsets), n_points))
        for i, subset in enumerate(self.train_subsets):
            X_design[i, subset] = 1
            
        # Calculate value of full set for constraint
        full_util = None
        for subset, util in zip(self.train_subsets, self.train_accuracies):
            if len(subset) == n_points:
                full_util = util
                break
        
        if full_util is None:
            # Compute full set utility if not available
            full_util = compute_true_utility(
                None, list(range(n_points)), 
                self.x_train, self.y_train, 
                self.x_valid, self.y_valid
            )
            print(f"Computed full set utility: {full_util:.4f}")

        # Parameter search for MLP
        # Define hyperparameter combinations
        learning_rates = ['constant', 'adaptive']
        max_iters = [100, 200, 500]
        alphas = [0, 0.0001, 0.001]  
        
        best_mlp = None
        best_score = float('inf')
        best_params = {}
        
        # Split training data for validation (50% validation)
        n_train = len(self.train_subsets)
        train_indices = np.random.choice(n_train, n_train // 2, replace=False)
        val_indices = np.array([i for i in range(n_train) if i not in train_indices])
        
        X_train = X_design[train_indices]
        y_train = np.array(self.train_accuracies)[train_indices]
        X_val = X_design[val_indices]
        y_val = np.array(self.train_accuracies)[val_indices]
        
        print("Starting MLP parameter search...")
        
        for lr in learning_rates:
            for mi in max_iters:
                for a in alphas:  # 添加alpha搜索循环
                    print(f"Testing learning_rate={lr}, max_iter={mi}, alpha={a}")
                    
                    mlp_model = MLPRegressor(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        activation=self.activation,
                        solver=self.solver,
                        alpha=a,  # 使用搜索的alpha值
                        learning_rate=lr,
                        max_iter=mi,
                        random_state=self.random_state
                    )
                    
                    # Train on subset
                    mlp_model.fit(X_train, y_train)
                    
                    # Evaluate on validation set
                    y_pred = mlp_model.predict(X_val)
                    val_score = mean_squared_error(y_val, y_pred)
                    
                    print(f"  Validation MSE: {val_score:.6f}")
                    
                    # Keep track of best model
                    if val_score < best_score:
                        best_score = val_score
                        best_mlp = mlp_model
                        best_params = {'learning_rate': lr, 'max_iter': mi, 'alpha': a}
        
        print(f"Best parameters: {best_params}, Validation MSE: {best_score:.6f}")
        
        # Retrain best model on all data
        if best_mlp is not None:
            final_mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=best_params['alpha'],  # 使用最佳alpha
                learning_rate=best_params['learning_rate'],
                max_iter=best_params['max_iter'],
                random_state=self.random_state
            )
            
            final_mlp.fit(X_design, self.train_accuracies)
            self.mlp_model = final_mlp
            
            # For interpretation, we need to determine feature importances
            # We'll use a simple method: train a linear model on the same data and use its coefficients
            linear_model = LinearRegression(fit_intercept=False)
            linear_model.fit(X_design, self.train_accuracies)
            
            # Store data values from linear model
            self.data_values = linear_model.coef_
            
            # Optionally normalize to ensure data values sum to full_util
            scaling_factor = full_util / np.sum(self.data_values)
            self.data_values = self.data_values * scaling_factor
            
            # Debug output
            print(f"MLP - Best parameters: {best_params}")
            print(f"MLP - Coefficient range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
            print(f"MLP - Coefficient mean: {self.data_values.mean():.6f}")
            print(f"MLP - Coefficient sum: {self.data_values.sum():.6f}")
        else:
            # Fallback to original approach
            mlp_model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            
            mlp_model.fit(X_design, self.train_accuracies)
            self.mlp_model = mlp_model
            
            # Use linear model for interpretability
            linear_model = LinearRegression(fit_intercept=False)
            linear_model.fit(X_design, self.train_accuracies)
            
            self.data_values = linear_model.coef_
            scaling_factor = full_util / np.sum(self.data_values)
            self.data_values = self.data_values * scaling_factor
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values

class BipartiteEvaluator(BaseDataEvaluator):
    """Class-aware greedy bipartite matching data valuation method"""
    
    def __init__(self, train_subsets=None, train_accuracies=None, n_samples=50, random_state=None, threshold_range=None):
        super().__init__(n_samples, random_state)
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        
        if threshold_range is None:
            # Use quantile range consistent with original implementation
            self.threshold_range = np.linspace(0.5, 0.95, 1000)  # Focus on high similarity region
        else:
            self.threshold_range = threshold_range
            
        self.optimal_threshold = None
        self.valid_edges = None
        self.coverage_sequence = None
        
    def _get_class_labels(self, y):
        """Get class labels"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        # Handle one-hot encoding    
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel()
        
    def _compute_similarity_matrix(self, x_train, x_valid):
        """Compute similarity matrix between training and validation sets"""
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.cpu().numpy()
        if isinstance(x_valid, torch.Tensor):    
            x_valid = x_valid.cpu().numpy()
            
        # Calculate Euclidean distance and standardize
        distances = cdist(x_train, x_valid, 'euclidean')
        distances = distances / distances.std()
        # Convert to similarity scores
        similarities = np.exp(-distances**2/2)
        
        return similarities
    
    def _precompute_similarity_distribution(self, x_train, x_valid, train_labels, valid_labels):
        """Precompute intra-class similarity distribution"""
        similarities = self._compute_similarity_matrix(x_train, x_valid)
        
        # Keep only same-class similarities
        class_mask = train_labels.reshape(-1, 1) == valid_labels.reshape(1, -1)
        intra_class_similarities = similarities[class_mask]
        
        # Save as instance variables
        self.base_similarity_matrix = similarities
        self.intra_class_similarities = intra_class_similarities
        
        return similarities
        
    def _compute_valid_edges(self, quantile, train_labels, valid_labels):
        """Compute valid edges using quantile threshold"""
        if not hasattr(self, 'base_similarity_matrix'):
            raise ValueError("Must call _precompute_similarity_distribution first")
            
        # Use quantile as threshold
        threshold = np.quantile(self.intra_class_similarities, quantile)
        
        n_train = len(train_labels)
        n_valid = len(valid_labels)
        valid_edges = np.zeros((n_train, n_valid), dtype=bool)
        
        # Calculate valid edges
        for i in range(n_train):
            same_class = train_labels[i] == valid_labels
            high_similarity = self.base_similarity_matrix[i] >= threshold
            valid_edges[i] = same_class & high_similarity
            
        return valid_edges
        
    def _compute_greedy_coverage(self, valid_edges):
        """Compute maximum coverage sequence using greedy strategy"""
        n_train, n_valid = valid_edges.shape
        coverage_sequence = []
        remaining_edges = valid_edges.copy()
        remaining_valid = np.ones(n_valid, dtype=bool)
        
        while True:
            # Calculate how many uncovered validation points each training point can cover
            coverage_counts = (remaining_edges & remaining_valid).sum(axis=1)
            
            # If no new coverage, add remaining points
            if coverage_counts.max() == 0:
                remaining_points = set(range(n_train)) - set(coverage_sequence)
                remaining_list = list(remaining_points)
                self.random_state.shuffle(remaining_list)
                coverage_sequence.extend(remaining_list)
                break
                
            # Choose point with maximum coverage
            best_point = np.argmax(coverage_counts)
            coverage_sequence.append(best_point)
            
            # Update remaining uncovered validation points
            newly_covered = remaining_edges[best_point] & remaining_valid
            remaining_valid[newly_covered] = False
            remaining_edges[:, newly_covered] = False
            
        return coverage_sequence
        
    def _generate_subsets_and_accuracies(self, x_train, y_train, x_valid, y_valid):
        """Pre-generate all subsets and their corresponding validation accuracies"""
        print("Pre-generating subsets and calculating accuracies...")
        
        # Use already generated subsets from input
        subsets = self.train_subsets
        accuracies = self.train_accuracies
        
        return subsets, accuracies
        
    def _find_optimal_threshold(self, x_train, y_train, x_valid, y_valid):
        """Find optimal threshold through grid search in quantile space"""
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(y_valid)
        
        # Precompute similarity distribution
        self._precompute_similarity_distribution(x_train, x_valid, train_labels, valid_labels)
        
        # Use pre-generated subsets and accuracies
        subsets, accuracies = self._generate_subsets_and_accuracies(
            x_train, y_train, x_valid, y_valid
        )
        
        print("Starting grid search for optimal quantile threshold...")
        best_threshold = None
        min_error = float('inf')
        
        # Search in quantile space
        for quantile in tqdm(self.threshold_range):
            mse = 0
            # Calculate valid edges using quantile
            valid_edges = self._compute_valid_edges(
                quantile, train_labels, valid_labels
            )
            
            # For each pre-generated subset
            for subset_idx, (indices, true_acc) in enumerate(zip(subsets, accuracies)):
                # Calculate subset coverage score
                valid_edges_subset = valid_edges[indices]
                coverage = valid_edges_subset.any(axis=0).mean()
                mse += (coverage - true_acc) ** 2
                
            avg_mse = mse / len(subsets)
            if avg_mse < min_error:
                min_error = avg_mse
                best_threshold = quantile
                
        return best_threshold, min_error
        
    def train_data_values(self, *args, **kwargs):
        """Train data valuation model"""
        print(f"Starting optimal quantile threshold search...")
        self.optimal_threshold, self.validation_error = self._find_optimal_threshold(
            self.x_train, self.y_train,
            self.x_valid, self.y_valid
        )
        print(f"Found optimal quantile threshold: {self.optimal_threshold:.3f}, "
              f"validation MSE: {self.validation_error:.3f}")
        
        # Get class labels
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        # Calculate valid edges using optimal quantile
        self.valid_edges = self._compute_valid_edges(
            self.optimal_threshold,
            train_labels,
            valid_labels
        )
        
        # Find coverage sequence using greedy strategy
        self.coverage_sequence = self._compute_greedy_coverage(self.valid_edges)
        
        # Calculate data values based on sequence position
        n_train = len(self.x_train)
        self.data_values = np.zeros(n_train)
        for i, idx in enumerate(self.coverage_sequence):
            self.data_values[idx] = n_train - i
            
        # Normalize data values
        self.data_values = (self.data_values - self.data_values.min())
        if self.data_values.max() > self.data_values.min():
            self.data_values /= (self.data_values.max() - self.data_values.min())
        
        print(f"Bipartite - Data values range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
        print(f"Bipartite - Data values mean: {self.data_values.mean():.6f}")
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values

class ShapleyLinear(BaseDataEvaluator):
    """Linear approximator with Shapley weighting using CVXPY"""
    
    def __init__(self, train_subsets=None, train_accuracies=None, n_samples=50, random_state=None):
        super().__init__(n_samples, random_state)
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        
    def _compute_shapley_weights(self, n, sizes):
        """Compute Shapley weights for each subset size"""
        weights = np.zeros(len(sizes))
        for i, size in enumerate(sizes):
            if size <= 0 or size >= n:
                weights[i] = 0  # Exclude empty set and full set
            else:
                # Shapley weight calculation
                weights[i] = 1.0 / (comb(n-1, size-1) * n)
                
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(sizes)) / len(sizes)
            
        return weights
        
    def train_data_values(self, *args, **kwargs):
        """Calculate Shapley data values using CVXPY for constrained optimization"""
        if self.train_subsets is None or self.train_accuracies is None:
            raise ValueError("Training subsets and accuracies must be provided")
        
        n_points = len(self.x_train)
        
        # Create design matrix
        X_design = np.zeros((len(self.train_subsets), n_points))
        for i, subset in enumerate(self.train_subsets):
            X_design[i, subset] = 1
            
        # Calculate utility of full set for constraint
        # Assuming the last subset is close to full set, or we compute it
        full_util = None
        for subset, util in zip(self.train_subsets, self.train_accuracies):
            if len(subset) == n_points:
                full_util = util
                break
        
        if full_util is None:
            # Compute full set utility if not available
            full_util = compute_true_utility(
                None, list(range(n_points)), 
                self.x_train, self.y_train, 
                self.x_valid, self.y_valid
            )
            print(f"Computed full set utility: {full_util:.4f}")
        
        # Calculate Shapley weights
        sizes = X_design.sum(axis=1).astype(int)
        weights = self._compute_shapley_weights(n_points, sizes)
        
        # Set up CVXPY problem
        data_values = cp.Variable(n_points)
        
        # Weighted objective
        objective = cp.sum(cp.multiply(weights, cp.square(self.train_accuracies - X_design @ data_values)))
        
        # Constraints
        constraints = [
            cp.sum(data_values) == full_util,  # Efficiency constraint
            data_values >= -1,                 # Lower bound constraint
            data_values <= 1                   # Upper bound constraint
        ]
        
        # Solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: CVXPY optimization problem status: {problem.status}")
        
        # Store data values
        self.data_values = data_values.value
        
        # Debug output
        print(f"Shapley Linear - Weight range: {weights.min():.6f} to {weights.max():.6f}")
        print(f"Shapley Linear - Coefficient range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
        print(f"Shapley Linear - Coefficient mean: {self.data_values.mean():.6f}")
        print(f"Shapley Linear - Coefficient sum: {self.data_values.sum():.6f} (should be close to {full_util:.6f})")
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values


class BetaShapleyLinear(BaseDataEvaluator):
    """Linear approximator with Beta-Shapley weighting using CVXPY"""
    
    def __init__(self, alpha=2, beta=2, train_subsets=None, train_accuracies=None, n_samples=50, random_state=None):
        super().__init__(n_samples, random_state)
        self.alpha = alpha
        self.beta = beta
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        
    def _compute_beta_weights(self, n, sizes):
            """Compute Beta-Shapley weights for each subset"""
            weights = np.zeros(len(sizes))
            
            # Calculate Beta weights according to Beta-Shapley formula
            for i, size in enumerate(sizes):
                if size <= 0 or size >= n:
                    weights[i] = 0  # Exclude empty set and full set
                else:
                    # Beta-Shapley weight calculation using Beta function
                    weights[i] = beta(self.alpha + size - 1, self.beta + n - size) / beta(self.alpha, self.beta)
                    
            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 1e-10)
            weights = weights / weights.sum()
                
            return weights
        
    def train_data_values(self, *args, **kwargs):
        """Calculate Beta-Shapley data values using CVXPY for constrained optimization"""
        if self.train_subsets is None or self.train_accuracies is None:
            raise ValueError("Training subsets and accuracies must be provided")
        
        n_points = len(self.x_train)
        
        # Create design matrix
        X_design = np.zeros((len(self.train_subsets), n_points))
        for i, subset in enumerate(self.train_subsets):
            X_design[i, subset] = 1
            
        # Calculate utility of full set for constraint
        # Assuming the last subset is close to full set, or we compute it
        full_util = None
        for subset, util in zip(self.train_subsets, self.train_accuracies):
            if len(subset) == n_points:
                full_util = util
                break
        
        if full_util is None:
            # Compute full set utility if not available
            full_util = compute_true_utility(
                None, list(range(n_points)), 
                self.x_train, self.y_train, 
                self.x_valid, self.y_valid
            )
            print(f"Computed full set utility: {full_util:.4f}")
        
        # Calculate Beta-Shapley weights
        sizes = X_design.sum(axis=1).astype(int)
        weights = self._compute_beta_weights(n_points, sizes)
        
        # Set up CVXPY problem
        data_values = cp.Variable(n_points)
        
        # Weighted objective
        objective = cp.sum(cp.multiply(weights, cp.square(self.train_accuracies - X_design @ data_values)))
        
        # Constraints
        constraints = [
            cp.sum(data_values) == full_util,  # Efficiency constraint
            data_values >= -1,                 # Lower bound constraint
            data_values <= 1                   # Upper bound constraint
        ]
        
        # Solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: CVXPY optimization problem status: {problem.status}")
        
        # Store data values
        self.data_values = data_values.value
        
        # Debug output
        print(f"BetaShapley Linear - Weight range: {weights.min():.6f} to {weights.max():.6f}")
        print(f"BetaShapley Linear - Coefficient range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
        print(f"BetaShapley Linear - Coefficient mean: {self.data_values.mean():.6f}")
        print(f"BetaShapley Linear - Coefficient sum: {self.data_values.sum():.6f} (should be close to {full_util:.6f})")
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values

class BanzhafLinear(BaseDataEvaluator):
    """Linear approximator with Banzhaf weighting using CVXPY"""
    
    def __init__(self, train_subsets=None, train_accuracies=None, n_samples=50, random_state=None):
        super().__init__(n_samples, random_state)
        self.train_subsets = train_subsets
        self.train_accuracies = train_accuracies
        
    def _compute_banzhaf_weights(self, n):
        """Compute Banzhaf weights - uniform 2^(-n-1) weights"""
        # In Banzhaf index, all coalitions get equal weight 2^(-n-1)
        weight = 2.0 ** (-n-1)
        return weight
        
    def train_data_values(self, *args, **kwargs):
        """Calculate Banzhaf data values using CVXPY for constrained optimization"""
        if self.train_subsets is None or self.train_accuracies is None:
            raise ValueError("Training subsets and accuracies must be provided")
        
        n_points = len(self.x_train)
        
        # Create design matrix
        X_design = np.zeros((len(self.train_subsets), n_points))
        for i, subset in enumerate(self.train_subsets):
            X_design[i, subset] = 1
            
        # Calculate utility of full set for constraint
        # Assuming the last subset is close to full set, or we compute it
        full_util = None
        for subset, util in zip(self.train_subsets, self.train_accuracies):
            if len(subset) == n_points:
                full_util = util
                break
        
        if full_util is None:
            # Compute full set utility if not available
            full_util = compute_true_utility(
                None, list(range(n_points)), 
                self.x_train, self.y_train, 
                self.x_valid, self.y_valid
            )
            print(f"Computed full set utility: {full_util:.4f}")
        
        # Calculate Banzhaf weight - uniform across all subsets
        weight = self._compute_banzhaf_weights(n_points)
        weights = np.ones(len(self.train_subsets)) * weight
        
        # Set up CVXPY problem
        data_values = cp.Variable(n_points)
        
        # Weighted objective - all weights are equal, so we can simplify
        objective = cp.sum(cp.square(self.train_accuracies - X_design @ data_values))
        
        # Constraints
        constraints = [
            cp.sum(data_values) == full_util,  # Efficiency constraint
            data_values >= -1,                 # Lower bound constraint
            data_values <= 1                   # Upper bound constraint
        ]
        
        # Solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: CVXPY optimization problem status: {problem.status}")
        
        # Store data values
        self.data_values = data_values.value
        
        # Debug output
        print(f"Banzhaf Linear - Weight: {weight:.6f}")
        print(f"Banzhaf Linear - Coefficient range: {self.data_values.min():.6f} to {self.data_values.max():.6f}")
        print(f"Banzhaf Linear - Coefficient mean: {self.data_values.mean():.6f}")
        print(f"Banzhaf Linear - Coefficient sum: {self.data_values.sum():.6f} (should be close to {full_util:.6f})")
        
        return self
    
    def evaluate_data_values(self):
        """Implement abstract method, return calculated data values"""
        if not hasattr(self, 'data_values'):
            raise ValueError("Data values not yet calculated, please call train_data_values method first")
        return self.data_values


def compute_bipartite_utility(bipartite_evaluator, subset_indices):
    """Calculate bipartite approximation utility function value"""
    if len(subset_indices) == 0:
        return 0.0
    
    if bipartite_evaluator.valid_edges is None:
        raise ValueError("Bipartite evaluator not yet trained")
    
    # Get subset valid edges
    subset_valid_edges = bipartite_evaluator.valid_edges[subset_indices]
    
    # Calculate coverage - 对于每个验证节点，检查是否至少有一个训练节点与其连接
    covered_valid_nodes = subset_valid_edges.any(axis=0)
    coverage_rate = covered_valid_nodes.mean()
    
    return coverage_rate
    
def compute_linear_utility(evaluator, subset_indices):
   """Calculate linear approximation utility function value (applicable to Shapley, BetaShapley, Banzhaf)"""
   if len(subset_indices) == 0:
       return 0.0
   
   # Use linear combination approach (sum)
   values = evaluator.data_values
   subset_utility = np.sum(values[subset_indices])
   return subset_utility

def compute_mlp_utility(evaluator, subset_indices):
    """Calculate MLP approximation utility function value"""
    if len(subset_indices) == 0:
        return 0.0
    
    # For our case, we use similar approach as linear utility
    values = evaluator.data_values
    subset_utility = np.sum(values[subset_indices])
    return subset_utility

def compute_pairwise_ordering_preservation(true_utilities, approx_utilities):
   """Calculate ordering preservation rate - measure how many pairwise relationships are preserved between utility functions"""
   n = len(true_utilities)
   preserved_count = 0
   comparison_count = 0
   
   for i in range(n):
       for j in range(i+1, n):
           comparison_count += 1
           true_order = np.sign(true_utilities[i] - true_utilities[j])
           approx_order = np.sign(approx_utilities[i] - approx_utilities[j])
           
           if true_order == approx_order:
               preserved_count += 1
   
   return preserved_count / comparison_count if comparison_count > 0 else 0.0

def run_experiment(dataset_name, seed, num_samples=300, train_size=100, valid_size=500):
   """Run data valuation experiment for a single dataset and seed"""
   # Set random seed
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   
   print(f"Running experiment for dataset: {dataset_name}, seed: {seed}")
   
   # Create output directory
   output_dir = f"results/{dataset_name}/seed_{seed}"
   os.makedirs(output_dir, exist_ok=True)
   
   # Initialize experiment environment
   exper_med = ExperimentMediator.model_factory_setup(
       dataset_name=dataset_name,
       model_name="LogisticRegression",
       train_count=train_size,
       valid_count=valid_size,
       test_count=500,
       add_noise=None,
       noise_kwargs={},
       metric_name="accuracy",
       device="cpu"
   )
   
   # Get data
   fetcher = exper_med.fetcher
   x_train, y_train, x_valid, y_valid = fetcher.x_train, fetcher.y_train, fetcher.x_valid, fetcher.y_valid
   model = exper_med.pred_model
   
   # Generate training subsets and calculate true utilities
   train_subsets, train_accuracies, train_sizes = generate_random_subsets(
       x_train, y_train, x_valid, y_valid, model, n_samples=num_samples, random_state=seed
   )
   
   # Generate testing subsets separately with different logic
   test_subsets, test_accuracies, test_sizes = generate_testing_subsets(
       x_train, y_train, x_valid, y_valid, model, n_samples=num_samples, random_state=seed+1  # Different seed
   )
   
   print(f"Training subsets: {len(train_subsets)}, Test subsets: {len(test_subsets)}")
   
   # Initialize different evaluators
   print("Training evaluators...")
   evaluators = {
       'Bipartite': BipartiteEvaluator(train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, random_state=seed),
       'Linear': SimpleLinear(train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, random_state=seed),
       'MLP': MLPRegressorEvaluator(train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, 
                                    hidden_layer_sizes=(100, 50), activation='relu', random_state=seed),
       'Shapley': ShapleyLinear(train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, random_state=seed),
       'BetaShapley': BetaShapleyLinear(alpha=2, beta=2, train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, random_state=seed),
       'Banzhaf': BanzhafLinear(train_subsets=train_subsets, train_accuracies=train_accuracies, n_samples=50, random_state=seed)
   }
   
   # Train all evaluators
   training_times = {}
   for name, evaluator in evaluators.items():
       start_time = time.time()
       evaluator.x_train = x_train
       evaluator.y_train = y_train
       evaluator.x_valid = x_valid
       evaluator.y_valid = y_valid
       evaluator.train_data_values()
       training_times[name] = time.time() - start_time
       print(f"Training {name} completed in {training_times[name]:.2f} seconds")
   
   # Initialize result storage for this seed
   results = {
       'dataset': dataset_name,
       'seed': seed,
       'train_true_utilities': train_accuracies,
       'test_true_utilities': test_accuracies,
       'train_approximations': {name: [] for name in evaluators.keys()},
       'test_approximations': {name: [] for name in evaluators.keys()},
       'train_correlations': {name: {} for name in evaluators.keys()},
       'test_correlations': {name: {} for name in evaluators.keys()},
       'timing': training_times
   }
   
   # Calculate query efficiency
   train_query_times = {name: 0 for name in evaluators.keys()}
   test_query_times = {name: 0 for name in evaluators.keys()}
   
   # Evaluate each method on training set
   print("Evaluating on training set...")
   for name, evaluator in evaluators.items():
       for subset in tqdm(train_subsets):
           start_time = time.time()
           
           if name == 'Bipartite':
               approx_util = compute_bipartite_utility(evaluator, subset)
           elif name == 'MLP':
               approx_util = compute_mlp_utility(evaluator, subset)
           else:
               approx_util = compute_linear_utility(evaluator, subset)
           
           query_time = time.time() - start_time
           train_query_times[name] += query_time
           
           results['train_approximations'][name].append(approx_util)
   
   # Evaluate each method on test set
   print("Evaluating on test set...")
   for name, evaluator in evaluators.items():
       for subset in tqdm(test_subsets):
           start_time = time.time()
           
           if name == 'Bipartite':
               approx_util = compute_bipartite_utility(evaluator, subset)
           elif name == 'MLP':
               approx_util = compute_mlp_utility(evaluator, subset)
           else:
               approx_util = compute_linear_utility(evaluator, subset)
           
           query_time = time.time() - start_time
           test_query_times[name] += query_time
           
           results['test_approximations'][name].append(approx_util)
   
   # Record average query times
   for name in evaluators.keys():
       results['timing'][f"{name}_train_query_avg"] = train_query_times[name] / len(train_subsets)
       results['timing'][f"{name}_test_query_avg"] = test_query_times[name] / len(test_subsets)
   
   # Calculate correlation metrics - Training set
   print("Computing correlation metrics for training set...")
   train_true_utilities = np.array(results['train_true_utilities'])
   
   for name in evaluators.keys():
       train_approx_utilities = np.array(results['train_approximations'][name])
       
       # Calculate MAE
       mae = mean_absolute_error(train_true_utilities, train_approx_utilities)
       results['train_correlations'][name]['mae'] = mae
       
       # Calculate MSE
       mse = mean_squared_error(train_true_utilities, train_approx_utilities)
       results['train_correlations'][name]['mse'] = mse
       
       print(f"{name} (Train) - MAE: {mae:.4f}, MSE: {mse:.4f}")
   
   # Calculate correlation metrics - Test set
   print("Computing correlation metrics for test set...")
   test_true_utilities = np.array(results['test_true_utilities'])
   
   for name in evaluators.keys():
       test_approx_utilities = np.array(results['test_approximations'][name])
       
       # Calculate MAE
       mae = mean_absolute_error(test_true_utilities, test_approx_utilities)
       results['test_correlations'][name]['mae'] = mae
       
       # Calculate MSE
       mse = mean_squared_error(test_true_utilities, test_approx_utilities)
       results['test_correlations'][name]['mse'] = mse
       
       print(f"{name} (Test) - MAE: {mae:.4f}, MSE: {mse:.4f}")
   
   # Save results as CSV
   results_df = pd.DataFrame({
       'dataset': dataset_name,
       'seed': seed,
       'split': ['train'] * 12 + ['test'] * 12,  # 6 methods × 2 metrics × 2 splits = 24 rows
       'method': list(evaluators.keys()) * 4,  # 6 methods repeated for each metric and split
       'metric': ['mae'] * 6 + ['mse'] * 6 + ['mae'] * 6 + ['mse'] * 6,
       'value': (
           # Train metrics
           [results['train_correlations'][name]['mae'] for name in evaluators.keys()] +
           [results['train_correlations'][name]['mse'] for name in evaluators.keys()] +
           # Test metrics
           [results['test_correlations'][name]['mae'] for name in evaluators.keys()] +
           [results['test_correlations'][name]['mse'] for name in evaluators.keys()]
       )
   })
   
   # Save the results
   results_df.to_csv(os.path.join(output_dir, f"correlation_results.csv"), index=False)
   
   print(f"Results saved to {output_dir}")
   return results
    
def parse_args():
   parser = argparse.ArgumentParser(description='Run utility correlation experiment')
   parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
   parser.add_argument('--seed', type=int, required=True, help='Random seed')
   parser.add_argument('--num_samples', type=int, default=500, help='Number of random subsets to generate')
   parser.add_argument('--train_size', type=int, default=100, help='Number of training points')
   parser.add_argument('--valid_size', type=int, default=500, help='Number of validation points')
   return parser.parse_args()

if __name__ == "__main__":
   args = parse_args()
   # Run experiment with the specified dataset and seed
   run_experiment(
       dataset_name=args.dataset,
       seed=args.seed,
       num_samples=args.num_samples,
       train_size=args.train_size,
       valid_size=args.valid_size
   )