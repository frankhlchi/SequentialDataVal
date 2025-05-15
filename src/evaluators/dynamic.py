from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch
from torch.utils.data import Subset
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path

class DynamicProgrammingEvaluator(DataEvaluator, ModelMixin):
   """Dynamic Programming based data evaluator
   
   This evaluator calculates data point values using dynamic programming. It uses caching
   mechanisms to avoid redundant computations, improving efficiency. Key steps include:
   1. Building and caching state spaces and transitions
   2. Computing and caching state utility values
   3. Calculating optimal value functions through dynamic programming
   4. Extracting data point values based on optimal policy
   """
   
   def __init__(self, 
                max_subset_size: int = 50,
                random_state=None):
       """Initialize the evaluator
       
       Args:
           max_subset_size: Maximum subset size, used to control state space
           random_state: Random number generator 
       """
       super().__init__(random_state=random_state)
       self.max_subset_size = max_subset_size
       
       # State-related caches
       self.value_cache = {}      # Cache for state utility values
       self.transition_cache = {} # Cache for state transitions
       self.dp_cache = {}        # Cache for dynamic programming intermediate results

   def _encode_state(self, state: Set[int]) -> str:
       """Encode state set as string
       
       Converts set to sorted string representation for use as cache key.
       """
       return ','.join(map(str, sorted(state)))
       
   def _decode_state(self, state_str: str) -> Set[int]:
       """Decode state string into set"""
       if not state_str:
           return set()
       return set(map(int, state_str.split(',')))
       
   def _evaluate_utility(self, state: Set[int], model_kwargs: dict) -> float:
       """Calculate utility value for given state
       
       Uses caching to avoid repeated computation of the same state.
       If state exists in cache, returns cached result directly.
       """
       # Generate cache key
       state_key = self._encode_state(state)
       
       # Check cache
       if state_key in self.value_cache:
           return self.value_cache[state_key]
           
       # Calculate utility value
       if not state:
           utility = 0.0
       else:
           curr_model = self.pred_model.clone()
           indices = list(state)
           curr_model.fit(
               Subset(self.x_train, indices),
               Subset(self.y_train, indices),
               **model_kwargs
           )
           y_valid_pred = curr_model.predict(self.x_valid)
           utility = self.evaluate(self.y_valid, y_valid_pred)
       
       # Save to cache
       self.value_cache[state_key] = utility
       return utility
       
   def _get_next_states(self, curr_state: Set[int]) -> List[Set[int]]:
       """Generate all possible successor states from current state
       
       Uses caching to avoid regenerating successor states for the same state.
       """
       state_key = self._encode_state(curr_state)
       
       # Check cache
       if state_key in self.transition_cache:
           return self.transition_cache[state_key]
           
       # Generate successor states
       if len(curr_state) >= self.max_subset_size:
           next_states = []
       else:
           next_states = []
           all_indices = set(range(len(self.x_train)))
           remaining = all_indices - curr_state
           
           for idx in remaining:
               next_state = curr_state | {idx}
               next_states.append(next_state)
               
       # Save to cache
       self.transition_cache[state_key] = next_states
       return next_states
       
   def _compute_optimal_value_function(self, model_kwargs: dict) -> Tuple[Dict[str, float], Dict[str, int]]:
       """Compute optimal value function using dynamic programming
       
       Performs dynamic programming by enumerating all possible states. Uses caching to avoid
       repeated computations.
       """
       V = {}  # State value function 
       pi = {}  # Optimal policy
       
       # Dynamic programming from larger to smaller subset sizes
       for size in range(self.max_subset_size, -1, -1):
           print(f"Computing values for states of size {size}")
           
           # Generate all states of current size
           curr_states = []
           if size == 0:
               curr_states = [set()]
           else:
               from itertools import combinations
               all_indices = range(len(self.x_train))
               for combo in combinations(all_indices, size):
                   curr_states.append(set(combo))
           
           # Process each state
           for state in curr_states:
               state_str = self._encode_state(state)
               curr_utility = self._evaluate_utility(state, model_kwargs)
               
               next_states = self._get_next_states(state)
               if not next_states:  # Terminal state
                   V[state_str] = curr_utility
                   continue
                   
               # Calculate optimal successor state
               max_future_value = float('-inf')
               best_action = None
               
               for next_state in next_states:
                   next_state_str = self._encode_state(next_state)
                   if next_state_str in V:  # Use pre-computed value
                       future_value = V[next_state_str]
                       if future_value > max_future_value:
                           max_future_value = future_value
                           best_action = next_state - state
                           
               if max_future_value == float('-inf'):
                   max_future_value = 0
                   
               V[state_str] = curr_utility + max_future_value
               if best_action:
                   pi[state_str] = best_action.pop()
            
           # Cache results for current size
           self.dp_cache[f'v_{size}'] = V.copy()
           self.dp_cache[f'pi_{size}'] = pi.copy()
       
       return V, pi
       
   def train_data_values(self, *args, **kwargs):
       """Train data valuation model
       
       Computes optimal value function and policy, then generates data point
       selection sequence based on this. Entire process uses caching for efficiency.
       """
       # Compute value function and policy
       self.V, self.pi = self._compute_optimal_value_function(kwargs)
       
       # Generate selection sequence using policy
       curr_state = set()
       selection_sequence = []
       
       while len(curr_state) < self.max_subset_size:
           state_str = self._encode_state(curr_state)
           if state_str not in self.pi:
               break
           next_point = self.pi[state_str]
           selection_sequence.append(next_point)
           curr_state.add(next_point)
           
       # Add remaining unselected points
       remaining = set(range(len(self.x_train))) - set(selection_sequence)
       selection_sequence.extend(list(remaining))
       
       self.selection_sequence = selection_sequence
       return self
       
   def evaluate_data_values(self) -> np.ndarray:
       """Return normalized data values
       
       Calculates values based on position in selection sequence.
       Earlier positions indicate higher value.
       """
       n_train = len(self.selection_sequence)
       data_values = np.zeros(n_train)
       
       # Assign values based on position
       for i, idx in enumerate(self.selection_sequence):
           data_values[idx] = n_train - i
           
       # Normalize to [0,1] range
       return data_values / n_train