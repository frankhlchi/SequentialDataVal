from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch
from torch.utils.data import Subset
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path

class DynamicProgrammingEvaluator(DataEvaluator, ModelMixin):
    """基于动态规划的数据点评估器
    
    该评估器通过动态规划方法计算数据点的价值。使用缓存机制来避免重复计算，
    提高了计算效率。主要步骤包括：
    1. 构建并缓存状态空间和状态转移
    2. 计算并缓存状态效用值
    3. 通过动态规划计算最优值函数
    4. 基于最优策略提取数据点价值
    """
    
    def __init__(self, 
                 max_subset_size: int = 50,
                 random_state=None):
        """初始化评估器
        
        Args:
            max_subset_size: 最大子集规模,用于控制状态空间大小
            random_state: 随机数生成器 
        """
        super().__init__(random_state=random_state)
        self.max_subset_size = max_subset_size
        
        # 状态相关的缓存
        self.value_cache = {}      # 状态效用值的缓存
        self.transition_cache = {} # 状态转移的缓存
        self.dp_cache = {}        # 动态规划中间结果缓存

    def _encode_state(self, state: Set[int]) -> str:
        """将状态集合编码为字符串
        
        将集合转换为排序后的字符串表示，用作缓存的键值。
        """
        return ','.join(map(str, sorted(state)))
        
    def _decode_state(self, state_str: str) -> Set[int]:
        """将状态字符串解码为集合"""
        if not state_str:
            return set()
        return set(map(int, state_str.split(',')))
        
    def _evaluate_utility(self, state: Set[int], model_kwargs: dict) -> float:
        """计算给定状态的效用值
        
        使用缓存机制避免重复计算相同状态的效用值。
        如果状态已在缓存中，直接返回缓存的结果。
        """
        # 生成缓存键
        state_key = self._encode_state(state)
        
        # 检查缓存
        if state_key in self.value_cache:
            return self.value_cache[state_key]
            
        # 计算效用值
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
        
        # 保存到缓存
        self.value_cache[state_key] = utility
        return utility
        
    def _get_next_states(self, curr_state: Set[int]) -> List[Set[int]]:
        """生成当前状态的所有可能后继状态
        
        使用缓存机制避免重复生成相同状态的后继状态。
        """
        state_key = self._encode_state(curr_state)
        
        # 检查缓存
        if state_key in self.transition_cache:
            return self.transition_cache[state_key]
            
        # 生成后继状态
        if len(curr_state) >= self.max_subset_size:
            next_states = []
        else:
            next_states = []
            all_indices = set(range(len(self.x_train)))
            remaining = all_indices - curr_state
            
            for idx in remaining:
                next_state = curr_state | {idx}
                next_states.append(next_state)
                
        # 保存到缓存
        self.transition_cache[state_key] = next_states
        return next_states
        
    def _compute_optimal_value_function(self, model_kwargs: dict) -> Tuple[Dict[str, float], Dict[str, int]]:
        """使用动态规划计算最优值函数
        
        通过穷举所有可能的状态来执行动态规划。利用缓存机制避免重复计算。
        """
        V = {}  # 状态值函数 
        pi = {}  # 最优策略
        
        # 按子集大小从大到小进行动态规划
        for size in range(self.max_subset_size, -1, -1):
            print(f"Computing values for states of size {size}")
            
            # 生成当前大小的所有状态
            curr_states = []
            if size == 0:
                curr_states = [set()]
            else:
                from itertools import combinations
                all_indices = range(len(self.x_train))
                for combo in combinations(all_indices, size):
                    curr_states.append(set(combo))
            
            # 处理每个状态
            for state in curr_states:
                state_str = self._encode_state(state)
                curr_utility = self._evaluate_utility(state, model_kwargs)
                
                next_states = self._get_next_states(state)
                if not next_states:  # 终止状态
                    V[state_str] = curr_utility
                    continue
                    
                # 计算最优后继状态
                max_future_value = float('-inf')
                best_action = None
                
                for next_state in next_states:
                    next_state_str = self._encode_state(next_state)
                    if next_state_str in V:  # 使用已计算的值
                        future_value = V[next_state_str]
                        if future_value > max_future_value:
                            max_future_value = future_value
                            best_action = next_state - state
                            
                if max_future_value == float('-inf'):
                    max_future_value = 0
                    
                V[state_str] = curr_utility + max_future_value
                if best_action:
                    pi[state_str] = best_action.pop()
             
            # 缓存当前规模的结果
            self.dp_cache[f'v_{size}'] = V.copy()
            self.dp_cache[f'pi_{size}'] = pi.copy()
        
        return V, pi
        
    def train_data_values(self, *args, **kwargs):
        """训练数据估值模型
        
        计算最优值函数和策略，并基于此生成数据点选择序列。
        整个过程利用缓存机制提高效率。
        """
        # 计算值函数和策略
        self.V, self.pi = self._compute_optimal_value_function(kwargs)
        
        # 使用策略生成选择序列
        curr_state = set()
        selection_sequence = []
        
        while len(curr_state) < self.max_subset_size:
            state_str = self._encode_state(curr_state)
            if state_str not in self.pi:
                break
            next_point = self.pi[state_str]
            selection_sequence.append(next_point)
            curr_state.add(next_point)
            
        # 添加未选中的点
        remaining = set(range(len(self.x_train))) - set(selection_sequence)
        selection_sequence.extend(list(remaining))
        
        self.selection_sequence = selection_sequence
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值
        
        基于数据点在选择序列中的位置计算其价值。
        位置越靠前表示价值越高。
        """
        n_train = len(self.selection_sequence)
        data_values = np.zeros(n_train)
        
        # 基于位置赋值
        for i, idx in enumerate(self.selection_sequence):
            data_values[idx] = n_train - i
            
        # 归一化到[0,1]区间
        return data_values / n_train