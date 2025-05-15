from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch
from torch.utils.data import Subset
import torch


class PredictionBasedMatchingEvaluator(DataEvaluator, ModelMixin):
    """基于预测准确性和类别感知的数据点评估器
    
    该评估器通过构建二分图来评估数据点的价值：
    1. 只有同类别的训练点和验证点之间才可能建立连接
    2. 只有当添加训练点能够改善验证点的预测时才建立连接
    3. 使用贪心策略选择覆盖序列来确定数据点价值
    """
    
    def __init__(self, num_samples=100, num_trials=10, random_state=None):
        """初始化评估器
        
        Args:
            num_samples: 每次试验中使用的样本数
            num_trials: 重复试验的次数
            random_state: 随机数种子
        """
        super().__init__(random_state=random_state)
        self.num_samples = num_samples
        self.num_trials = num_trials
        
    def _get_class_labels(self, y):
        """获取数据点的类别标签
        
        处理张量/数组格式和one-hot编码的情况
        
        Args:
            y: 输入标签(torch.Tensor或numpy.ndarray)
            
        Returns:
            numpy.ndarray: 一维类别标签数组
        """
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        if len(y.shape) > 1 and y.shape[1] > 1:  # one-hot编码
            return np.argmax(y, axis=1)
        return y.ravel()
        
    def _evaluate_single_point(self, curr_model, x, y):
        """评估单个点的预测是否正确
        
        Args:
            curr_model: 当前训练的模型
            x: 输入数据
            y: 真实标签
            
        Returns:
            numpy.ndarray: 布尔数组，表示每个点的预测是否正确
        """
        pred = curr_model.predict(x)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
            y = y.argmax(axis=1)
        return pred == y
    
    def _compute_bipartite_graph(self, model_kwargs):
        """计算训练点和验证点之间的二分图
        
        构建满足以下条件的邻接矩阵：
        1. 训练点和验证点必须属于相同类别
        2. 添加训练点后能够改善对应验证点的预测
        
        Args:
            model_kwargs: 模型训练的参数
            
        Returns:
            numpy.ndarray: 二分图的邻接矩阵
        """
        n_train = len(self.x_train)
        n_valid = len(self.x_valid)
        adjacency_matrix = np.zeros((n_train, n_valid))
        
        # 预处理训练集和验证集的类别标签
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        # 预计算类别匹配矩阵
        class_match = train_labels[:, np.newaxis] == valid_labels[np.newaxis, :]
        
        # 获取所有类别
        unique_classes = np.unique(train_labels)
        
        for trial in range(self.num_trials):
            if trial % 10 == 0:
                print(f"Trial {trial}")
                
            # 为每个类别随机选择一个基础点
            base_points = []
            for cls in unique_classes:
                cls_indices = np.where(train_labels == cls)[0]
                base_idx = self.random_state.choice(cls_indices, size=1)[0]
                base_points.append(base_idx)
            
            # 从剩余点中随机选择训练点序列
            remaining_points = list(set(range(n_train)) - set(base_points))
            train_sequence = self.random_state.choice(
                remaining_points,
                size=min(self.num_samples - len(base_points), len(remaining_points)),
                replace=False
            )
            
            curr_model = self.pred_model.clone()
            matched_valid_points = np.zeros(n_valid, dtype=bool)
            
            # 使用所有基础点训练初始模型
            curr_model.fit(
                Subset(self.x_train, base_points),
                Subset(self.y_train, base_points),
                **model_kwargs
            )
            
            # 记录基础模型的预测结果
            base_predictions = self._evaluate_single_point(
                curr_model,
                self.x_valid,
                self.y_valid
            ).astype(bool)
            
            curr_train_set = set(base_points)
            
            # 从剩余点开始添加并计算匹配
            for idx in train_sequence:
                # 将新点加入训练集
                curr_train_set.add(idx)
                curr_indices = list(curr_train_set)
                
                # 训练模型
                curr_model.fit(
                    Subset(self.x_train, curr_indices),
                    Subset(self.y_train, curr_indices),
                    **model_kwargs
                )
                
                # 评估验证点
                curr_correct = self._evaluate_single_point(
                    curr_model,
                    self.x_valid,
                    self.y_valid
                ).astype(bool)
                
                # 记录满足类别约束且预测改进的匹配
                new_matches = ((curr_correct & ~base_predictions) & 
                             ~matched_valid_points & 
                             class_match[idx])
                
                if np.any(new_matches):
                    adjacency_matrix[idx, new_matches] = 1
                    matched_valid_points |= new_matches
                    
        return (adjacency_matrix > 0).astype(float)
        
    def _compute_coverage_sequence(self, adjacency_matrix):
        """计算最大化覆盖的序列
        
        使用贪心策略选择每次能覆盖最多未覆盖验证点的训练点
        
        Args:
            adjacency_matrix: 二分图的邻接矩阵
            
        Returns:
            list: 训练点的最优覆盖序列
        """
        n_train = adjacency_matrix.shape[0]
        coverage_sequence = []
        remaining_matrix = adjacency_matrix.copy()
        
        while len(coverage_sequence) < n_train:
            # 计算每个训练点的当前覆盖度
            coverage_scores = (remaining_matrix > 0).sum(axis=1)
            
            # 选择覆盖度最大的点
            if coverage_scores.max() == 0:
                # 如果没有剩余连接,将剩余点随机排序
                remaining_indices = list(set(range(n_train)) - set(coverage_sequence))
                self.random_state.shuffle(remaining_indices)
                coverage_sequence.extend(remaining_indices)
                break
                
            best_idx = coverage_scores.argmax()
            coverage_sequence.append(best_idx)
            
            # 将已选点的连接置0
            remaining_matrix[best_idx, :] = 0
            # 将已覆盖的验证点对应的所有连接置0
            covered_valid = adjacency_matrix[best_idx, :] > 0
            remaining_matrix[:, covered_valid] = 0
            
        return coverage_sequence

    def train_data_values(self, *args, **kwargs):
        """训练数据估值模型
        
        Returns:
            self: 返回评估器实例
        """
        # 构建二分图
        self.adjacency_matrix = self._compute_bipartite_graph(kwargs)
        
        # 计算贪心覆盖序列
        self.coverage_sequence = self._compute_coverage_sequence(self.adjacency_matrix)
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值
        
        数据点的价值基于其在覆盖序列中的位置，位置越靠前价值越高
        
        Returns:
            numpy.ndarray: 归一化的数据点价值
        """
        n_train = len(self.coverage_sequence)
        data_values = np.zeros(n_train)
        
        # 基于序列位置赋值 - 位置越前价值越高
        for i, idx in enumerate(self.coverage_sequence):
            data_values[idx] = n_train - i
            
        return data_values / n_train  # 归一化到[0,1]区间