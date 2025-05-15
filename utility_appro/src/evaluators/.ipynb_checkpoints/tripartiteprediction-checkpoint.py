import torch
import numpy as np
from torch.utils.data import Subset
from opendataval.dataval.api import DataEvaluator, ModelMixin
from scipy.spatial.distance import cdist
import tqdm
from sklearn.base import clone

class DualThresholdTripartiteEvaluator(DataEvaluator, ModelMixin):
    """基于双阈值三分图的数据估值方法
    
    三分图结构:
    1. 中间是训练节点
    2. 一边是同类别验证点(基于距离相似度大于阈值建立连接)
    3. 另一边是非同类别验证点(基于距离相似度小于阈值建立连接)
    
    评估策略:
    1. 同类验证点: 至少有一个被选中的训练点覆盖
    2. 非同类验证点: 所有被选中的训练点都保持连接(没有预测错误的风险)
    """
    
    def __init__(self, n_samples=10, threshold_range=None, random_state=None):
        super().__init__(random_state=random_state)
        
        self.n_samples = n_samples
        # 为同类和非同类分别创建阈值范围
        if threshold_range is None:
            self.pos_threshold_range = np.linspace(0.01, 0.99, 30)
            self.neg_threshold_range = np.linspace(0.01, 0.99, 30)
        else:
            self.pos_threshold_range = threshold_range
            self.neg_threshold_range = threshold_range
            
        self.optimal_pos_threshold = None
        self.optimal_neg_threshold = None

    def _get_class_labels(self, y):
        """获取类别标签"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel()

    def _compute_similarity_matrix(self, x_train, x_valid):
        """计算训练集和验证集之间的相似度矩阵"""
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.cpu().numpy()
        if isinstance(x_valid, torch.Tensor):    
            x_valid = x_valid.cpu().numpy()
            
        # 计算欧氏距离并标准化
        distances = cdist(x_train, x_valid, 'euclidean')
        distances = distances / distances.std()
        # 转换为相似度分数
        similarities = np.exp(-distances**2/2)
        
        return similarities


    def _precompute_class_similarities(self, similarity_matrix, train_labels, valid_labels):
        """预计算每个训练点的同类和非同类相似度"""
        n_train = len(train_labels)
        pos_similarities = []  # 存储每个训练点与同类验证点的相似度
        neg_similarities = []  # 存储每个训练点与非同类验证点的相似度
        
        for i in range(n_train):
            same_class = train_labels[i] == valid_labels
            diff_class = ~same_class
            pos_similarities.append(similarity_matrix[i][same_class])
            neg_similarities.append(similarity_matrix[i][diff_class])
            
        return pos_similarities, neg_similarities

    def _compute_valid_edges(self, similarity_matrix, pos_threshold, neg_threshold, train_labels, valid_labels):
        """计算有效的边"""
        if not hasattr(self, 'pos_similarities'):
            # 第一次调用时预计算
            self.pos_similarities, self.neg_similarities = self._precompute_class_similarities(
                similarity_matrix, train_labels, valid_labels
            )
        
        n_train, n_valid = similarity_matrix.shape
        positive_edges = np.zeros((n_train, n_valid), dtype=bool)
        negative_edges = np.zeros((n_train, n_valid), dtype=bool)
        
        # 对每个训练点
        for i in range(n_train):
            same_class = train_labels[i] == valid_labels
            diff_class = ~same_class
            
            # 使用预计算的相似度
            if len(self.pos_similarities[i]) > 0:
                pos_thresh = np.quantile(self.pos_similarities[i], pos_threshold)
                positive_edges[i] = same_class & (similarity_matrix[i] >= pos_thresh)
                
            if len(self.neg_similarities[i]) > 0:
                neg_thresh = np.quantile(self.neg_similarities[i], neg_threshold)
                negative_edges[i] = diff_class & (similarity_matrix[i] <= neg_thresh)
                
        return positive_edges, negative_edges
        
            
    def _compute_greedy_coverage(self, positive_edges, negative_edges):
        """计算最大覆盖序列"""
        n_train = positive_edges.shape[0]
        n_valid = positive_edges.shape[1]
        coverage_sequence = []
        
        # 复制邻接矩阵
        remaining_positive = positive_edges.copy()
        remaining_negative = negative_edges.copy()
        
        # 记录状态
        covered_positive = np.zeros(n_valid, dtype=bool)  # 已覆盖的同类点
        valid_negative = np.ones(n_valid, dtype=bool)     # 仍然有效的非同类点
        
        while len(coverage_sequence) < n_train:
            # 计算覆盖度
            pos_coverage = (remaining_positive & ~covered_positive).sum(axis=1)
            neg_coverage = (remaining_negative & valid_negative).sum(axis=1)
            total_coverage = pos_coverage + neg_coverage
            
            if total_coverage.max() == 0:
                remaining_points = list(set(range(n_train)) - set(coverage_sequence))
                self.random_state.shuffle(remaining_points)
                coverage_sequence.extend(remaining_points)
                break
                
            # 选择最大覆盖点
            best_idx = total_coverage.argmax()
            coverage_sequence.append(best_idx)
            
            # 更新positive side - 移除已覆盖的验证点
            newly_covered = remaining_positive[best_idx] > 0
            covered_positive |= newly_covered
            remaining_positive[best_idx] = 0
            
            # 更新negative side - 将没有连接的点立即移除
            unconnected = ~remaining_negative[best_idx] & valid_negative
            valid_negative[unconnected] = False
            remaining_negative[best_idx] = 0
            
        return coverage_sequence

    def _generate_subsets_and_accuracies(self, x_train, y_train, x_valid, y_valid):
        """预先生成所有子集及其对应的验证准确率"""
        print("预先生成子集和计算准确率...")
        
        # 生成随机采样比例
        ratios = self.random_state.uniform(0.1, 0.9, self.n_samples)
        
        subsets = []  # 存储所有子集
        accuracies = []  # 存储对应的准确率
        
        for ratio in tqdm.tqdm(ratios):
            size = max(1, int(ratio * len(x_train)))
            indices = self.random_state.choice(len(x_train), size, replace=False)
            
            # 训练子集模型获取准确率
            subset_model = self.pred_model.clone()
            subset_model.fit(
                Subset(x_train, indices),
                Subset(y_train, indices)
            )
            y_pred = subset_model.predict(x_valid)
            val_acc = self.evaluate(y_valid, y_pred)
            
            subsets.append(indices)
            accuracies.append(val_acc)
            
        return subsets, accuracies

    def _find_optimal_thresholds(self, x_train, y_train, x_valid, y_valid):
        """通过网格搜索找到最优阈值对"""
        # 先计算相似度矩阵
        base_similarity = self._compute_similarity_matrix(x_train, x_valid)
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(y_valid)
        
        # 预先生成子集和计算准确率
        subsets, accuracies = self._generate_subsets_and_accuracies(
            x_train, y_train, x_valid, y_valid
        )
        
        print("开始网格搜索最优阈值...")
        best_pos_threshold = None
        best_neg_threshold = None
        min_error = float('inf')
        
        # 网格搜索
        for pos_threshold in tqdm.tqdm(self.pos_threshold_range):
            for neg_threshold in self.neg_threshold_range:
                # 计算两种边
                pos_edges, neg_edges = self._compute_valid_edges(
                    base_similarity, pos_threshold, neg_threshold,
                    train_labels, valid_labels
                )
                
                mse = 0
                for indices, true_acc in zip(subsets, accuracies):
                    # 取子集的边
                    pos_edges_subset = pos_edges[indices]
                    neg_edges_subset = neg_edges[indices]
                    
                    # 对每个验证点,检查是否:
                    # 1. 至少有一个同类训练点覆盖
                    has_pos_coverage = pos_edges_subset.any(axis=0)
                    # 2. 所有非同类训练点都保持连接(预测正确)
                    maintains_neg_edges = (~neg_edges_subset).all(axis=0)
                    
                    # 计算同时满足两个条件的验证点比例
                    predicted_correct = (has_pos_coverage & maintains_neg_edges).mean()
                    #print('predicted_correct',predicted_correct,'true_acc',true_acc)
                    # 计算MSE
                    mse += (predicted_correct - true_acc)**2
                    
                avg_mse = mse / len(subsets)
                if avg_mse < min_error:
                    min_error = avg_mse
                    best_pos_threshold = pos_threshold
                    best_neg_threshold = neg_threshold
                    
        return best_pos_threshold, best_neg_threshold, min_error

    def train_data_values(self, *args, **kwargs):
        """训练数据估值模型"""
        print(f"开始寻找最优阈值对 (采样{self.n_samples}个不同大小的子集)...")
        self.optimal_pos_threshold, self.optimal_neg_threshold, self.validation_error = (
            self._find_optimal_thresholds(
                self.x_train, self.y_train,
                self.x_valid, self.y_valid
            )
        )
        print(f"找到最优阈值对: pos={self.optimal_pos_threshold:.3f}, "
              f"neg={self.optimal_neg_threshold:.3f}, "
              f"MSE={self.validation_error:.3f}")
        
        # 计算相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(self.x_train, self.x_valid)
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        # 使用最优阈值计算两类边
        self.positive_edges, self.negative_edges = self._compute_valid_edges(
            similarity_matrix,
            self.optimal_pos_threshold, 
            self.optimal_neg_threshold,
            train_labels, 
            valid_labels
        )
        
        # 使用贪心策略找到覆盖序列
        self.coverage_sequence = self._compute_greedy_coverage(
            self.positive_edges, 
            self.negative_edges
        )
        
        # 基于序列位置计算数据值
        n_train = len(self.x_train)
        self.data_values = np.zeros(n_train)
        for i, idx in enumerate(self.coverage_sequence):
            self.data_values[idx] = n_train - i
            
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值"""
        normalized_values = (self.data_values - self.data_values.min())
        if self.data_values.max() > self.data_values.min():
            normalized_values /= (self.data_values.max() - self.data_values.min())
        return normalized_values