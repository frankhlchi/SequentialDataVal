from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch  # 添加这行
from torch.utils.data import Subset
from scipy.spatial.distance import cdist
import tqdm
from sklearn.base import clone


class BipartiteMatchingEvaluator(DataEvaluator, ModelMixin):
    """基于类别感知的贪心二分图匹配的数据估值方法
    
    这个实现结合了:
    1. 自动寻找最优相似度阈值(基于分位数)
    2. 基于类别的贪心匹配策略
    3. 考虑了特征相似度和类别一致性
    """
    
    def __init__(self, n_samples=10, threshold_range=None, random_state=None):
        super().__init__(random_state=random_state)
        
        self.n_samples = n_samples
        if threshold_range is None:
            # 使用分位数范围
            self.threshold_range = np.linspace(0.5, 0.95, 1000)  # 关注高相似度区间
        else:
            self.threshold_range = threshold_range
            
        self.optimal_threshold = None

    def _get_class_labels(self, y):
        """获取类别标签"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        # 处理one-hot编码    
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
    
    def _precompute_similarity_distribution(self, x_train, x_valid, train_labels, valid_labels):
        """预计算类内相似度分布"""
        similarities = self._compute_similarity_matrix(x_train, x_valid)
        
        # 只保留同类点的相似度
        class_mask = train_labels.reshape(-1, 1) == valid_labels.reshape(1, -1)
        intra_class_similarities = similarities[class_mask]
        
        # 保存为实例变量
        self.base_similarity_matrix = similarities
        self.intra_class_similarities = intra_class_similarities
        
        return similarities

    def _compute_valid_edges(self, quantile, train_labels, valid_labels):
        """使用分位数计算有效边
        
        Args:
            quantile: 相似度分位数阈值
            train_labels: 训练集标签
            valid_labels: 验证集标签
            
        Returns:
            valid_edges: 布尔矩阵,表示训练点和验证点之间的有效连接
        """
        if not hasattr(self, 'base_similarity_matrix'):
            raise ValueError("需要先调用_precompute_similarity_distribution")
            
        # 使用分位数作为阈值
        threshold = np.quantile(self.intra_class_similarities, quantile)
        
        n_train = len(train_labels)
        n_valid = len(valid_labels)
        valid_edges = np.zeros((n_train, n_valid), dtype=bool)
        
        # 计算有效边
        for i in range(n_train):
            same_class = train_labels[i] == valid_labels
            high_similarity = self.base_similarity_matrix[i] >= threshold
            valid_edges[i] = same_class & high_similarity
            
        return valid_edges

    def _compute_greedy_coverage(self, valid_edges):
        """使用贪心策略计算最大覆盖序列"""
        n_train, n_valid = valid_edges.shape
        coverage_sequence = []
        remaining_edges = valid_edges.copy()
        remaining_valid = np.ones(n_valid, dtype=bool)
        
        while True:
            # 计算每个训练点能覆盖多少还未覆盖的验证点
            coverage_counts = (remaining_edges & remaining_valid).sum(axis=1)
            
            # 如果没有新的覆盖,添加剩余点
            if coverage_counts.max() == 0:
                remaining_points = set(range(n_train)) - set(coverage_sequence)
                remaining_list = list(remaining_points)
                self.random_state.shuffle(remaining_list)
                coverage_sequence.extend(remaining_list)
                break
                
            # 选择覆盖最多的点
            best_point = np.argmax(coverage_counts)
            coverage_sequence.append(best_point)
            
            # 更新剩余未覆盖的验证点
            newly_covered = remaining_edges[best_point] & remaining_valid
            remaining_valid[newly_covered] = False
            remaining_edges[:, newly_covered] = False
            
        return coverage_sequence

    def _generate_subsets_and_accuracies(self, x_train, y_train, x_valid, y_valid):
        """预先生成所有子集及其对应的验证准确率"""
        print("预先生成子集和计算准确率...")
        
        # 生成随机采样比例
        ratios = self.random_state.uniform(0.01, 0.99, self.n_samples)
        
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

    def _find_optimal_threshold(self, x_train, y_train, x_valid, y_valid):
        """通过网格搜索在分位数空间找到最优阈值"""
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(y_valid)
        
        # 预计算相似度分布
        self._precompute_similarity_distribution(x_train, x_valid, train_labels, valid_labels)
        
        # 预先生成子集和计算准确率
        subsets, accuracies = self._generate_subsets_and_accuracies(
            x_train, y_train, x_valid, y_valid
        )
        
        print("开始在分位数空间搜索最优阈值...")
        best_threshold = None
        min_error = float('inf')
        
        # 在分位数空间搜索
        for quantile in tqdm.tqdm(self.threshold_range):
            mse = 0
            # 使用分位数计算有效边
            valid_edges = self._compute_valid_edges(
                quantile, train_labels, valid_labels
            )
            
            # 对每个已生成的子集
            for subset_idx, (indices, true_acc) in enumerate(zip(subsets, accuracies)):
                # 计算子集的覆盖分数
                valid_edges_subset = valid_edges[indices]
                coverage = valid_edges_subset.any(axis=0).mean()
                mse += (coverage - true_acc) ** 2
                
            avg_mse = mse / len(subsets)
            if avg_mse < min_error:
                min_error = avg_mse
                best_threshold = quantile
                
        return best_threshold, min_error

    def train_data_values(self, *args, **kwargs):
        """训练数据估值模型"""
        print(f"开始寻找最优分位数阈值 (采样{self.n_samples}个不同大小的子集)...")
        self.optimal_threshold, self.validation_error = self._find_optimal_threshold(
            self.x_train, self.y_train,
            self.x_valid, self.y_valid
        )
        print(f"找到最优分位数阈值: {self.optimal_threshold:.3f}, "
              f"验证MSE: {self.validation_error:.3f}")
        
        # 获取类别标签
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        # 使用最优分位数计算有效边
        valid_edges = self._compute_valid_edges(
            self.optimal_threshold,
            train_labels,
            valid_labels
        )
        
        # 使用贪心策略找到覆盖序列
        self.coverage_sequence = self._compute_greedy_coverage(valid_edges)
        
        # 基于序列位置计算数据值
        n_train = len(self.x_train)
        self.data_values = np.zeros(n_train)
        for i, idx in enumerate(self.coverage_sequence):
            self.data_values[idx] = n_train - i
        print ('self.data_values', self.data_values)
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值"""
        normalized_values = (self.data_values - self.data_values.min()) 
        if self.data_values.max() > self.data_values.min():
            normalized_values /= (self.data_values.max() - self.data_values.min())
        return normalized_values