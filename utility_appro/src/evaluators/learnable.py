import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import Optional, Union, Callable
import numpy as np
import tqdm
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.metrics import Metrics

class EmbeddingNetwork(nn.Module):
    """特征映射网络,将原始特征映射到低维空间
    
    将输入特征通过多层感知机映射到低维空间,使相似数据点在该空间中的距离更近
    
    Args:
        input_dim: 输入特征维度
        embedding_dim: 目标embedding维度
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embedding_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class EdgePredictor(nn.Module):
    """边预测器网络，使用LayerNorm替代BatchNorm以支持单样本处理
    
    预测两个数据点之间是否存在有效连接。使用LayerNorm代替BatchNorm以支持单样本处理。
    
    Args:
        embedding_dim: embedding特征维度
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
             nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """前向传播,计算两个数据点之间的连接概率
        
        Args:
            x1: 第一个数据点的embedding
            x2: 第二个数据点的embedding
            
        Returns:
            两点间的连接概率
        """
        assert x1.dim() == 2 and x2.dim() == 2
        assert x1.size(1) == x2.size(1)
        combined = torch.cat([x1, x2], dim=1)
        return self.network(combined)

class LearnableEmbeddingMatchingEvaluator(DataEvaluator, ModelMixin):
    """结合可学习embedding和预测验证的数据评估器
    
    特点:
    1. 学习特征的低维表示
    2. 基于预测改善验证点的效果学习边的预测
    3. 使用阈值选择和贪心匹配确定最终数据价值
    
    Args:
        embedding_dim: embedding维度,None则根据输入维度自动设置
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        threshold_range: 阈值搜索范围
        n_samples: 每个阈值的采样次数
        num_trials: 训练数据生成的试验次数
        samples_per_trial: 每次试验的采样数量
        metric: 评估指标
        device: 运行设备('cuda'或'cpu')
        random_state: 随机数生成器
    """
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        threshold_range: Optional[np.ndarray] = None,
        n_samples: int = 10,
        num_trials: int = 50,
        samples_per_trial: int = 100,
        metric: Optional[Union[str, Callable, Metrics]] = None,
        device: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None
    ):
        super().__init__(random_state=random_state)
        # 设置网络运行设备
        self.device = torch.device(device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_samples = n_samples
        self.num_trials = num_trials
        self.samples_per_trial = samples_per_trial
        self.threshold_range = (
            np.linspace(0.1, 0.9, 20) if threshold_range is None else threshold_range
        )
        
        if metric is not None:
            self.input_metric(metric)
            
    def _to_numpy(self, data):
        """将数据转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)
        
    def input_metric(self, metric):
        """输入评估指标
        
        支持多种形式的度量标准输入:
        1. 字符串 (如 "accuracy")
        2. Metrics枚举值 (如 Metrics.ACCURACY)
        3. 可调用函数
        """
        if metric is None:
            self.metric = Metrics.ACCURACY
        elif isinstance(metric, str):
            self.metric = Metrics(metric)
        elif isinstance(metric, Metrics):
            self.metric = metric
        elif callable(metric):
            self.metric = metric
        else:
            raise ValueError(f"不支持的度量标准类型: {type(metric)}")
        return self
            
    def _get_class_labels(self, y):
        """获取类别标签"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel()
    
    def _initialize_networks(self, input_dim):
        """初始化embedding网络和边预测器"""
        if self.embedding_dim is None:
            self.embedding_dim = max(input_dim // 2, 1)
                
        self.embedding_net = EmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        self.edge_predictor = EdgePredictor(
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        return self.embedding_net, self.edge_predictor
    


    def _generate_edge_training_data(self, model_kwargs):
        """生成用于训练边预测器的数据
        
        通过比较加入新训练点前后的预测改善来生成训练样本:
        1. 为每个类别选择一个初始点,确保所有类别都有代表
        2. 逐个添加新的训练点并评估其影响
        3. 如果新点能改善某些验证点的预测,这些点对构成正样本
        4. 如果新点未能改善预测,这些点对构成负样本
        5. 确保正负样本数量平衡且满足类别约束
        
        Args:
            model_kwargs: 传递给模型训练的参数
    
        Returns:
            tuple: (训练点对的张量, 边标签的张量)
        """
        train_pairs = []
        edge_labels = []
        
        # 获取类别信息
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        unique_classes = np.unique(train_labels)
        
        # 进行多次试验
        for trial in range(self.num_trials):
            if trial % 10 == 0:
                print(f"Trial {trial}")
                
            # 为每个类别随机选择一个基础点
            base_points = []
            for cls in unique_classes:
                # 找出属于当前类别的所有点
                cls_indices = np.where(train_labels == cls)[0]
                # 随机选择一个作为该类的代表点
                base_idx = self.random_state.choice(cls_indices, size=1)[0]
                base_points.append(base_idx)
                
            # 将基础点集合转换为set以方便更新
            current_points = set(base_points)
            
            # 训练初始模型
            base_model = self.pred_model.clone()
            base_model.fit(
                Subset(self.x_train, list(current_points)),
                Subset(self.y_train, list(current_points)),
                **model_kwargs
            )
            base_preds = self._evaluate_predictions(base_model, self.x_valid, self.y_valid)
            
            # 随机选择要评估的新点
            remaining_points = list(set(range(len(self.x_train))) - current_points)
            if len(remaining_points) > self.samples_per_trial:
                selected_points = self.random_state.choice(
                    remaining_points,
                    size=self.samples_per_trial, 
                    replace=False
                )
            else:
                selected_points = remaining_points
                
            # 评估每个新点的效果
            for point_idx in selected_points:
                # 训练包含新点的模型
                new_points = list(current_points | {point_idx})
                curr_model = self.pred_model.clone()
                curr_model.fit(
                    Subset(self.x_train, new_points),
                    Subset(self.y_train, new_points),
                    **model_kwargs
                )
                curr_preds = self._evaluate_predictions(curr_model, self.x_valid, self.y_valid)
                
                # 找到预测改善的验证点(之前错误现在正确)
                improved_valid = (~base_preds) & curr_preds
                # 找到预测仍然错误的验证点  
                still_wrong = (~base_preds) & (~curr_preds)
                
                # 应用类别约束
                class_match = train_labels[point_idx] == valid_labels
                improved_points = improved_valid & class_match
                wrong_points = still_wrong & class_match
                
                if improved_points.any():
                    # 添加正样本
                    improved_indices = np.where(improved_points)[0]
                    for valid_idx in improved_indices:
                        train_pairs.append((point_idx, valid_idx))
                        edge_labels.append(1)
                    
                    # 从未改善的点中采样等量负样本
                    wrong_indices = np.where(wrong_points)[0]
                    if len(wrong_indices) > 0:
                        # 上采样确保负样本数量与正样本相同
                        num_samples = len(improved_indices)
                        if len(wrong_indices) < num_samples:
                            # 如果负样本不足,进行有放回采样 
                            neg_samples = self.random_state.choice(
                                wrong_indices,
                                size=num_samples,
                                replace=True
                            )
                        else:
                            # 如果负样本充足,进行无放回采样
                            neg_samples = self.random_state.choice(
                                wrong_indices, 
                                size=num_samples,
                                replace=False
                            )
                        
                        for valid_idx in neg_samples:
                            train_pairs.append((point_idx, valid_idx))
                            edge_labels.append(0)
                
                # 更新当前点集和基准预测
                current_points.add(point_idx)
                base_preds = curr_preds
    
        if len(train_pairs) == 0:
            raise ValueError("未能生成有效的训练样本对")
            
        return (
            torch.tensor(train_pairs, device=self.device),
            torch.tensor(edge_labels, dtype=torch.float32, device=self.device)
        )

    
    
    def _evaluate_predictions(self, model, x, y):
        """评估预测是否正确"""
        preds = model.predict(x)
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds = preds.argmax(axis=1)
            y = y.argmax(axis=1)
        return preds == y

    def _compute_edge_probabilities(self):
        """计算所有训练点和验证点之间的边概率"""
        x_train_tensor = (
            self.x_train.to(self.device)
            if isinstance(self.x_train, torch.Tensor)
            else torch.tensor(self.x_train, device=self.device)
        )
        x_valid_tensor = (
            self.x_valid.to(self.device)
            if isinstance(self.x_valid, torch.Tensor)
            else torch.tensor(self.x_valid, device=self.device)
        )
        
        edge_probs = []
        
        with torch.no_grad():
            train_embeddings = self.embedding_net(x_train_tensor)
            valid_embeddings = self.embedding_net(x_valid_tensor)
            
            for i in range(0, len(train_embeddings), self.batch_size):
                batch_probs = []
                train_batch = train_embeddings[i:i+self.batch_size]
                
                for j in range(0, len(valid_embeddings), self.batch_size):
                    valid_batch = valid_embeddings[j:j+self.batch_size]
                    
                    train_expanded = train_batch.unsqueeze(1).expand(-1, len(valid_batch), -1)
                    valid_expanded = valid_batch.unsqueeze(0).expand(len(train_batch), -1, -1)
                    
                    flat_train = train_expanded.reshape(-1, self.embedding_dim)
                    flat_valid = valid_expanded.reshape(-1, self.embedding_dim)
                    
                    probs = self.edge_predictor(flat_train, flat_valid)
                    probs = probs.reshape(len(train_batch), len(valid_batch))
                    batch_probs.append(probs)
                
                edge_probs.append(torch.cat(batch_probs, dim=1))
                
        return torch.cat(edge_probs, dim=0).cpu().numpy()
    
    def _compute_greedy_coverage(self, valid_edges):
        """使用贪心策略计算覆盖序列"""
        n_train = len(self.x_train)
        coverage_sequence = []
        remaining_edges = valid_edges.copy()
        remaining_valid = np.ones(len(self.x_valid), dtype=bool)
        
        while True:
            coverage_counts = (remaining_edges & remaining_valid).sum(axis=1)
            
            if coverage_counts.max() == 0:
                remaining_points = set(range(n_train)) - set(coverage_sequence)
                remaining_list = list(remaining_points)
                self.random_state.shuffle(remaining_list)
                coverage_sequence.extend(remaining_list)
                break
                
            best_point = np.argmax(coverage_counts)
            coverage_sequence.append(best_point)
            
            newly_covered = remaining_edges[best_point] & remaining_valid
            remaining_valid[newly_covered] = False
            remaining_edges[:, newly_covered] = False
            
        return coverage_sequence
    
    
    def _find_optimal_threshold(self, edge_probs, model_kwargs):
        """找到最优阈值
        
        通过比较不同阈值下的图覆盖率和验证准确率来找到最优阈值。
        对每个阈值:
        1. 多次随机采样训练子集
        2. 计算图覆盖率(被选中训练点能覆盖的验证点比例)
        3. 在子集上训练模型得到验证准确率
        4. 选择使覆盖率和准确率最接近的阈值
        """
        print("开始寻找最优阈值...")
        
        # 准备数据并转换到CPU
        x_train = self._to_numpy(self.x_train)
        y_train = self._to_numpy(self.y_train)
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        best_threshold = None
        min_error = float('inf')
        
        # 遍历所有候选阈值
        for threshold in tqdm.tqdm(self.threshold_range, desc="尝试不同阈值"):
            try:
                mse = 0
                for sample_idx in range(self.n_samples):
                    try:
                        # 随机采样训练子集
                        sample_size = self.random_state.randint(
                            len(x_train) // 4,
                            len(x_train) // 2
                        )
                        indices = self.random_state.choice(
                            len(x_train),
                            size=sample_size,
                            replace=False
                        )
                        
                        # 计算该阈值下的有效边
                        valid_edges = (edge_probs >= threshold) & (
                            train_labels[:, np.newaxis] == valid_labels
                        )
                        valid_edges_subset = valid_edges[indices]
                        
                        # 计算图覆盖率
                        coverage = valid_edges_subset.any(axis=0).mean()
                        
                        # 在子集上训练模型并计算验证准确率
                        subset_model = self.pred_model.clone()
                        subset_model.fit(
                            x_train[indices],
                            y_train[indices],
                            **model_kwargs
                        )
                        val_acc = self.evaluate(
                            self.y_valid,
                            subset_model.predict(self.x_valid)
                        )
                        
                        # 计算覆盖率和准确率的均方差
                        mse += (coverage - val_acc) ** 2
                        
                        if sample_idx % (self.n_samples // 5) == 0:
                            print(f"\r阈值 {threshold:.3f} - 样本 {sample_idx+1}/{self.n_samples}"
                                  f" - 当前MSE: {mse/(sample_idx+1):.4f}", end="")
                            
                    except Exception as e:
                        print(f"\n样本 {sample_idx} 处理出错: {str(e)}")
                        continue
                        
                # 计算平均MSE并更新最优阈值
                avg_mse = mse / self.n_samples
                if avg_mse < min_error:
                    min_error = avg_mse
                    best_threshold = threshold
                    print(f"\n找到新的最优阈值: {best_threshold:.3f}, MSE: {min_error:.4f}")
                    
            except Exception as e:
                print(f"\n阈值 {threshold} 处理出错: {str(e)}")
                continue
        
        if best_threshold is None:
            print("警告: 未找到有效阈值,使用默认值0.5")
            return 0.5, float('inf')
        
        print(f"\n完成阈值搜索 - 最优阈值: {best_threshold:.3f}, 最终MSE: {min_error:.4f}")
        return best_threshold, min_error

    def train_data_values(self, *args, **kwargs):
        """训练数据评估器
        
        整体训练流程:
        1. 初始化embedding网络和边预测器
        2. 生成边预测的训练数据
        3. 训练网络学习特征表示和边的预测
        4. 计算所有可能边的概率
        5. 寻找最优阈值
        6. 使用贪心策略计算覆盖序列
        7. 基于序列位置分配数据价值
        
        Args:
            *args: 可变位置参数,传递给模型训练
            **kwargs: 可变关键字参数,传递给模型训练
            
        Returns:
            self: 返回训练好的评估器实例
        """
        # 获取输入特征维度
        input_dim = (
            self.x_train.shape[1] 
            if isinstance(self.x_train, np.ndarray)
            else self.x_train[0].shape[0]
        )
        
        try:
            # 初始化网络
            print("初始化网络...")
            self._initialize_networks(input_dim)
            
            # 生成训练数据
            print("生成边预测训练数据...")
            train_pairs, edge_labels = self._generate_edge_training_data(kwargs)
            
            # 训练网络
            print("训练embedding网络和边预测器...")
            self._train_networks(train_pairs, edge_labels)
            
            # 计算所有边的概率
            print("计算边概率...")
            edge_probs = self._compute_edge_probabilities()
            
            # 找到最优阈值
            print("寻找最优阈值...")
            self.optimal_threshold, self.validation_error = self._find_optimal_threshold(
                edge_probs,
                kwargs
            )
            print(f"找到最优阈值: {self.optimal_threshold:.3f}, 验证MSE: {self.validation_error:.3f}")
    
            # 计算有效边 - 同时满足概率阈值和类别匹配条件
            valid_edges = (edge_probs >= self.optimal_threshold) & (
                self._get_class_labels(self.y_train)[:, np.newaxis] == 
                self._get_class_labels(self.y_valid)
            )
            
            # 使用贪心策略找到覆盖序列
            coverage_sequence = self._compute_greedy_coverage(valid_edges)
            
            # 基于序列位置计算数据值 - 位置越靠前价值越高
            n_train = len(self.x_train)
            self._raw_data_values = np.zeros(n_train)
            for i, idx in enumerate(coverage_sequence):
                self._raw_data_values[idx] = n_train - i
    
        except Exception as e:
            print(f"训练过程发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            raise
            
        return self
    
    def _train_networks(self, train_pairs, edge_labels):
        """训练embedding网络和边预测器
        
        使用mini-batch方式训练网络,包含:
        1. 数据移动到正确设备
        2. 前向传播计算loss
        3. 反向传播更新参数
        4. 梯度裁剪防止爆炸
        """
        try:
            print(f'使用设备: {self.device}')
            
            # 初始化优化器
            optimizer = torch.optim.Adam(
                list(self.embedding_net.parameters()) + 
                list(self.edge_predictor.parameters()),
                lr=self.lr
            )
            
            dataset = TensorDataset(train_pairs, edge_labels)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            print(f'开始训练: {len(loader)} 批次的数据')
            
            # 确保模型处于训练模式
            self.embedding_net.train()
            self.edge_predictor.train()
            
            # 预先将训练数据和验证数据移至目标设备
            if isinstance(self.x_train, np.ndarray):
                self.x_train = torch.tensor(self.x_train, device=self.device)
            else:
                self.x_train = self.x_train.to(self.device)
                
            if isinstance(self.x_valid, np.ndarray):
                self.x_valid = torch.tensor(self.x_valid, device=self.device)
            else:
                self.x_valid = self.x_valid.to(self.device)
            
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                processed_batches = 0
                
                # 使用tqdm显示进度条
                for batch_pairs, batch_labels in tqdm.tqdm(loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                    try:
                        # 将索引移至正确设备
                        train_indices = batch_pairs[:, 0].to(self.device)
                        valid_indices = batch_pairs[:, 1].to(self.device)
                        
                        # 直接索引已在设备上的数据
                        x_train_batch = self.x_train[train_indices]
                        x_valid_batch = self.x_valid[valid_indices]
                        
                        # 获取embedding
                        train_embeddings = self.embedding_net(x_train_batch)
                        valid_embeddings = self.embedding_net(x_valid_batch)
                        
                        # 预测边的概率
                        pred_probs = self.edge_predictor(train_embeddings, valid_embeddings)
                        
                        # 计算损失
                        loss = F.binary_cross_entropy(
                            pred_probs.squeeze(),
                            batch_labels.to(self.device)
                        )
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪防止爆炸
                        torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(self.edge_predictor.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        processed_batches += 1
                        
                    except Exception as e:
                        print(f"批次处理出错: {e}")
                        continue
                
                # 每个epoch结束后打印统计信息
                avg_loss = epoch_loss / processed_batches if processed_batches > 0 else float('inf')
                print(f'\nEpoch {epoch+1}/{self.num_epochs} - 平均损失: {avg_loss:.4f}')
                
                # 每10个epoch保存检查点
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} 检查点 - 当前损失: {avg_loss:.4f}")
            
            print("\n训练完成!")
            
        except Exception as e:
            print(f"训练过程发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            raise
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值
        
        将基于覆盖序列计算的原始数据值归一化到[0,1]区间。
        注意:这个方法需要先调用train_data_values来计算原始数据值。
        
        Returns:
            np.ndarray: 归一化后的数据点价值,维度为(n_samples,)
        """
        # 确保已经通过train_data_values计算了原始数据值
        if not hasattr(self, '_raw_data_values'):
            raise ValueError("请先调用train_data_values方法计算数据值")
            
        # 如果所有点价值相同,返回全1数组    
        if self._raw_data_values.max() == self._raw_data_values.min():
            return np.ones_like(self._raw_data_values)
            
        # 最小-最大归一化    
        normalized_values = (self._raw_data_values - self._raw_data_values.min())
        normalized_values = normalized_values / (self._raw_data_values.max() - self._raw_data_values.min())
        
        return normalized_values


