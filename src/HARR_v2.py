import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score

# 假设 HARRDataset 就在同一个目录下或已安装
from HARRDataset import HARRDataSet


class ImprovedHARR:
    def __init__(self, X, n_clusters=2, mds_components=2):
        """
        :param X: 输入的DataFrame
        :param n_clusters: 聚类簇数
        :param mds_components: MDS降维后的维度。建议设为 1, 2 或 3。
                               原论文那种膨胀到 v(v-1)/2 的做法是愚蠢的，
                               这里我们将其压缩到紧凑的流形空间。
        """
        self.X = X.copy()
        self.n_clusters = n_clusters
        self.mds_components = mds_components
        self.X_encoded = None
        self.feature_groups = {}  # 记录每个原始特征对应编码后的哪些列索引 {col_name: [idx1, idx2...]}
        self.weights = None  # 属性组的权重

    def calculate_base_distance(self, target_col, allowed_context_cols):
        """
        保留原论文中唯一有点道理的部分：利用上下文的条件概率分布差异来定义距离。
        这里计算的是 Statistical Distance (基于 Cityblock/L1)。
        """
        target_values = self.X[target_col].unique()
        # 如果只有一个值，距离为0
        if len(target_values) < 2:
            return pd.DataFrame(0, index=target_values, columns=target_values)

        probability_vector = {val: [] for val in target_values}
        context_cols = [c for c in allowed_context_cols if c != target_col]

        for ctx_col in context_cols:
            ctx_values = self.X[ctx_col].unique()

            # 预计算：避免在循环中反复做 DataFrame 过滤，提升效率
            # 实际生产中这里应该用 groupby，但为了保持逻辑清晰暂且保留原逻辑结构
            for ctx_val in ctx_values:
                mask = (self.X[ctx_col] == ctx_val)
                subset_indices = mask[mask].index

                if len(subset_indices) == 0:
                    for val in target_values:
                        probability_vector[val].append(0.0)
                    continue

                # 快速统计
                subset_target = self.X.loc[subset_indices, target_col]
                counts = subset_target.value_counts()
                subset_len = len(subset_target)

                for val in target_values:
                    prob = counts.get(val, 0) / subset_len
                    probability_vector[val].append(prob)

        profile_matrix = pd.DataFrame(probability_vector).T

        # 计算距离矩阵 (Dissimilarity Matrix)
        # 既然是概率分布，Cityblock (Total Variation) 是合理的
        dist_array = pdist(profile_matrix.values, metric='cityblock')

        dist_matrix = pd.DataFrame(
            squareform(dist_array),
            index=profile_matrix.index,
            columns=profile_matrix.index
        )

        return dist_matrix

    def preprocess(self, numerical_cols, categorical_cols):
        """
        改进后的预处理：
        1. 数值列归一化。
        2. 类别列：计算距离矩阵 -> MDS 降维 -> 获得紧凑的数值嵌入 (Embedding)。
        """
        # --- 兼容性修复 START ---
        # 如果传入的是单个数字（标量），强制转为列表
        if isinstance(numerical_cols, (int, np.integer)):
            numerical_cols = [numerical_cols]
        # 如果是 numpy array，转为 list
        elif hasattr(numerical_cols, 'tolist'):
            numerical_cols = numerical_cols.tolist()

        if isinstance(categorical_cols, (int, np.integer)):
            categorical_cols = [categorical_cols]
        elif hasattr(categorical_cols, 'tolist'):
            categorical_cols = categorical_cols.tolist()

        # 确保不为 None
        numerical_cols = numerical_cols if numerical_cols is not None else []
        categorical_cols = categorical_cols if categorical_cols is not None else []
        # --- 兼容性修复 END ---

        # 1. 处理数值列
        scaler = MinMaxScaler()
        if len(numerical_cols) > 0:
            # 这里加个由 DataFrame 到 DataFrame 的转换，防止索引丢失
            # 注意：如果 numerical_cols 是整数索引，X[numerical_cols] 可能会返回 Series，
            # 所以我们用 self.X.iloc[:, numerical_cols] 或者保证输入是列名

            # 假设 numerical_cols 存储的是列名或 DataFrame 接受的索引格式
            X_num_data = self.X[numerical_cols]

            # 如果是 Series（只有一列），将其转为 DataFrame
            if isinstance(X_num_data, pd.Series):
                X_num_data = X_num_data.to_frame()

            X_num = pd.DataFrame(scaler.fit_transform(X_num_data),
                                 columns=numerical_cols, index=self.X.index)
        else:
            X_num = pd.DataFrame(index=self.X.index)

        # 记录数值列的组信息（每个数值列自成一组）
        current_col_idx = 0
        for col in numerical_cols:
            self.feature_groups[col] = [current_col_idx]
            current_col_idx += 1

        # 2. 处理类别列 (MDS Embedding)
        encoded_dfs = []

        for col in categorical_cols:
            # 计算基础距离矩阵
            dist_matrix = self.calculate_base_distance(col, categorical_cols)
            n_values = len(dist_matrix)

            if n_values < 2:
                # 只有1个值的特征没有信息量，直接填0
                real_n_components = 1
                col_df = pd.DataFrame(0.0, index=self.X.index, columns=[f"{col}_mds_0"])
            else:
                # MDS 降维：解决原论文的维度爆炸问题
                n_comp = min(self.mds_components, n_values - 1)
                mds = MDS(n_components=n_comp, dissimilarity='precomputed',
                          random_state=42, normalized_stress='auto')
                embedding = mds.fit_transform(dist_matrix.values)
                vals = dist_matrix.index
                real_n_components = n_comp

                # 建立映射表
                mapping_dict = {val: embedding[i] for i, val in enumerate(vals)}

                # 映射回 DataFrame
                col_mapped = self.X[col].map(lambda x: mapping_dict.get(x, np.zeros(real_n_components)))
                col_df = pd.DataFrame(col_mapped.tolist(), index=self.X.index)
                col_df.columns = [f"{col}_mds_{i}" for i in range(real_n_components)]

                # 归一化 MDS 坐标
                col_df = pd.DataFrame(scaler.fit_transform(col_df),
                                      columns=col_df.columns, index=col_df.index)

            encoded_dfs.append(col_df)

            # 记录组信息
            group_indices = list(range(current_col_idx, current_col_idx + real_n_components))
            self.feature_groups[col] = group_indices
            current_col_idx += len(group_indices)

        # 合并所有特征
        if encoded_dfs:
            X_cat_encoded = pd.concat(encoded_dfs, axis=1)
            # 确保索引一致
            X_num.index = self.X.index
            self.X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)
        else:
            self.X_encoded = X_num

        return self.X_encoded

    def fit_improved(self, max_iter=50, tol=1e-4, lambda_reg=0.01):
        """
        改进后的聚类算法：Group-Weighted K-Means。
        不使用原论文那套复杂的迭代公式。
        逻辑：
        1. 标准 K-Means 指派。
        2. 计算每个 Feature Group (原始属性) 的组内离散度 (Dispersion)。
        3. 离散度越大的组，权重越小 (类似 Group Lasso 的特征选择思想)。
        """
        if self.X_encoded is None:
            raise ValueError("Data not preprocessed.")

        X = self.X_encoded.values
        n_samples, n_features = X.shape

        # 初始化中心
        random_idx = random.sample(range(n_samples), self.n_clusters)
        centroids = X[random_idx]

        # 初始化权重：每个原始特征(Group)权重相等
        n_groups = len(self.feature_groups)
        group_weights = {g_name: 1.0 / n_groups for g_name in self.feature_groups}

        # 将 Group 权重扩展到 Feature 维度方便计算
        # 比如 Cat_A 被 MDS 成了 2 列，这两列共享 Cat_A 的权重
        feature_weights = np.zeros(n_features)

        labels = np.zeros(n_samples)

        for iteration in range(max_iter):
            # --- 0. 构建特征权重向量 ---
            for g_name, indices in self.feature_groups.items():
                feature_weights[indices] = group_weights[g_name]

            # --- 1. E-Step: 分配簇 (基于加权欧氏距离) ---
            # Dist = sum( w_j * (x_ij - c_kj)^2 )
            # 这里的距离计算使用了 MDS 后的欧氏空间，物理意义比原论文的拼凑距离强得多
            distances = np.zeros((n_samples, self.n_clusters))
            for k in range(self.n_clusters):
                # 加权平方距离
                weighted_diff_sq = feature_weights * np.square(X - centroids[k])
                distances[:, k] = np.sum(weighted_diff_sq, axis=1)

            new_labels = np.argmin(distances, axis=1)

            # 判断收敛
            if iteration > 0 and np.sum(new_labels != labels) < tol * n_samples:
                break
            labels = new_labels

            # --- 2. M-Step: 更新中心 ---
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    centroids[k] = np.mean(X[mask], axis=0)
                else:
                    # 处理空簇：随机重置
                    centroids[k] = X[random.randint(0, n_samples - 1)]

            # --- 3. W-Step: 更新组权重 (Group Weighting) ---
            # 只有当 n_groups > 1 时才需要学习权重
            if n_groups > 1:
                # 计算每个组的 "Group Dispersion" (组内方差)
                # D_g = sum_k sum_{x in C_k} || x_g - c_kg ||^2
                group_dispersions = {}
                total_dispersion = 0.0

                diff_sq_all = np.square(X[:, np.newaxis, :] - centroids[labels, :])  # (N, dim)

                for g_name, indices in self.feature_groups.items():
                    # 取出属于该组的维度的平方差
                    g_diff_sq = diff_sq_all[:, indices]
                    # 求和：所有样本，在该组所有维度上的距离和
                    D_g = np.sum(g_diff_sq)
                    group_dispersions[g_name] = D_g

                # 简单的倒数权重策略 (类似 k-modes 的频率权重或 feature weighting k-means)
                # 如果某个属性很散 (D_g 很大)，权重应变小。
                # 加上一个小 epsilon 防止除零
                # 公式参考: w_g = 1 / D_g (归一化后)

                # 为了增强区分度，可以使用指数衰减或 Softmax
                # 这里使用简化的倒数归一化
                inv_dispersions = {g: 1.0 / (d + 1e-5) for g, d in group_dispersions.items()}
                sum_inv = sum(inv_dispersions.values())
                group_weights = {g: val / sum_inv for g, val in inv_dispersions.items()}

        self.weights = group_weights
        return labels, group_weights


if __name__ == '__main__':
    # 假设数据加载逻辑不变
    harr_data = HARRDataSet()
    all_data = harr_data.get_all_data_1()

    ARI_scores = {}

    # 仅运行一次演示，不再需要复杂的随机重复，因为 MDS 使得结果相对稳定
    for dataset_name, (dataset, columns) in all_data.items():
        print(f"Processing {dataset_name}...")

        # 实例化改进版 HARR
        # mds_components=2 意味着我们将复杂的类别关系映射到2D平面，足够表达大部分语义
        optimizer = ImprovedHARR(dataset, n_clusters=2, mds_components=2)

        # 预处理：MDS 替代了原来的暴力展开
        optimizer.preprocess(columns.numerical_columns, columns.categorical_columns)

        # 聚类：使用 Group-Weighted K-Means
        Q, w = optimizer.fit_improved(max_iter=50)

        # 评估
        ari = adjusted_rand_score(columns.y_true, Q)
        ARI_scores[dataset_name] = ARI_scores.get(dataset_name, []) + [ari]

        print(f'-- Improved HARR ARI: {ari:.4f}')
        # 打印一下学到的权重，看看哪些特征重要（排除了冗余特征）
        sorted_weights = sorted(w.items(), key=lambda x: x[1], reverse=True)
        print(f'-- Top 3 Important Attributes: {sorted_weights[:3]}')
        print("-" * 30)

    print("\nFinal Results:", ARI_scores)