import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm  # 引入进度条库，需 pip install tqdm


class HARR:
    def __init__(self, X, n_clusters=2, numerical_cols=None, nominal_cols=None, ordinal_cols=None):
        self.X = X.copy()
        self.X_encoded = None
        self.n_clusters = n_clusters

        self.numerical_cols = list(numerical_cols) if numerical_cols is not None else []
        self.nominal_cols = list(nominal_cols) if nominal_cols is not None else []
        self.ordinal_cols = list(ordinal_cols) if ordinal_cols is not None else []
        self.categorical_cols = self.nominal_cols + self.ordinal_cols

    def calculate_base_distance_vectorized(self, target_col, context_df):
        """
        [优化版] 计算基础距离矩阵
        使用 pd.crosstab 替代循环筛选，极大提升速度
        """
        target_values = self.X[target_col].unique()
        # 确保顺序一致，便于后续 DataFrame 索引
        target_values.sort()

        # 存储所有上下文属性生成的概率分布片段
        # 最终形状应为: (n_target_values, n_total_context_values)
        profile_blocks = []

        context_cols = [c for c in context_df.columns if c != target_col]

        for ctx_col in context_cols:
            # 1. 交叉表统计: 行=TargetVal, 列=ContextVal (方便后续 concat)
            # 论文逻辑是 P(Target=v | Context=c)，即给定 Context 下 Target 的分布
            # 所以我们先算 Joint Count，然后按 Context 列归一化

            # crosstab: index=Target, columns=Context
            ct = pd.crosstab(self.X[target_col], context_df[ctx_col])

            # 2. 归一化: 按列 (Context Value) 求和，计算 P(Target | Context)
            # 每一列代表一个特定的 Context Value 下，Target Value 的概率分布
            # 加上 1e-10 防止除零
            prob_matrix = ct.div(ct.sum(axis=0) + 1e-10, axis=1)

            # 确保包含所有 target_values (防止某些值从未出现导致行缺失)
            prob_matrix = prob_matrix.reindex(target_values, fill_value=0)

            profile_blocks.append(prob_matrix)

        # 3. 横向拼接所有上下文的分布特征
        # 最终矩阵: 行是 Target 的每个取值，列是所有 Context 属性的所有可能取值
        full_profile_matrix = pd.concat(profile_blocks, axis=1)

        # 4. 计算行与行之间的曼哈顿距离 (CityBlock)
        # metric='cityblock' 对应论文中的 sum(|p - p'|)
        dist_array = pdist(full_profile_matrix.values, metric='cityblock')

        dist_matrix = pd.DataFrame(
            squareform(dist_array),
            index=full_profile_matrix.index,
            columns=full_profile_matrix.index
        )

        return dist_matrix

    @staticmethod
    def project_nominal(kappa):
        n_values = len(kappa.index)
        if n_values < 2:
            return np.zeros((n_values, 1), dtype=np.float32)

        pairs = list(combinations(kappa.index, 2))

        # 预分配 numpy 数组加速
        n_dims = len(pairs)
        projected_matrix = np.zeros((n_values, n_dims), dtype=np.float32)

        # 获取索引映射，避免 loc 的开销
        idx_map = {val: i for i, val in enumerate(kappa.index)}
        kappa_values = kappa.values

        for dim_idx, (g, h) in enumerate(pairs):
            idx_g, idx_h = idx_map[g], idx_map[h]
            dist_g_h = kappa_values[idx_g, idx_h]

            if dist_g_h > 1e-9:  # 避免除零
                dist_t_g = kappa_values[:, idx_g]
                dist_t_h = kappa_values[:, idx_h]

                # 向量化计算
                numerator = np.square(dist_t_g) - np.square(dist_t_h) + np.square(dist_g_h)
                denominator = 2 * dist_g_h
                coords = numerator / denominator
                projected_matrix[:, dim_idx] = coords

        return projected_matrix

    @staticmethod
    def project_ordinal(kappa):
        values = kappa.index
        n_values = len(values)
        if n_values < 2:
            return np.zeros((n_values, 1), dtype=np.float32)

        # 假设 kappa.index 已经有序，取第一个作为原点
        # 直接取第一列 (所有点到第一个点的距离)
        coords = kappa.iloc[:, 0].values.astype(np.float32)
        return coords.reshape(-1, 1)

    def preprocess(self):
        print("正在进行预处理 (HARR Preprocess)...")

        # 1. 数值属性处理
        X_num = pd.DataFrame()
        if self.numerical_cols:
            num_data = self.X[self.numerical_cols]
            range_val = num_data.max() - num_data.min()
            range_val[range_val == 0] = 1
            X_num = (num_data - num_data.min()) / range_val
            # 转为 float32
            X_num = X_num.astype(np.float32)

        # 2. 准备上下文
        context_df = self.X[self.categorical_cols].copy()
        if self.numerical_cols:
            est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            X_num_discrete = est.fit_transform(self.X[self.numerical_cols])
            X_num_discrete_df = pd.DataFrame(
                X_num_discrete,
                columns=[f"{c}_disc" for c in self.numerical_cols],
                index=self.X.index
            )
            context_df = pd.concat([context_df, X_num_discrete_df], axis=1)

        encoding_frames = []

        # 3. 遍历分类属性 (添加进度条)
        # 使用 tqdm 显示进度
        for col in tqdm(self.categorical_cols, desc="Calculating Kappa & Projecting"):
            # 使用向量化方法计算 Kappa
            col_kappa = self.calculate_base_distance_vectorized(col, context_df)

            if col in self.ordinal_cols:
                col_vectors = self.project_ordinal(col_kappa)
            else:
                col_vectors = self.project_nominal(col_kappa)

            # 映射回数据
            # 使用 numpy 索引映射比 map 更快
            val_to_idx = {val: i for i, val in enumerate(col_kappa.index)}
            # 将原始列转换为索引
            col_indices = self.X[col].map(val_to_idx).fillna(-1).astype(int).values

            # 根据索引直接获取投影向量 (N_samples, N_dims)
            expanded_data = col_vectors[col_indices]

            # 归一化 (Min-Max)
            col_min = expanded_data.min(axis=0)
            col_max = expanded_data.max(axis=0)
            col_range = col_max - col_min
            col_range[col_range == 0] = 1
            expanded_data = (expanded_data - col_min) / col_range

            # 转为 DataFrame (为了后续 concat)
            expanded_df = pd.DataFrame(
                expanded_data,
                index=self.X.index,
                columns=[f'{col}_{i}' for i in range(expanded_data.shape[1])]
            )
            encoding_frames.append(expanded_df)

        if encoding_frames:
            X_cat_encoded = pd.concat(encoding_frames, axis=1)
            self.X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)
        else:
            self.X_encoded = X_num

        # 强制转换为 float32 节省内存
        self.X_encoded = self.X_encoded.astype(np.float32)
        print(f"预处理完成。扩展后特征维度: {self.X_encoded.shape[1]}")
        return self.X_encoded

    def fitV(self, max_iter=50, epsilon=1e-6):  # 减少默认迭代次数
        if self.X_encoded is None:
            raise ValueError("Run preprocess first.")

        n_samples, n_attributes = self.X_encoded.shape
        X = self.X_encoded.values  # float32

        # 随机初始化
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        M = X[random_idx].copy()

        weights = np.ones(n_attributes, dtype=np.float32) / n_attributes

        Q = np.zeros(n_samples, dtype=int)
        Q_prime = np.zeros(n_samples, dtype=int)

        # 优化：预分配距离矩阵
        dist_matrix = np.zeros((n_samples, self.n_clusters), dtype=np.float32)

        cur_step = 3
        # 使用 tqdm 显示迭代进度
        pbar = tqdm(range(max_iter), desc="HARR-V Iteration")

        for _ in pbar:
            if cur_step == -1: break

            # Step 3: 更新 Q
            if cur_step == 3:
                for j in range(self.n_clusters):
                    # 向量化加权距离
                    # abs(X - M) -> (N, D), * w -> (N, D), sum -> (N,)
                    dist_matrix[:, j] = np.sum(np.abs(X - M[j]) * weights, axis=1)

                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # Step 4: 更新 M
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    mask = (Q == j)
                    if np.any(mask):
                        M[j] = np.mean(X[mask], axis=0)
                cur_step = 3

            # Step 5: 更新 W
            elif cur_step == 5:
                # 这一步计算量大，进行向量化处理
                # 计算所有样本到所有中心的距离张量会爆内存 (8000 * 2 * 1000 * 4 bytes ≈ 64MB, 其实还行)
                # 但为了保险，我们可以稍微优化一下

                # Intra & Inter 计算
                D_r = np.zeros(n_attributes, dtype=np.float32)
                S_r = np.zeros(n_attributes, dtype=np.float32)

                for j in range(self.n_clusters):
                    # 中心 M[j]
                    diff = np.abs(X - M[j])  # (N, D)

                    mask_in = (Q == j)
                    mask_out = (Q != j)

                    if np.any(mask_in):
                        D_r += np.sum(diff[mask_in], axis=0)
                    if np.any(mask_out):
                        S_r += np.sum(diff[mask_out], axis=0)

                D_r /= n_samples
                S_r /= (n_samples * (self.n_clusters - 1))

                I_r = S_r / (D_r + epsilon)
                weights = I_r / np.sum(I_r)

                cur_step = -1  # 停止

        return Q, weights

    def fitM(self, max_iter=50, epsilon=1e-6):
        if self.X_encoded is None: raise ValueError("Preprocess first.")

        n_samples, n_attributes = self.X_encoded.shape
        X = self.X_encoded.values

        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        M = X[random_idx].copy()
        weights = np.full((self.n_clusters, n_attributes), 1 / n_attributes, dtype=np.float32)

        Q = np.zeros(n_samples, dtype=int)
        Q_prime = np.zeros(n_samples, dtype=int)
        dist_matrix = np.zeros((n_samples, self.n_clusters), dtype=np.float32)

        cur_step = 3
        pbar = tqdm(range(max_iter), desc="HARR-M Iteration")

        for _ in pbar:
            if cur_step == -1: break

            if cur_step == 3:
                for j in range(self.n_clusters):
                    # 使用对应的权重 weights[j]
                    dist_matrix[:, j] = np.sum(np.abs(X - M[j]) * weights[j], axis=1)
                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            elif cur_step == 4:
                for j in range(self.n_clusters):
                    mask = (Q == j)
                    if np.any(mask):
                        M[j] = np.mean(X[mask], axis=0)
                cur_step = 3

            elif cur_step == 5:
                # 优化内存的写法，不生成 (N, K, D) 的大张量
                new_weights = np.zeros_like(weights)

                for j in range(self.n_clusters):
                    diff_center = np.abs(X - M[j])  # (N, D)

                    mask_member = (Q == j)
                    mask_non_member = (Q != j)

                    n_member = np.sum(mask_member)
                    n_non_member = np.sum(mask_non_member)

                    if n_member > 0:
                        D_l = np.sum(diff_center[mask_member], axis=0) / n_member
                    else:
                        D_l = np.zeros(n_attributes)

                    if n_non_member > 0:
                        S_l = np.sum(diff_center[mask_non_member], axis=0) / n_non_member
                    else:
                        S_l = np.zeros(n_attributes)

                    I_l = S_l / (D_l + epsilon)
                    sum_I = np.sum(I_l)
                    if sum_I > 0:
                        new_weights[j] = I_l / sum_I
                    else:
                        new_weights[j] = 1.0 / n_attributes

                weights = new_weights
                cur_step = -1

        return Q, weights