import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


class HARR:
    def __init__(self, X, n_clusters=2):
        self.X = X.copy()
        self.X_encoded = None
        self.n_clusters = n_clusters

    def calculate_base_distance_optimized(self, target_col, allowed_context_cols):
        target_values = sorted(self.X[target_col].unique())
        collected_probs = []
        context_cols = [c for c in allowed_context_cols if c != target_col]

        for ctx_col in context_cols:
            ct = pd.crosstab(self.X[target_col], self.X[ctx_col])
            ctx_counts = self.X[ctx_col].value_counts()
            ctx_counts = ctx_counts[ct.columns]
            cond_probs = ct.div(ctx_counts, axis=1)
            collected_probs.append(cond_probs)

        if not collected_probs:
            return pd.DataFrame(0, index=target_values, columns=target_values)

        profile_matrix = pd.concat(collected_probs, axis=1).fillna(0)

        # 论文 Eq. (5) 使用绝对值差之和
        dist_array = pdist(profile_matrix.values, metric='cityblock')

        dist_matrix = pd.DataFrame(
            squareform(dist_array),
            index=profile_matrix.index,
            columns=profile_matrix.index
        )
        return dist_matrix

    @staticmethod
    def calculate_cast_distance_vector(kappa):
        n_values = len(kappa.index)
        if n_values < 2: return np.zeros((n_values, 1))

        pairs = list(combinations(kappa.index, 2))
        projected_columns = []

        for dim_idx, (g, h) in enumerate(pairs):
            dist_g_h = kappa.loc[g, h]
            if dist_g_h == 0:
                coords = np.zeros(n_values)
            else:
                dist_t_g = kappa.loc[:, g]
                dist_t_h = kappa.loc[:, h]
                numerator = np.square(dist_t_g) - np.square(dist_t_h) + np.square(dist_g_h)
                denominator = 2 * dist_g_h
                coords = numerator / denominator
            projected_columns.append(coords)

        return np.array(projected_columns).T

    def preprocess(self, numerical_cols, categorical_cols):
        # 辅助函数
        def to_list(cols):
            if hasattr(cols, '__iter__') and not isinstance(cols, str): return list(cols)
            return [cols] if cols is not None else []

        numerical_cols = to_list(numerical_cols)
        categorical_cols = to_list(categorical_cols)

        # 1. 数值列归一化 [0, 1]
        if len(numerical_cols) > 0:
            X_num = self.X[numerical_cols].copy().astype(float)
            for col in numerical_cols:
                _min, _max = X_num[col].min(), X_num[col].max()
                if _max != _min:
                    X_num[col] = (X_num[col] - _min) / (_max - _min)
                else:
                    X_num[col] = 0.0
        else:
            X_num = pd.DataFrame(index=self.X.index)

        # 2. 分类列 HARR 投影
        encoding_frames = []
        for col in categorical_cols:
            col_kappa = self.calculate_base_distance_optimized(col, categorical_cols)
            col_vectors = self.calculate_cast_distance_vector(col_kappa)

            mapping_dict = {val: col_vectors[i] for i, val in enumerate(col_kappa.index)}
            mapping_series = self.X[col].map(mapping_dict)

            expanded_df = pd.DataFrame(mapping_series.tolist(), index=self.X.index)
            expanded_df.columns = [f'{col}_{i}' for i in range(expanded_df.shape[1])]

            # 【核心修正 1】对投影后的分类属性也进行 Min-Max 归一化
            # 论文 Section 3.2: "divide ... by the maximum distance"
            # 这一步保证了分类属性的尺度与数值属性(0-1)一致，防止权重偏差
            _min = expanded_df.min()
            _max = expanded_df.max()
            # 避免除以0
            diff = _max - _min
            diff[diff == 0] = 1.0
            expanded_df = (expanded_df - _min) / diff

            encoding_frames.append(expanded_df)

        # 3. 合并
        if encoding_frames:
            X_cat_encoded = pd.concat(encoding_frames, axis=1)
            if not X_num.empty:
                self.X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)
            else:
                self.X_encoded = X_cat_encoded
        else:
            self.X_encoded = X_num

        return self.X_encoded

    def fitV(self, max_iter=100, epsilon=1e-4):
        # epsilon 调大一点点防止 D_r 极小导致 unstable
        if self.X_encoded is None: raise ValueError("Preprocessing required.")

        n_samples, n_attributes = self.X_encoded.shape
        X = self.X_encoded.values

        # 随机初始化
        random_idx = random.sample(range(n_samples), self.n_clusters)
        M = X[random_idx].copy()

        weights = np.ones(n_attributes) / n_attributes

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # === Step 3: 更新 Q ===
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    # 【核心修正 2】回归 Manhattan Distance (L1)
                    # 论文 Eq 14: sum( w * |x - m| )
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights, axis=1)

                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # === Step 4: 更新 M ===
            elif cur_step == 4:
                # 论文 Eq 2 明确使用 Mean
                # 虽然 Mean 最小化的是 L2，但在 K-Prototypes/HARR 类算法中
                # 常用 Mean + L1 这种启发式组合
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # === Step 5: 更新 w ===
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    # 计算 Manhattan 距离矩阵 (n, k, d)
                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])
                    labels_onehot = np.eye(self.n_clusters)[Q.astype(int)]

                    # D^r: Intra-cluster (L1)
                    intra_sum = np.sum(diff_all * labels_onehot[:, :, np.newaxis], axis=(0, 1))
                    D_r = intra_sum / n_samples

                    # S^r: Inter-cluster (L1)
                    inter_sum = np.sum(diff_all * (1 - labels_onehot[:, :, np.newaxis]), axis=(0, 1))
                    denom_S = n_samples * (self.n_clusters - 1)
                    S_r = inter_sum / (denom_S if denom_S > 0 else 1.0)

                    # Update Weights
                    I_r = S_r / (D_r + epsilon)
                    weights = I_r / np.sum(I_r)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights

    def fitM(self, max_iter=100, epsilon=1e-4):
        if self.X_encoded is None: raise ValueError("Preprocessing required.")

        n_samples, n_attributes = self.X_encoded.shape
        X = self.X_encoded.values

        random_idx = random.sample(range(n_samples), self.n_clusters)
        M = X[random_idx].copy()

        weights = np.full((self.n_clusters, n_attributes), 1.0 / n_attributes)

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # === Step 3 ===
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    # 【核心修正 2】Manhattan Distance
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights[j], axis=1)

                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # === Step 4 ===
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # === Step 5 ===
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])

                    for j in range(self.n_clusters):
                        mask_j = (Q == j)
                        n_j = np.sum(mask_j)
                        if n_j == 0: continue

                        # Intra (D)
                        intra_dist = diff_all[mask_j, j, :]
                        D_l = np.sum(intra_dist, axis=0) / n_j

                        # Inter (S)
                        dist_to_all = np.sum(diff_all[mask_j, :, :], axis=1)
                        dist_to_others = dist_to_all - intra_dist

                        denom_S = n_j * (self.n_clusters - 1)
                        if denom_S > 0:
                            S_l = np.sum(dist_to_others, axis=0) / denom_S
                        else:
                            S_l = np.zeros(n_attributes)

                        I_l = S_l / (D_l + epsilon)
                        weights[j] = I_l / np.sum(I_l)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights