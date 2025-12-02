"""
HARR修正版
"""
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import kmeans_plusplus


class HARR:
    def __init__(self, X, n_clusters=2, numerical_cols=None, nominal_cols=None, ordinal_cols=None):
        """
        :param X: 原始DataFrame数据
        :param n_clusters: 聚类簇数
        :param numerical_cols: 数值属性列名列表
        :param nominal_cols: 名义(Nominal)分类属性列名列表 (无序，如颜色)
        :param ordinal_cols: 序数(Ordinal)分类属性列名列表 (有序，如等级)
                             注意：对于Ordinal列，默认其在数据中的数值或字典序即为顺序，
                             如果需要指定顺序，建议先在外部将X中的值映射为有序整数。
        """
        self.X = X.copy()
        self.X_encoded = None
        self.n_clusters = n_clusters

        self.numerical_cols = list(numerical_cols) if numerical_cols is not None else []
        self.nominal_cols = list(nominal_cols) if nominal_cols is not None else []
        self.ordinal_cols = list(ordinal_cols) if ordinal_cols is not None else []

        # 所有的分类属性 = 名义 + 序数
        self.categorical_cols = self.nominal_cols + self.ordinal_cols

    def calculate_base_distance(self, target_col, context_df):
        """
        计算基础距离矩阵 (Base Distance Matrix / Kappa)
        论文 Section 3.2: distance between ... measured based on dataset statistics Eq.(5)(6)
        上下文 context_df 应该包含：其他分类属性 + 离散化后的数值属性
        """
        target_values = self.X[target_col].unique()
        probability_vector = {val: [] for val in target_values}

        # 上下文属性是 context_df 中的所有列（除了自己）
        context_cols = [c for c in context_df.columns if c != target_col]

        for ctx_col in context_cols:
            ctx_values = context_df[ctx_col].unique()

            for ctx_val in ctx_values:
                # 找出背景属性值为 ctx_val 的样本索引
                mask = (context_df[ctx_col] == ctx_val)
                # 注意：这里我们使用 mask 在 self.X 中筛选目标列，因为行索引是对齐的
                subset_target = self.X.loc[mask, target_col]
                subset_len = len(subset_target)

                if subset_len == 0:
                    for val in target_values:
                        probability_vector[val].append(0.0)
                    continue

                counts = subset_target.value_counts()

                for val in target_values:
                    # 计算 P(target_val | ctx_val)
                    prob = counts.get(val, 0) / subset_len
                    probability_vector[val].append(prob)

        # 构建概率分布矩阵 (Values x Features)
        profile_matrix = pd.DataFrame(probability_vector).T

        # 使用 CityBlock (Manhattan) 计算分布之间的差异，对应 Eq.(5) 的求和绝对值差异
        dist_array = pdist(profile_matrix.values, metric='cityblock')

        dist_matrix = pd.DataFrame(
            squareform(dist_array),
            index=profile_matrix.index,
            columns=profile_matrix.index
        )

        return dist_matrix

    @staticmethod
    def project_nominal(kappa):
        """
        名义属性投影：投影到 v(v-1)/2 个一维空间
        对应论文 Fig. 4 和 Eq.(7)-(8)
        """
        n_values = len(kappa.index)
        if n_values < 2:
            return np.zeros((n_values, 1))

        pairs = list(combinations(kappa.index, 2))
        projected_columns = []

        for dim_idx, (g, h) in enumerate(pairs):
            dist_g_h = kappa.loc[g, h]
            if dist_g_h == 0:
                coords = np.zeros(n_values)
            else:
                dist_t_g = kappa.loc[:, g]
                dist_t_h = kappa.loc[:, h]
                # 根据 Eq.(7) 或 (8) 的勾股定理推导
                # 论文 Eq.(7): phi(o_t, o_g; R^{r,b})
                numerator = np.square(dist_t_g) - np.square(dist_t_h) + np.square(dist_g_h)
                denominator = 2 * dist_g_h
                coords = numerator / denominator

                # 论文中取绝对值 |coords| 吗？
                # Eq.(11) 计算的是两个坐标差的绝对值作为距离。
                # 这里的 coords 是坐标点。
            projected_columns.append(coords)

        projected_matrix = np.array(projected_columns).T
        return projected_matrix

    @staticmethod
    def project_ordinal(kappa):
        """
        序数属性投影：只投影到 1 个一维空间
        对应论文 Section 3.2: "Thus, only one one-dimensional space... will be enough"
        及 Eq.(9)-(10).
        """
        values = kappa.index
        n_values = len(values)
        if n_values < 2:
            return np.zeros((n_values, 1))

        # 对于序数属性，我们假设 values 的顺序即为属性的内在顺序（或者需要用户预先排序）
        # 论文 Eq.(9) 指出 phi(o_t, o_g) = kappa(o_t, o_g)
        # 这意味着在 1D 空间中，两点间距离直接由 kappa 决定。
        # 我们可以取第一个值作为原点，后续值的坐标为它到原点的 Base Distance。
        # 这是一个简化处理，严格来说应使用 MDS(n=1)，但在论文“线性排列”假设下，直接取距离即可。

        ref_val = values[0]  # 选取第一个值作为参考点 (原点)
        coords = kappa.loc[:, ref_val].values  # 所有点到参考点的距离作为坐标

        return coords.reshape(-1, 1)

    def preprocess(self):
        # 1. 数值属性归一化
        X_num = pd.DataFrame()
        if self.numerical_cols:
            num_data = self.X[self.numerical_cols]
            # 避免分母为0
            range_val = num_data.max() - num_data.min()
            range_val[range_val == 0] = 1
            X_num = (num_data - num_data.min()) / range_val

        # 2. 准备上下文数据：分类属性 + 离散化后的数值属性 (论文 Fig. 3)
        # 论文提到: "we discrete numerical attributes and treat them as ordinal attributes"
        context_df = self.X[self.categorical_cols].copy()

        if self.numerical_cols:
            # 使用 K-Means 或等频离散化，通常选 5-10 个区间，这里设为 5
            est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            X_num_discrete = est.fit_transform(self.X[self.numerical_cols])
            X_num_discrete_df = pd.DataFrame(
                X_num_discrete,
                columns=[f"{c}_disc" for c in self.numerical_cols],
                index=self.X.index
            )
            # 将离散化后的数值属性加入上下文
            context_df = pd.concat([context_df, X_num_discrete_df], axis=1)

        encoding_frames = []

        # 3. 对每个分类属性进行投影表示
        for col in self.categorical_cols:
            # 计算 Base Distance Matrix (Kappa)
            # 上下文是 context_df 中的所有其他属性 (包含离散数值属性)
            col_kappa = self.calculate_base_distance(col, context_df)

            # 根据属性类型选择投影方式
            if col in self.ordinal_cols:
                # 序数属性：投影到 1 维
                col_vectors = self.project_ordinal(col_kappa)
            else:
                # 名义属性：投影到 v(v-1)/2 维
                col_vectors = self.project_nominal(col_kappa)

            # 映射回 DataFrame
            mapping_dict = {}
            for i, val in enumerate(col_kappa.index):
                mapping_dict[val] = col_vectors[i]

            mapping_series = self.X[col].map(mapping_dict)

            # 将得到的 list of arrays 展开
            expanded_data = np.stack(mapping_series.values)
            expanded_df = pd.DataFrame(
                expanded_data,
                index=self.X.index,
                columns=[f'{col}_{i}' for i in range(expanded_data.shape[1])]
            )

            # 论文 Section 3.2 提到: "divide all distances... by the maximum distance"
            # 为了使不同属性的距离可比 (归一化到 [0,1])。
            # 在 Manhattan 距离下，对坐标进行 Min-Max 归一化可以达到此效果。
            col_min = expanded_df.min()
            col_max = expanded_df.max()
            col_range = col_max - col_min
            col_range[col_range == 0] = 1
            expanded_df = (expanded_df - col_min) / col_range

            encoding_frames.append(expanded_df)

        # 4. 合并数值和分类表示
        if encoding_frames:
            X_cat_encoded = pd.concat(encoding_frames, axis=1)
            self.X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)
        else:
            self.X_encoded = X_num

        return self.X_encoded

    def fitV(self, max_iter=100, epsilon=1e-6):
        if self.X_encoded is None:
            raise ValueError("X_encoded is None. Please preprocess the data first.")

        n_samples, n_attributes = self.X_encoded.shape
        # 初始化中心
        # random_idx = random.sample(range(n_samples), self.n_clusters)
        # centroids = self.X_encoded.iloc[random_idx].values
        #
        X = self.X_encoded.values
        # M = centroids
        # 使用 k-means++ 初始化中心
        M, _ = kmeans_plusplus(self.X_encoded.values, self.n_clusters)

        # 初始化权重 (全局权重)
        weights = np.ones(n_attributes) / n_attributes

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)  # 用于判定 Step 3 收敛
        Q_double_prime = np.zeros(n_samples)  # 用于判定整体收敛

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # =======================
            # Step 3: 固定M、w，更新簇标签Q (Eq. 1)
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    # 加权 Manhattan 距离
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights, axis=1)

                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、w，更新簇中心M (Eq. 2)
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重w (HARR-V, Eq. 15-18)
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])
                    # labels_onehot: (n_samples, n_clusters)
                    labels_onehot = np.eye(self.n_clusters)[Q.astype(int)]

                    # Intra: 只算属于该簇的
                    # shape: (n_attributes,)
                    intra_numerator = np.sum(diff_all * labels_onehot[:, :, np.newaxis], axis=(0, 1))
                    D_r = intra_numerator / n_samples

                    # Inter: 只算不属于该簇的 (平均距离)
                    inter_numerator = np.sum(diff_all * (1 - labels_onehot[:, :, np.newaxis]), axis=(0, 1))
                    S_r = inter_numerator / (n_samples * (self.n_clusters - 1))

                    I_r = S_r / (D_r + epsilon)
                    weights = I_r / np.sum(I_r)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights

    def calculate_Z(self, Q, M, weights):
        """
        计算所有点到簇中心的加权距离 (Z)
        """
        n_samples, n_attributes = self.X_encoded.shape
        Z = np.zeros(n_samples)
        for j in range(self.n_clusters):
            diff = np.abs(self.X_encoded.values - M[j])
            Z[Q == j] = np.sum(diff[Q == j] * weights[j], axis=1)
        # 返回平均值
        Z_avg = np.mean(Z)

        return Z_avg

    def fitV_with_history(self, max_iter=100, epsilon=1e-6):
        Q_history = []
        W_history = []
        Z_history = []      # 所有点到簇中心距离

        if self.X_encoded is None:
            raise ValueError("X_encoded is None. Please preprocess the data first.")

        n_samples, n_attributes = self.X_encoded.shape
        X = self.X_encoded.values
        M, _ = kmeans_plusplus(self.X_encoded.values, self.n_clusters)

        weights = np.ones(n_attributes) / n_attributes
        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)  # 用于判定 Step 3 收敛
        Q_double_prime = np.zeros(n_samples)  # 用于判定整体收敛

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1
            Q_history.append(Q.copy())
            W_history.append(weights.copy())
            Z_history.append(self.calculate_Z(Q, M, weights))

            # =======================
            # Step 3: 固定M、w，更新簇标签Q (Eq. 1)
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    # 加权 Manhattan 距离
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights, axis=1)

                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、w，更新簇中心M (Eq. 2)
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重w (HARR-V, Eq. 15-18)
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])
                    # labels_onehot: (n_samples, n_clusters)
                    labels_onehot = np.eye(self.n_clusters)[Q.astype(int)]

                    # Intra: 只算属于该簇的
                    # shape: (n_attributes,)
                    intra_numerator = np.sum(diff_all * labels_onehot[:, :, np.newaxis], axis=(0, 1))
                    D_r = intra_numerator / n_samples

                    # Inter: 只算不属于该簇的 (平均距离)
                    inter_numerator = np.sum(diff_all * (1 - labels_onehot[:, :, np.newaxis]), axis=(0, 1))
                    S_r = inter_numerator / (n_samples * (self.n_clusters - 1))

                    I_r = S_r / (D_r + epsilon)
                    weights = I_r / np.sum(I_r)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights, Q_history, W_history, Z_history

    def fitM(self, max_iter=100, epsilon=1e-6):
        if self.X_encoded is None:
            raise ValueError("X_encoded is None. Please preprocess the data first.")

        n_samples, n_attributes = self.X_encoded.shape
        random_idx = random.sample(range(n_samples), self.n_clusters)
        centroids = self.X_encoded.iloc[random_idx].values

        X = self.X_encoded.values
        M = centroids
        # 初始化权重矩阵 (n_clusters, n_attributes)
        weights = np.full((self.n_clusters, n_attributes), 1 / n_attributes)

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # =======================
            # Step 3: 固定M、W，更新簇标签Q (Eq. 1)
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    diff = np.abs(X - M[j])
                    # 使用对应簇 j 的权重 weights[j]
                    dist_matrix[:, j] = np.sum(diff * weights[j], axis=1)
                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、W，更新簇中心M (Eq. 2)
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重矩阵 W (HARR-M, Eq. 19-22)
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    # diff_all shape: (n_samples, n_clusters, n_attributes)
                    # diff_all[i, j, k] = |x_i^k - m_j^k|
                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])

                    for j in range(self.n_clusters):
                        # -------------------------
                        # 修正点：D_l (Eq. 21)
                        # -------------------------
                        mask_member = (Q == j)
                        n_member = np.sum(mask_member)

                        if n_member == 0:
                            # 避免空簇除零，保持原权重或设为均匀
                            continue

                        # 计算簇内成员到中心 M[j] 的距离总和
                        # diff_all[mask_member, j, :] 形状 (n_members, n_attributes)
                        sum_intra = np.sum(diff_all[mask_member, j, :], axis=0)
                        D_l = sum_intra / n_member

                        # -------------------------
                        # 修正点：S_l (Eq. 22)
                        # 逻辑：非成员点到当前中心 M[j] 的平均距离
                        # -------------------------
                        mask_non_member = (Q != j)
                        n_non_member = np.sum(mask_non_member)

                        if n_non_member == 0:
                            # 极端情况：只有一个簇
                            S_l = np.zeros(n_attributes)
                        else:
                            # 关键修正：取 non_member 到 center j 的距离
                            sum_inter = np.sum(diff_all[mask_non_member, j, :], axis=0)
                            S_l = sum_inter / n_non_member

                        # Eq. 20
                        I_l = S_l / (D_l + epsilon)

                        # Eq. 19
                        if np.sum(I_l) == 0:
                            weights[j] = np.ones(n_attributes) / n_attributes
                        else:
                            weights[j] = I_l / np.sum(I_l)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights

    def fitM_with_history(self, max_iter=100, epsilon=1e-6):
        if self.X_encoded is None:
            raise ValueError("X_encoded is None. Please preprocess the data first.")

        n_samples, n_attributes = self.X_encoded.shape
        random_idx = random.sample(range(n_samples), self.n_clusters)
        centroids = self.X_encoded.iloc[random_idx].values

        X = self.X_encoded.values
        M = centroids
        # 初始化权重矩阵 (n_clusters, n_attributes)
        weights = np.full((self.n_clusters, n_attributes), 1 / n_attributes)

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        Q_history = []
        W_history = []
        Z_history = []

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1
            # 记录当前迭代的 Q、W、Z
            Q_history.append(Q.copy())
            W_history.append(weights.copy())
            Z_history.append(self.calculate_Z(Q, M, weights))

            # =======================
            # Step 3: 固定M、W，更新簇标签Q (Eq. 1)
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    diff = np.abs(X - M[j])
                    # 使用对应簇 j 的权重 weights[j]
                    dist_matrix[:, j] = np.sum(diff * weights[j], axis=1)
                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、W，更新簇中心M (Eq. 2)
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)
                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重矩阵 W (HARR-M, Eq. 19-22)
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    # diff_all shape: (n_samples, n_clusters, n_attributes)
                    # diff_all[i, j, k] = |x_i^k - m_j^k|
                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])

                    for j in range(self.n_clusters):
                        # -------------------------
                        # 修正点：D_l (Eq. 21)
                        # -------------------------
                        mask_member = (Q == j)
                        n_member = np.sum(mask_member)

                        if n_member == 0:
                            # 避免空簇除零，保持原权重或设为均匀
                            continue

                        # 计算簇内成员到中心 M[j] 的距离总和
                        # diff_all[mask_member, j, :] 形状 (n_members, n_attributes)
                        sum_intra = np.sum(diff_all[mask_member, j, :], axis=0)
                        D_l = sum_intra / n_member

                        # -------------------------
                        # 修正点：S_l (Eq. 22)
                        # 逻辑：非成员点到当前中心 M[j] 的平均距离
                        # -------------------------
                        mask_non_member = (Q != j)
                        n_non_member = np.sum(mask_non_member)

                        if n_non_member == 0:
                            # 极端情况：只有一个簇
                            S_l = np.zeros(n_attributes)
                        else:
                            # 关键修正：取 non_member 到 center j 的距离
                            sum_inter = np.sum(diff_all[mask_non_member, j, :], axis=0)
                            S_l = sum_inter / n_non_member

                        # Eq. 20
                        I_l = S_l / (D_l + epsilon)

                        # Eq. 19
                        if np.sum(I_l) == 0:
                            weights[j] = np.ones(n_attributes) / n_attributes
                        else:
                            weights[j] = I_l / np.sum(I_l)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights, Q_history, W_history, Z_history
