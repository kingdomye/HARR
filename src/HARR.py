import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from HARRDataset import HARRDataSet


class HARR:
    def __init__(self, X, n_clusters=2):
        self.X = X
        self.X_encoded = None
        self.n_clusters = n_clusters

    def calculate_base_distance(self, target_col, allowed_context_cols):
        target_values = self.X[target_col].unique()  # 目标列的唯一值
        probability_vector = {val: [] for val in target_values}  # 目标列的唯一值的概率向量
        context_cols = [c for c in allowed_context_cols if c != target_col]  # 上下文列(背景列)

        for ctx_col in context_cols:
            ctx_values = self.X[ctx_col].unique()  # 背景列的属性值
            ctx_prob_vector = {val: [] for val in ctx_values}  # 背景列的属性值的概率向量

            for ctx_val in ctx_values:  # 遍历背景列的属性值
                context_mask = (self.X[ctx_col] == ctx_val)
                subset = self.X[context_mask]  # 背景列属性值为ctx_val的样本
                subset_len = len(subset)  # 背景列属性值为ctx_val的样本数量

                if subset_len == 0:
                    for val in target_values:
                        probability_vector[val].append(0.0)
                    continue

                counts = subset[target_col].value_counts()  # 背景列属性值为ctx_val的样本中，目标列的唯一值的数量

                for val in target_values:
                    count = counts.get(val, 0)
                    prob = count / subset_len
                    probability_vector[val].append(prob)
                    ctx_prob_vector[ctx_val].append(prob)

        profile_matrix = pd.DataFrame(probability_vector).T

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
                numerator = np.square(dist_t_g) - np.square(dist_t_h) + np.square(dist_g_h)
                denominator = 2 * dist_g_h
                coords = numerator / denominator

            projected_columns.append(coords)

        projected_matrix = np.array(projected_columns).T
        return projected_matrix

    def preprocess(self, numerical_cols, categorical_cols):
        # 对数值列归一化，一行代码
        self.X[numerical_cols] = (self.X[numerical_cols] - self.X[numerical_cols].min()) / (self.X[numerical_cols].max() - self.X[numerical_cols].min())
        X_num = self.X[numerical_cols].copy()

        encoding_frames = []

        for col in categorical_cols:
            col_kappa = self.calculate_base_distance(col, categorical_cols)
            col_vectors = self.calculate_cast_distance_vector(col_kappa)

            mapping_dict = {}
            for i, val in enumerate(col_kappa.index):
                mapping_dict[val] = col_vectors[i]

            mapping_series = self.X[col].map(mapping_dict)
            expanded_df = pd.DataFrame(
                mapping_series.tolist(),
                index=self.X.index
            )
            n_dims = expanded_df.shape[1]
            expanded_df.columns = [f'{col}_{i}' for i in range(n_dims)]
            encoding_frames.append(expanded_df)

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
        random_idx = random.sample(range(n_samples), self.n_clusters)

        centroids = self.X_encoded.iloc[random_idx]
        X = self.X_encoded.values
        M = centroids.values
        weights = np.ones(n_attributes) / n_attributes

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # =======================
            # Step 3: 固定M、w，更新簇标签Q
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights, axis=1)
                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、w，更新簇中心M
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)

                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重w
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])
                    labels_onehot = np.eye(self.n_clusters)[Q.astype(int)]

                    intra_sum = np.sum(diff_all * labels_onehot[:, :, np.newaxis], axis=(0, 1))
                    D_r = intra_sum / n_samples

                    intra_sum = np.sum(diff_all * (1 - labels_onehot[:, :, np.newaxis]), axis=(0, 1))
                    S_r = intra_sum / (n_samples * (self.n_clusters - 1))

                    I_r = S_r / (D_r + epsilon)
                    weights = I_r / np.sum(I_r)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights

    def fitM(self, max_iter=100, epsilon=1e-6):
        if self.X_encoded is None:
            raise ValueError("X_encoded is None. Please preprocess the data first.")

        n_samples, n_attributes = self.X_encoded.shape
        random_idx = random.sample(range(n_samples), self.n_clusters)

        centroids = self.X_encoded.iloc[random_idx]
        X = self.X_encoded.values
        M = centroids.values
        weights = np.full((self.n_clusters, n_attributes), 1 / n_attributes)

        Q = np.zeros(n_samples)
        Q_prime = np.zeros(n_samples)
        Q_double_prime = np.zeros(n_samples)

        cur_step = 3
        iteration_counter = 0

        while cur_step != -1 and iteration_counter < max_iter:
            iteration_counter += 1

            # =======================
            # Step 3: 固定M、w，更新簇标签Q
            # =======================
            if cur_step == 3:
                dist_matrix = np.zeros((n_samples, self.n_clusters))
                for j in range(self.n_clusters):
                    diff = np.abs(X - M[j])
                    dist_matrix[:, j] = np.sum(diff * weights[j], axis=1)
                Q = np.argmin(dist_matrix, axis=1)

                if not np.array_equal(Q, Q_prime):
                    Q_prime = Q.copy()
                    cur_step = 4
                else:
                    cur_step = 5

            # =======================
            # Step 4：固定Q、w，更新簇中心M
            # =======================
            elif cur_step == 4:
                for j in range(self.n_clusters):
                    points = X[Q == j]
                    if len(points) > 0:
                        M[j] = np.mean(points, axis=0)

                cur_step = 3

            # =======================
            # Step 5: 固定M、Q，计算权重w
            # =======================
            elif cur_step == 5:
                if not np.array_equal(Q, Q_double_prime):
                    Q_double_prime = Q.copy()

                    diff_all = np.abs(X[:, np.newaxis, :] - M[np.newaxis, :, :])
                    for j in range(self.n_clusters):
                        mask_j = (Q == j)
                        n_j = np.sum(mask_j)
                        if n_j == 0: continue

                        intra_dist = diff_all[mask_j, j, :]
                        D_l = np.sum(intra_dist, axis=0) / n_j

                        dist_to_all_centers = diff_all[mask_j, :, :]
                        sum_dist_to_all = np.sum(dist_to_all_centers, axis=1)
                        sum_dist_to_others = sum_dist_to_all - intra_dist

                        if self.n_clusters > 1:
                            S_l = np.sum(sum_dist_to_others, axis=0) / (n_j * (self.n_clusters - 1))
                        else:
                            S_l = np.zeros(n_attributes)

                        I_l = S_l / (D_l + epsilon)
                        weights[j] = I_l / np.sum(I_l)

                    cur_step = 3
                else:
                    cur_step = -1

        return Q, weights


if __name__ == '__main__':
    harr = HARRDataSet()
    all_data = harr.get_all_data_1()

    ARI_scores = {}
    for i in range(2):
        for dataset_name, (dataset, columns) in all_data.items():
            harrv = HARR(dataset)
            encoded_ds = harrv.preprocess(columns.numerical_columns, columns.categorical_columns)
            Q, w = harrv.fitM()

            # 计算两种ARI
            harrv_ari = adjusted_rand_score(columns.y_true, Q)
            ARI_scores[dataset_name] = ARI_scores.get(dataset_name, []) + [harrv_ari]

            print(f'{dataset_name} HARR_V ARI: {harrv_ari}')

    print(ARI_scores)
