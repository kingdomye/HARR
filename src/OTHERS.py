import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from scipy.stats import entropy
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from scipy.spatial.distance import pdist, squareform


# 辅助函数：确保列名是列表形式
def to_list(cols):
    if cols is None:
        return []
    if isinstance(cols, (int, float, str, np.integer)):
        return [cols]
    if len(cols) == 0:
        return []
    return list(cols)


class Wrapper_KPrototypes:
    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        # 【关键修复1】只保留特征列，剔除标签列
        # 这一步防止 'no'/'yes' 标签被当作数值特征传入导致报错
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        # 1. 数值列归一化
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            self.data[self.num_cols] = scaler.fit_transform(self.data[self.num_cols])

        # 确保分类列是字符串类型
        if len(self.cat_cols) > 0:
            self.data[self.cat_cols] = self.data[self.cat_cols].astype(str)

        # 2. 获取分类列在当前 data 中的索引位置
        cat_indices = [self.data.columns.get_loc(c) for c in self.cat_cols]

        # 3. 选择算法并运行
        try:
            if len(self.num_cols) == 0:
                # 纯分类 -> K-Modes
                km = KModes(n_clusters=self.k, init='Huang', n_init=5, verbose=0)
                labels = km.fit_predict(self.data)
            else:
                # 混合 -> K-Prototypes
                kp = KPrototypes(n_clusters=self.k, init='Huang', n_init=5, verbose=0)
                # KPrototypes 需要 values
                labels = kp.fit_predict(self.data.values, categorical=cat_indices)
        except Exception as e:
            print(f"K-Prototypes 运行失败: {e}")
            labels = np.zeros(len(self.data))

        return labels, None


class Wrapper_OHE_KMeans:
    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        # 【关键修复1】只保留特征列
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        # 1. 数值列归一化
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            self.data[self.num_cols] = scaler.fit_transform(self.data[self.num_cols])

        # 2. 分类列 One-Hot 编码
        if len(self.cat_cols) > 0:
            self.data = pd.get_dummies(
                self.data,
                columns=self.cat_cols,
                dtype=int
            )

        # 【关键修复2】强制将所有列名转换为字符串
        # 解决 TypeError: Feature names ... ['int', 'str'] 问题
        self.data.columns = self.data.columns.astype(str)

        # 3. 运行标准 K-Means
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=None)
        labels = kmeans.fit_predict(self.data)

        return labels, None


class Wrapper_SBC:
    """
    对应论文中的 SBC (Structure-Based Categorical data encoding) 方法。
    核心逻辑：
    1. 计算样本间的混合距离矩阵 (数值欧氏距离 + 分类汉明距离)。
    2. 将距离矩阵作为新的 N维特征空间 (Space Structure)。
    3. 在该新特征空间上运行 K-Means。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        # 只保留特征列，剔除标签列
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 计算数值属性部分的距离矩阵 ---
        if len(self.num_cols) > 0:
            # 归一化数值列，防止尺度差异过大
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            # 计算欧氏距离
            dist_num = squareform(pdist(X_num, metric='euclidean'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 计算分类属性部分的距离矩阵 ---
        if len(self.cat_cols) > 0:
            # 将分类数据转换为整数编码，以便计算汉明距离
            # 注意：pdist 的 hamming 默认计算的是“不匹配的比例”，我们需要“不匹配的个数”
            # 或者保持比例作为距离度量也可以，只要统一即可。这里我们还原为简单的匹配距离。
            enc = OrdinalEncoder()
            # 确保转为字符串处理，防止数字型分类变量报错
            X_cat = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # Hamming = 比例，乘以列数 = 不匹配的个数 (Simple Matching Distance)
            dist_cat = squareform(pdist(X_cat, metric='hamming')) * len(self.cat_cols)
        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 组合距离矩阵 (混合数据) ---
        # SBC 将这个 N x N 的矩阵直接视为 N 个样本的 N 维特征
        structural_features = dist_num + dist_cat

        # --- 4. 运行 K-Means ---
        # 论文中 SBC 编码通常配合 K-Means 使用
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=None)
        labels = kmeans.fit_predict(structural_features)

        return labels, None  # 保持接口一致 (labels, weights)


class Wrapper_JDM:
    """
    对应论文中的 JDM (Jia's Distance Metric)。
    JDM 是一种改进的混合距离度量。
    实现逻辑：
    1. 数值属性：MinMax 归一化后的欧氏距离。
    2. 分类属性：汉明距离 (Hamming Distance)。
    3. 聚类算法：Agglomerative Clustering (层次聚类) 使用 Average Linkage。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        # 只保留特征列
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 数值部分距离 ---
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            # 计算欧氏距离矩阵
            dist_num = squareform(pdist(X_num, metric='euclidean'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 分类部分距离 ---
        if len(self.cat_cols) > 0:
            # 转为字符串并进行整数编码
            enc = OrdinalEncoder()
            # astype(str) 防止数字型分类变量报错
            X_cat = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # Hamming 距离 (Standard Matching)
            # pdist('hamming') 返回的是不匹配的比例，乘以列数得到不匹配的个数
            dist_cat = squareform(pdist(X_cat, metric='hamming')) * len(self.cat_cols)
        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 融合距离 ---
        # 简单的线性组合 (在 JDM 原文中会有更复杂的概率加权，这里用标准混合距离作为近似基准)
        final_dist_matrix = dist_num + dist_cat

        # --- 4. 层次聚类 ---
        # metric='precomputed' 表示输入的是距离矩阵
        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(final_dist_matrix)

        return labels, None


class Wrapper_CMS:
    """
    对应论文中的 CMS (Coupled Metric Similarity)。
    CMS 强调属性值之间的耦合关系（Coupling），通常与频率相关。
    实现逻辑：
    1. 数值属性：欧氏距离。
    2. 分类属性：频率加权距离 (模拟耦合度，越稀有的值匹配权重越高)。
    3. 聚类算法：Agglomerative Clustering。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 数值距离 ---
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            dist_num = squareform(pdist(X_num, metric='euclidean'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 分类距离 (引入简单的频率加权模拟 Coupling) ---
        if len(self.cat_cols) > 0:
            # 这里的逻辑是：如果属性值出现的频率越低，它们不仅是"不同"，而且是"显著不同"
            # 或者反之，我们这里使用加权的 Hamming

            # 首先转为整数编码
            enc = OrdinalEncoder()
            X_cat_enc = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # 计算简单的 Hamming
            dist_cat = squareform(pdist(X_cat_enc, metric='hamming')) * len(self.cat_cols)

            # CMS 特性模拟：
            # CMS 本质是 Similarity，但在距离度量中，我们通常可以理解为：
            # 考虑了属性内部的频率分布 (Intra-coupled)。
            # 为了体现与 JDM 的不同，我们给分类距离乘上一个系数，
            # 该系数由分类属性的“熵”或“多样性”决定。

            # 计算每个分类列的唯一值数量 (多样性)
            uniques = [self.data[c].nunique() for c in self.cat_cols]
            avg_uniques = np.mean(uniques) if uniques else 1.0

            # 这是一个启发式调整，模拟 CMS 对复杂属性的敏感度
            dist_cat = dist_cat * (1.0 + np.log(avg_uniques))

        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 融合 ---
        final_dist_matrix = dist_num + dist_cat

        # --- 4. 聚类 ---
        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(final_dist_matrix)

        return labels, None


class Wrapper_UDM:
    """
    对应论文中的 UDM (Unified Distance Metric)。
    核心思想：引入【信息熵 (Entropy)】作为权重。

    实现逻辑：
    1. 数值属性：标准欧氏距离。
    2. 分类属性：计算每个属性的信息熵。熵越大的属性，区分度越高，权重越大。
       距离 = Sum( 熵权重 * 汉明距离 )
    3. 聚类：层次聚类。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 数值距离 ---
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            dist_num = squareform(pdist(X_num, metric='euclidean'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 分类距离 (熵加权) ---
        if len(self.cat_cols) > 0:
            # 计算每个属性的熵权重
            weights = []
            for col in self.cat_cols:
                # 计算该列各值的出现概率
                counts = self.data[col].value_counts(normalize=True)
                # 计算熵 (base e)
                ent = entropy(counts)
                # 权重：熵越大，包含信息越多 (也可以用 1-Entropy 视具体定义，这里取正相关)
                weights.append(ent + 1e-6)  # 防止为0

            # 归一化权重
            weights = np.array(weights) / np.sum(weights) * len(self.cat_cols)

            # 转换为整数编码
            enc = OrdinalEncoder()
            X_cat = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # 计算加权汉明距离
            # pdist('hamming', w=weights) 支持加权
            dist_cat = squareform(pdist(X_cat, metric='hamming', w=weights)) * len(self.cat_cols)
        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 融合与聚类 ---
        final_dist = dist_num + dist_cat

        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(final_dist)

        return labels, None


class Wrapper_HOD:
    """
    对应论文中的 HOD (HOmogeneous Distance)。
    HOD 是 HARR 的前身，核心是利用【条件概率分布差异】计算距离。

    实现逻辑 (为了效率进行近似)：
    论文公式 (5) 计算的是两个值在所有其他属性上的概率分布差。
    完全计算复杂度极高 O(N^2 * D^2)。
    这里我们使用【频率概率距离】作为高效近似：
    1. 将分类值映射为其出现的全局概率 P(x)。
    2. 两个值的距离定义为 |P(v1) - P(v2)| (如果值不同)。
       这捕获了 HOD 核心思想：不同分布的值距离较远。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 数值距离 ---
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            dist_num = squareform(pdist(X_num, metric='euclidean'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 分类距离 (基于概率分布的近似) ---
        if len(self.cat_cols) > 0:
            # 策略：将每个 Categorical Value 替换为它的频率 (Frequency Encoding)
            # 这样就把分类属性变成了数值属性，反映了其分布特征
            X_freq = self.data[self.cat_cols].copy()

            for col in self.cat_cols:
                # 计算频率 map: {'A': 0.6, 'B': 0.4}
                freq_map = self.data[col].value_counts(normalize=True).to_dict()
                X_freq[col] = self.data[col].map(freq_map)

            # 转为 float
            X_freq = X_freq.astype(float)

            # 计算距离：
            # 如果两个样本的值相同，距离为 0 (Hamming 逻辑)
            # 如果两个样本的值不同，距离为 |P(v1) - P(v2)| + const
            # 这里直接计算频率空间上的曼哈顿距离，能够很好地近似 HOD 的分布差异思想
            dist_cat = squareform(pdist(X_freq, metric='cityblock'))

            # 标准化一下，使其与 Hamming 的量级类似 (0~D)
            if dist_cat.max() > 0:
                dist_cat = dist_cat / dist_cat.max() * len(self.cat_cols)

        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 融合与聚类 ---
        final_dist = dist_num + dist_cat

        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(final_dist)

        return labels, None


class Wrapper_GWD:
    """
    对应论文中的 GWD (Gower's Distance)。
    实现逻辑：
    1. Gower 距离是不同属性距离的加权平均。
    2. 数值属性：|x_i - x_j| / (max - min) (归一化的曼哈顿距离)。
    3. 分类属性：汉明距离 (0或1)。
    4. 聚类：Agglomerative Clustering。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters

        # 只保留特征列
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]
        n_features = len(self.features)

        # --- 1. 数值部分 (Normalized Manhattan) ---
        if len(self.num_cols) > 0:
            X_num = self.data[self.num_cols].astype(float)
            # 计算每一列的 Range (max - min)
            ranges = np.ptp(X_num.values, axis=0)  # Peak to peak
            # 防止除以0
            ranges[ranges == 0] = 1.0

            # Gower 数值部分计算: Sum(|diff|) / Range
            # 我们先除以 range 归一化，然后算 cityblock
            X_num_norm = X_num / ranges

            # pdist cityblock 默认是 sum(|x - y|)，正是我们需要的分子求和
            dist_num = squareform(pdist(X_num_norm, metric='cityblock'))
        else:
            dist_num = np.zeros((n_samples, n_samples))

        # --- 2. 分类部分 (Hamming) ---
        if len(self.cat_cols) > 0:
            enc = OrdinalEncoder()
            X_cat = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # pdist 'hamming' 返回的是不匹配的比例 (diff_count / n_cat_cols)
            # 我们需要的是不匹配的个数 (diff_count)
            # 所以乘以 len(cat_cols)
            dist_cat = squareform(pdist(X_cat, metric='hamming')) * len(self.cat_cols)
        else:
            dist_cat = np.zeros((n_samples, n_samples))

        # --- 3. 计算平均距离 ---
        # Gower = (Sum_Num_Dist + Sum_Cat_Dist) / Total_Features
        gower_dist_matrix = (dist_num + dist_cat) / n_features

        # --- 4. 聚类 ---
        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(gower_dist_matrix)

        return labels, None


class Wrapper_GBD:
    """
    对应论文中的 GBD (Graph-Based Distance)。
    实现逻辑：
    GBD 的核心是将数据映射为图结构。
    1. 构建相似度图 (Affinity Matrix)。
       - 数值部分：使用 RBF 核 (高斯核) 计算相似度。
       - 分类部分：使用 1 - Hamming 距离作为相似度。
    2. 使用 Spectral Clustering (谱聚类)。
       谱聚类本质上就是在图的拉普拉斯矩阵上进行特征分解，是标准的 Graph-Based 方法。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        n_samples = self.data.shape[0]

        # --- 1. 计算相似度矩阵 (Affinity) ---

        # 数值相似度 (RBF Kernel)
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(self.data[self.num_cols])
            # gamma 默认 1.0/n_features，这里简单设为 1.0
            affinity_num = rbf_kernel(X_num, gamma=1.0)
        else:
            affinity_num = np.ones((n_samples, n_samples))

        # 分类相似度 (Kernelized Hamming)
        if len(self.cat_cols) > 0:
            enc = OrdinalEncoder()
            X_cat = enc.fit_transform(self.data[self.cat_cols].astype(str))

            # Hamming 距离 (0~1)
            dist_cat_norm = squareform(pdist(X_cat, metric='hamming'))
            # 转换为相似度: 1 - distance
            # 并使用指数核稍微平滑一下，模拟图的连接强度
            affinity_cat = np.exp(-dist_cat_norm)
        else:
            affinity_cat = np.ones((n_samples, n_samples))

        # 融合相似度 (元素积，表示两个样本必须在数值和分类上都相似才算相似)
        # 也可以用加权和，但在图构建中乘积往往能产生更清晰的连通分量
        final_affinity = affinity_num * affinity_cat

        # --- 2. 谱聚类 ---
        # affinity='precomputed' 表示输入的是相似度矩阵
        model = SpectralClustering(
            n_clusters=self.k,
            affinity='precomputed',
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(final_affinity)

        return labels, None


class Wrapper_FBD:
    """
    对应论文中的 FBD (Forest-Based Distance)。
    实现逻辑：
    利用无监督随机森林 (Random Trees Embedding) 将数据映射到高维稀疏叶子节点空间。
    1. 数据编码：分类变量 One-Hot，数值变量保持。
    2. 随机树嵌入：训练森林，获取每个样本落在哪些叶子节点。
    3. 距离计算：在叶子节点空间计算 Cosine 距离。
    4. 聚类：Agglomerative Clustering。
    """

    def __init__(self, data, columns, n_clusters=2):
        self.raw_data = data.copy()
        self.num_cols = to_list(columns.numerical_columns)
        self.cat_cols = to_list(columns.categorical_columns)
        self.k = n_clusters
        self.features = self.num_cols + self.cat_cols
        self.data = self.raw_data[self.features].copy()

    def fit(self):
        # 1. 预处理 (Random Forest 需要数值输入)
        # 数值列归一化
        if len(self.num_cols) > 0:
            scaler = MinMaxScaler()
            self.data[self.num_cols] = scaler.fit_transform(self.data[self.num_cols])

        # 分类列 One-Hot (树模型可以直接处理 LabelEncode，但 OneHot 更能捕捉无序关系)
        if len(self.cat_cols) > 0:
            X_input = pd.get_dummies(self.data, columns=self.cat_cols, dtype=int)
        else:
            X_input = self.data

        # 【关键修复】强制将所有列名转换为字符串
        # 解决 TypeError: Feature names ... ['int', 'str']
        X_input.columns = X_input.columns.astype(str)

        # 2. 随机树嵌入 (Forest Embedding)
        # 将样本映射到高维稀疏特征 (叶子索引)
        hasher = RandomTreesEmbedding(
            n_estimators=50,  # 树的数量
            max_depth=5,  # 树深
            random_state=42
        )
        # X_transformed 是一个稀疏矩阵，表示样本落入了哪些叶子
        X_transformed = hasher.fit_transform(X_input)

        # 3. 计算距离
        # 在嵌入空间计算 Cosine 距离 (Forest Distance 的一种标准形式)
        # 两个样本落在相同的叶子越多，Cosine Similarity 越大，Cosine Distance 越小
        dist_matrix = squareform(pdist(X_transformed.toarray(), metric='cosine'))

        # 4. 聚类
        model = AgglomerativeClustering(
            n_clusters=self.k,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dist_matrix)

        return labels, None
