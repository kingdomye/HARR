import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, LabelEncoder
from tqdm import tqdm  # 进度条库

from HARRDataset import HARRDataSet
from HARR_v3 import HARR


def get_ohe_representation(df, cols_info):
    """提取 OHE 表示"""
    cat_cols = cols_info.categorical_columns
    X_cat = df[cat_cols].astype(str)
    enc = OneHotEncoder(sparse_output=False)
    X_encoded = enc.fit_transform(X_cat)
    return X_encoded, 'euclidean'  # OHE 通常配合欧氏距离或 simple matching


def get_gbd_distance_matrix(df, cols_info):
    """
    复用 Wrapper_GBD 的逻辑，提取谱嵌入 (Spectral Embedding) 或 相似度矩阵
    这里我们提取 谱嵌入后的特征，这是 GBD 的核心表示。
    """
    num_cols = list(cols_info.numerical_columns)
    cat_cols = list(cols_info.categorical_columns)

    n_samples = df.shape[0]

    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(df[num_cols])
        affinity_num = rbf_kernel(X_num, gamma=1.0)
    else:
        affinity_num = np.ones((n_samples, n_samples))

    if len(cat_cols) > 0:
        enc = OrdinalEncoder()
        X_cat = enc.fit_transform(df[cat_cols].astype(str))
        dist_cat_norm = pairwise_distances(X_cat, metric='hamming')
        affinity_cat = np.exp(-dist_cat_norm)
    else:
        affinity_cat = np.ones((n_samples, n_samples))

    final_affinity = affinity_num * affinity_cat

    sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=10)
    from sklearn.manifold import SpectralEmbedding
    embedder = SpectralEmbedding(n_components=10, affinity='precomputed')
    X_embedded = embedder.fit_transform(final_affinity)

    return X_embedded, 'euclidean'


def get_fbd_distance_matrix(df, cols_info):
    """
    提取 FBD 距离矩阵 (修复版)
    """
    # 1. 获取特征列名列表
    num_cols = list(cols_info.numerical_columns)
    cat_cols = list(cols_info.categorical_columns)

    feature_cols = num_cols + cat_cols
    data = df[feature_cols].copy()

    # 2. 数值预处理
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

    # 3. 分类预处理 (One-Hot)
    if len(cat_cols) > 0:
        X_input = pd.get_dummies(data, columns=cat_cols, dtype=int)
    else:
        X_input = data

    X_input.columns = X_input.columns.astype(str)

    # 4. 随机树嵌入
    hasher = RandomTreesEmbedding(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    X_transformed = hasher.fit_transform(X_input)

    # 5. 计算 Cosine 距离
    dist_matrix = pairwise_distances(X_transformed, metric='cosine')

    return dist_matrix, 'precomputed'


def get_harr_representation(df, cols_info, method='V'):
    """
    运行 HARR，获取加权后的特征表示
    """
    num_cols = list(cols_info.numerical_columns)
    nom_cols = list(cols_info.categorical_columns)

    harr_model = HARR(
        df,
        n_clusters=2,  # MR 数据集通常 k=2
        numerical_cols=num_cols,
        nominal_cols=nom_cols,
        ordinal_cols=[]
    )

    # 1. 预处理 (属性重构)
    harr_model.preprocess()

    # 2. 训练权重
    if method == 'V':
        Q, weights = harr_model.fitV(max_iter=15)
        X_encoded_vals = harr_model.X_encoded.values
        X_weighted = X_encoded_vals * weights

    elif method == 'M':
        Q, weights = harr_model.fitM(max_iter=15)
        X_encoded_vals = harr_model.X_encoded.values
        X_weighted = np.zeros_like(X_encoded_vals)

        for i in range(len(X_encoded_vals)):
            cluster_idx = int(Q[i])
            X_weighted[i, :] = X_encoded_vals[i, :] * weights[cluster_idx, :]

    # HARR 使用 Manhattan 距离
    return X_weighted, 'manhattan'


# ==========================================
# 2. 主绘图逻辑
# ==========================================

def plot_fig11_complete():
    # 1. 加载数据
    print("正在加载 MR 数据集...")
    dataset = HARRDataSet()
    df, cols_info = dataset.get_MR()

    # 获取真实标签用于上色
    y_true = cols_info.y_true
    # 编码标签 (e->0, p->1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)
    y_labels = ['Edible' if str(x).lower().startswith('e') else 'Poisonous' for x in y_true]

    # 定义要运行的方法
    methods = [
        ('a', 'OHE', get_ohe_representation),
        ('b', 'GBD', get_gbd_distance_matrix),
        ('c', 'FBD', get_fbd_distance_matrix),
        ('d', 'HARR-V', lambda d, c: get_harr_representation(d, c, 'V')),
        ('e', 'HARR-M', lambda d, c: get_harr_representation(d, c, 'M'))
    ]

    # 准备画布 (2行3列，最后一张图位置留空或用于放置大图例)
    # Fig 11 是 2x3 布局，但只有 5 张图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 隐藏第6个子图 (axes[5])
    axes[5].axis('off')

    # 配置进度条
    pbar = tqdm(methods, desc="Processing Methods", unit="method")

    for idx, (fig_idx, name, func) in enumerate(pbar):
        pbar.set_description(f"Running {name}")

        # 1. 获取数据的特征表示或距离矩阵
        #    X_data: 可能是特征矩阵，也可能是距离矩阵
        #    metric: 告诉 t-SNE 如何处理 ('precomputed' or 'euclidean' etc.)
        X_data, metric = func(df, cols_info)

        # 2. 运行 t-SNE
        #    对于距离矩阵 (FBD)，必须用 precomputed
        #    对于特征矩阵 (OHE, HARR, GBD-Spectral)，用对应 metric
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init='random' if metric == 'precomputed' else 'pca',  # precomputed 不支持 pca init
            learning_rate='auto',
            metric=metric,
            random_state=42,
            max_iter=1000
        )

        X_embedded = tsne.fit_transform(X_data)

        # 3. 绘图
        ax = axes[idx]

        # 构造绘图数据
        plot_df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            'label': y_labels
        })

        sns.scatterplot(
            data=plot_df, x='x', y='y', hue='label',
            palette={'Edible': '#1f77b4', 'Poisonous': '#ff7f0e'},
            s=15, alpha=0.6, edgecolor=None, ax=ax, legend=(idx == 0)  # 只在第一个图显示图例，避免混乱
        )

        # 设置标题
        ax.set_title(f"({fig_idx}) {name} @ MR", y=-0.15, fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # 移除刻度数字，保持干净

    # 统一图例：从第一个子图提取图例句柄，然后放到显眼位置（例如第6个空位）
    handles, labels_txt = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()  # 移除子图内部图例

    # 在第6个空图位置放一个大图例
    axes[5].legend(handles, labels_txt, title="True Labels", loc='center', fontsize=12, title_fontsize=14)

    plt.suptitle("Figure 11 Reproduction: t-SNE Visualization on MR Dataset", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_fig11_complete()
