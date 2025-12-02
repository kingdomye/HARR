import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering, KMeans
import warnings
import time

# 忽略特定警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入模块
# ==========================================
try:
    from HARR_v3 import HARR
except ImportError:
    print("错误：找不到 HARR_v3.py。")
    exit()

from HARRDataset import HARRDataSet
from src.OTHERS import Wrapper_KPrototypes

# ==========================================
# 2. 配置项
# ==========================================
# Table 7 包含所有 14 个数据集
DATASET_CONFIG = {
    # Mixed
    'DS': 2, 'HF': 2, 'AA': 2, 'AP': 2, 'DT': 6, 'AC': 2,
    # Categorical
    'SB': 15, 'SF': 6, 'T3': 2, 'HR': 3, 'LG': 4, 'MR': 2, 'LE': 5, 'SW': 4
}
TABLE7_DATASET_NAMES = list(DATASET_CONFIG.keys())

# Table 7 的消融实验变体列表 (增加了 BD)
VARIANTS = ['KMD/KPT', 'BD', 'HAR', 'HARR-V', 'HARR-M']


# ==========================================
# 3. 辅助类：Wrapper_BD (Base Distance)
# ==========================================
class Wrapper_BD:
    """
    对应 Table 7 中的 BD 列。
    仅使用 HARR 中的 calculate_base_distance 计算出的概率距离矩阵，
    不做投影，直接进行聚类 (通常配合层次聚类)。
    """

    def __init__(self, dataset, columns, n_clusters):
        self.dataset = dataset
        self.columns = columns
        self.k = n_clusters
        # 复用 HARR 类中的 calculate_base_distance 方法
        # 这里只是借用工具，不进行 fit
        self.harr_tool = HARR(dataset, n_clusters,
                              columns.numerical_columns,
                              columns.nominal_columns,
                              columns.ordinal_columns)

    def fit(self):
        # 1. 准备上下文 (Categorical + Discretized Numerical)
        context_df = self.harr_tool._prepare_context()

        n_samples = len(self.dataset)

        # 2. 计算所有样本两两之间的距离
        # 论文没有详细说 BD 怎么聚合所有属性的距离，
        # 通常做法是：Sum of Base Distances for all attributes

        total_dist_matrix = np.zeros((n_samples, n_samples))

        # 遍历所有分类属性
        for col in self.harr_tool.categorical_cols:
            # 计算该属性下值的距离矩阵 (Value-Value Matrix)
            val_dist_mat = self.harr_tool.calculate_base_distance(col, context_df)

            # 将 Value-Level 距离映射到 Object-Level
            # 获取每个样本在该列的值
            col_values = self.dataset[col].values

            # 这是一个高效映射：
            # 先构建映射字典 {val: index_in_dist_mat}
            val_to_idx = {v: i for i, v in enumerate(val_dist_mat.index)}

            # 将样本值转为索引
            indices = [val_to_idx.get(v, -1) for v in col_values]

            # 从距离矩阵中查表
            # valid_indices
            # 这里简化处理，直接双重循环太慢，使用 numpy 索引
            # dist_obj = val_dist_mat.values[np.ix_(indices, indices)]
            # 但 indices 可能有 -1 (未知值)，需处理

            # 更简单的做法：
            # Base Distance 定义的是值的距离。
            # Object Distance = Sum( dist(x_i^a, x_j^a) )

            # 这里为了效率，我们假设 calculate_base_distance 正确返回了 DataFrame
            # 我们将其转为 dict lookup
            v_dist_dict = val_dist_mat.to_dict()  # {val1: {val2: dist, ...}, ...}

            # 使用列表推导式构建矩阵 (较慢，但逻辑正确)
            # 或者使用 sklearn 的 pairwise_distances 自定义 metric
            # 这里为了简单，我们用一个近似：
            # 将每个样本映射到其属性值的 One-Hot，但这丢失了概率距离信息。

            # 正确做法：利用 MDS 将概率距离矩阵映射到低维欧氏空间，然后算欧氏距离
            # 这样就变成了 Object-Object Distance
            # 这其实有点像 HAR 的前身。

            # 这里我们采用一种替代方案：
            # 既然 BD 只是为了证明概率距离有效，我们计算样本间在每个属性上的 Base Distance 之和。

            # 优化：预计算 map
            v_mat = val_dist_mat.values
            idx_map = {v: k for k, v in enumerate(val_dist_mat.index)}

            # 样本值索引序列
            sample_indices = [idx_map[v] for v in col_values]

            # 利用 numpy 高级索引一次性提取 N*N 矩阵
            obj_dists = v_mat[np.ix_(sample_indices, sample_indices)]
            total_dist_matrix += obj_dists

        # 对于数值属性，直接加 Normalized Manhattan
        if len(self.harr_tool.numerical_cols) > 0:
            num_df = self.dataset[self.harr_tool.numerical_cols].copy()
            # 归一化
            num_df = (num_df - num_df.min()) / (num_df.max() - num_df.min() + 1e-6)
            num_dist = pairwise_distances(num_df, metric='manhattan')
            total_dist_matrix += num_dist

        # 3. 聚类 (基于距离矩阵，通常用层次聚类)
        model = AgglomerativeClustering(n_clusters=self.k, metric='precomputed', linkage='average')
        labels = model.fit_predict(total_dist_matrix)

        return labels


# ==========================================
# 4. 主流程
# ==========================================
def run_table7_experiment():
    # 获取数据
    harr_ds = HARRDataSet()
    print("正在加载所有数据集...")
    try:
        data1 = harr_ds.get_all_data_1()
        data2 = harr_ds.get_all_data_2()
        raw_data = {**data1, **data2}
    except:
        raw_data = harr_ds.get_all_data()

    # 筛选
    all_data = {name: raw_data[name] for name in TABLE7_DATASET_NAMES if name in raw_data}

    # 存储结果
    final_results = {ds: {m: [] for m in VARIANTS} for ds in all_data.keys()}

    n_repeats = 20

    print(f"开始串行运行 Table 7 (ARI) 实验 (共 {len(all_data)} 个数据集, 每个重复 {n_repeats} 次)...")

    # === 外层循环：数据集 ===
    for dataset_name, (dataset, columns) in all_data.items():
        k = DATASET_CONFIG.get(dataset_name, 2)
        print(f"\n>> 正在处理: {dataset_name} (k={k})")

        # --- 初始化 HARR 模型 (预处理只做一次) ---
        harr_ready = False
        harr_model = None

        try:
            harr_model = HARR(
                dataset,
                n_clusters=k,
                numerical_cols=columns.numerical_columns,
                nominal_cols=columns.nominal_columns,
                ordinal_cols=columns.ordinal_columns
            )
            harr_model.preprocess()
            if harr_model.X_encoded is not None:
                harr_model.X_encoded.columns = harr_model.X_encoded.columns.astype(str)
            harr_ready = True
        except Exception as e:
            print(f"   [HARR Preprocess Error]: {e}")

        # === 内层循环：重复实验 ===
        for i in range(n_repeats):
            print(f"   Run {i + 1}/{n_repeats}...", end="\r")

            # --- 1. KMD/KPT ---
            try:
                kpt = Wrapper_KPrototypes(dataset, columns, n_clusters=k)
                labels, _ = kpt.fit()
                score = adjusted_rand_score(columns.y_true, labels)
                final_results[dataset_name]['KMD/KPT'].append(score)
            except:
                final_results[dataset_name]['KMD/KPT'].append(0.0)

            # --- 2. BD (Base Distance) ---
            try:
                # 这是一个比较慢的操作，因为它每次都要算距离矩阵
                # 为了加速，其实可以在外层算好距离矩阵传进去，这里为了结构清晰在内层算
                bd_wrapper = Wrapper_BD(dataset, columns, n_clusters=k)
                # 注意：calculate_base_distance 已经在 HARR 类中实现，这里只是调用
                # 为了避免重复计算，我们在 Wrapper_BD 中实现逻辑
                # 但 Wrapper_BD 每次初始化都会重新计算，这会很慢
                # 优化策略：复用 harr_model 的方法，但只用其中间结果
                # 由于 BD 本质上是确定性的（给定距离矩阵后，层次聚类是确定的），
                # 其实不需要跑 20 次，跑一次就行。但为了保持代码结构一致，我们跑多次（每次结果一样）。
                if i == 0:
                    # 只在第一次跑，后面复制结果
                    labels_bd = bd_wrapper.fit()
                    score_bd = adjusted_rand_score(columns.y_true, labels_bd)

                final_results[dataset_name]['BD'].append(score_bd)
            except Exception as e:
                if i == 0: print(f"   [BD Error]: {e}")
                final_results[dataset_name]['BD'].append(0.0)

            # --- HARR 变体 ---
            if harr_ready:
                # --- 3. HAR (同质表示 + KMeans) ---
                try:
                    kmeans = KMeans(n_clusters=k, n_init=10, init='random', max_iter=300)
                    har_labels = kmeans.fit_predict(harr_model.X_encoded)
                    score = adjusted_rand_score(columns.y_true, har_labels)
                    final_results[dataset_name]['HAR'].append(score)
                except:
                    final_results[dataset_name]['HAR'].append(0.0)

                # --- 4. HARR-V ---
                try:
                    Q_v, _ = harr_model.fitV(max_iter=50)
                    score = adjusted_rand_score(columns.y_true, Q_v)
                    final_results[dataset_name]['HARR-V'].append(score)
                except:
                    final_results[dataset_name]['HARR-V'].append(0.0)

                # --- 5. HARR-M ---
                try:
                    Q_m, _ = harr_model.fitM(max_iter=50)
                    score = adjusted_rand_score(columns.y_true, Q_m)
                    final_results[dataset_name]['HARR-M'].append(score)
                except:
                    final_results[dataset_name]['HARR-M'].append(0.0)
            else:
                final_results[dataset_name]['HAR'].append(0.0)
                final_results[dataset_name]['HARR-V'].append(0.0)
                final_results[dataset_name]['HARR-M'].append(0.0)

        print("")

    # ==========================================
    # 5. 生成表格 (ARI Results & Average Rank)
    # ==========================================
    print("\n====== 生成 Table 7 (ARI Results) ======")
    table_data = {}
    avg_scores = {m: [] for m in VARIANTS}

    for m in VARIANTS:
        col_values = []
        for ds in TABLE7_DATASET_NAMES:
            if ds not in final_results:
                col_values.append("-")
                continue

            scores = final_results[ds][m]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                # 格式: 0.1234 ± 0.0012
                col_values.append(f"{mean_val:.4f} ± {std_val:.4f}")
                avg_scores[m].append(mean_val)
            else:
                col_values.append("N/A")
                avg_scores[m].append(0.0)
        table_data[m] = col_values

    # 创建 DataFrame
    df = pd.DataFrame(table_data, index=TABLE7_DATASET_NAMES)

    # 计算 Average Rank (AR)
    # ARI 数值越大越好，使用 descending 排序
    rank_df = pd.DataFrame(avg_scores)
    ranks = rank_df.rank(axis=1, ascending=False, method='min')
    ar_values = ranks.mean(axis=0)

    # 插入 AR 行
    ar_row = [f"{ar_values[m]:.4f}" for m in VARIANTS]
    df.loc['AR (Average Rank)'] = ar_row

    print(df)

    # 保存
    output_path = '../outputs/Table7_Ablation_ARI.csv'
    df.T.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    run_table7_experiment()