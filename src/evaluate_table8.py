import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
import time

# 忽略 KMeans 和 Pandas 的特定警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入 HARR 类
# ==========================================
try:
    # 假设你之前保存的文件名为 HARR_v3.py
    from HARR_v3 import HARR
except ImportError:
    print("错误：找不到 HARR_v3.py，请确认文件名。")
    exit()

from HARRDataset import HARRDataSet
from src.OTHERS import Wrapper_KPrototypes

# ==========================================
# 2. 配置项
# ==========================================
# Table 8 包含所有 14 个数据集
DATASET_CONFIG = {
    # Mixed
    'DS': 2, 'HF': 2, 'AA': 2, 'AP': 2, 'DT': 6, 'AC': 2,
    # Categorical
    'SB': 15, 'SF': 6, 'T3': 2, 'HR': 3, 'LG': 4, 'MR': 2, 'LE': 5, 'SW': 4
}
TABLE8_DATASET_NAMES = list(DATASET_CONFIG.keys())

# 要对比的变体
VARIANTS = ['KMD/KPT', 'HAR', 'HARR-V', 'HARR-M']


# ==========================================
# 3. 辅助函数
# ==========================================
def clustering_accuracy(y_true, y_pred):
    """ 计算聚类准确率 (CA) """
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)
    y_pred = np.array(y_pred).astype(np.int64)

    if y_pred.size != y_true.size:
        return 0.0

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def run_experiment_serial():
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
    all_data = {name: raw_data[name] for name in TABLE8_DATASET_NAMES if name in raw_data}

    # 存储结果
    final_results = {ds: {m: [] for m in VARIANTS} for ds in all_data.keys()}

    n_repeats = 20

    print(f"开始串行运行 Table 8 实验 (共 {len(all_data)} 个数据集, 每个重复 {n_repeats} 次)...")

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

            # 【关键修复】: 强制将所有列名转换为字符串，解决 ['int', 'str'] 混合类型的报错
            if harr_model.X_encoded is not None:
                harr_model.X_encoded.columns = harr_model.X_encoded.columns.astype(str)

            harr_ready = True
        except Exception as e:
            print(f"   [HARR Preprocess Error]: {e}")

        # === 内层循环：重复实验 ===
        for i in range(n_repeats):
            # 简单的进度提示
            print(f"   Run {i + 1}/{n_repeats}...", end="\r")

            # --- 1. KMD/KPT (基线) ---
            try:
                kpt = Wrapper_KPrototypes(dataset, columns, n_clusters=k)
                labels, _ = kpt.fit()
                score = clustering_accuracy(columns.y_true, labels)
                final_results[dataset_name]['KMD/KPT'].append(score)
            except Exception as e:
                if i == 0: print(f"\n   [KPT Error]: {e}")
                final_results[dataset_name]['KMD/KPT'].append(0.0)

            # --- HARR 相关变体 ---
            if harr_ready and harr_model.X_encoded is not None:
                # --- 2. HAR (同质表示 + KMeans, 无权重学习) ---
                try:
                    # 使用预处理好的 X_encoded (列名已修复为 str)
                    kmeans = KMeans(n_clusters=k, n_init=10, init='random', max_iter=300)
                    har_labels = kmeans.fit_predict(harr_model.X_encoded)
                    score = clustering_accuracy(columns.y_true, har_labels)
                    final_results[dataset_name]['HAR'].append(score)
                except Exception as e:
                    if i == 0: print(f"\n   [HAR Error]: {e}")
                    final_results[dataset_name]['HAR'].append(0.0)

                # --- 3. HARR-V (向量权重) ---
                try:
                    # fitV 内部会用到 random 初始化，每次结果不同
                    Q_v, _ = harr_model.fitV(max_iter=50)
                    score = clustering_accuracy(columns.y_true, Q_v)
                    final_results[dataset_name]['HARR-V'].append(score)
                except Exception as e:
                    if i == 0: print(f"\n   [HARR-V Error]: {e}")
                    final_results[dataset_name]['HARR-V'].append(0.0)

                # --- 4. HARR-M (矩阵权重) ---
                try:
                    Q_m, _ = harr_model.fitM(max_iter=50)
                    score = clustering_accuracy(columns.y_true, Q_m)
                    final_results[dataset_name]['HARR-M'].append(score)
                except Exception as e:
                    if i == 0: print(f"\n   [HARR-M Error]: {e}")
                    final_results[dataset_name]['HARR-M'].append(0.0)
            else:
                # 如果预处理失败，填 0
                final_results[dataset_name]['HAR'].append(0.0)
                final_results[dataset_name]['HARR-V'].append(0.0)
                final_results[dataset_name]['HARR-M'].append(0.0)

        print("")  # 换行

    # ==========================================
    # 4. 生成结果表格
    # ==========================================
    print("\n====== 生成 Table 8 (CA Results) ======")
    table_data = {}
    avg_scores = {m: [] for m in VARIANTS}  # 用于计算平均排名

    for m in VARIANTS:
        col_values = []
        for ds in TABLE8_DATASET_NAMES:
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
    df = pd.DataFrame(table_data, index=TABLE8_DATASET_NAMES)

    # 计算 Average Rank (AR)
    # 1. 临时 DataFrame (行=数据集, 列=方法, 值=CA均值)
    rank_df = pd.DataFrame(avg_scores)  # 默认索引 0..13

    # 2. 对每行进行排名 (数值越大越好，descending)
    # rank=1 表示最大值(第一名)
    ranks = rank_df.rank(axis=1, ascending=False, method='min')

    # 3. 计算列均值作为 AR
    ar_values = ranks.mean(axis=0)

    # 4. 插入 AR 行到 DataFrame 底部
    ar_row = [f"{ar_values[m]:.4f}" for m in VARIANTS]
    df.loc['AR (Average Rank)'] = ar_row

    # 打印并保存
    print(df)

    # 转置保存，符合论文格式 (行是 AR, IP 等)
    output_path = '../outputs/Table8_Ablation_CA.csv'
    df.T.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    run_experiment_serial()