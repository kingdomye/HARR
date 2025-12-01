import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
import time
from joblib import Parallel, delayed, cpu_count
import warnings

# 请确保这些模块在你的路径下
from HARRDataset import HARRDataSet

# 使用优化后的 HARR 类 (支持向量化和进度条)
try:
    from HARR_V3_opt import HARR
except ImportError:
    from HARR_opt import HARR

from src.OTHERS import *

# ==========================================
# 配置部分
# Table 6: CA performance on Categorical Datasets
# ==========================================
DATASET_CONFIG = {
    'SB': 15, 'SF': 6, 'T3': 2, 'HR': 3,
    'LG': 4, 'MR': 2, 'LE': 5, 'SW': 4
}

# Table 6 的数据集顺序
TABLE6_DATASET_NAMES = ['SB', 'SF', 'T3', 'HR', 'LG', 'MR', 'LE', 'SW']

# 方法列表
METHODS_LIST = [
    'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
    'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
    'HARR-V', 'HARR-M'
]


# ==========================================
# 工具函数: 聚类准确率 (CA)
# ==========================================
def clustering_accuracy(y_true, y_pred):
    """
    计算聚类准确率 (CA)，自动处理字符串标签
    """
    # 1. 编码真实标签 (如 'cat', 'dog' -> 0, 1)
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    # 2. 确保预测标签为整数
    y_pred = np.array(y_pred).astype(np.int64)

    if y_pred.size != y_true.size:
        return 0.0

    # 3. 匈牙利算法匹配
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


# ==========================================
# 并行 Worker 函数
# ==========================================
def run_single_experiment_ca(run_idx, dataset, columns, k, harr_model_base):
    """
    执行单次实验，返回所有方法的 CA 分数
    """
    scores = {m: 0.0 for m in METHODS_LIST}

    # --- A. 运行对比方法 ---
    models = {
        'KMD/KPT': Wrapper_KPrototypes(dataset, columns, n_clusters=k),
        'OHE+OC': Wrapper_OHE_KMeans(dataset, columns, n_clusters=k),
        'SBC': Wrapper_SBC(dataset, columns, n_clusters=k),
        'JDM': Wrapper_JDM(dataset, columns, n_clusters=k),
        'CMS': Wrapper_CMS(dataset, columns, n_clusters=k),
        'UDM': Wrapper_UDM(dataset, columns, n_clusters=k),
        'HOD': Wrapper_HOD(dataset, columns, n_clusters=k),
        'GWD': Wrapper_GWD(dataset, columns, n_clusters=k),
        'GBD': Wrapper_GBD(dataset, columns, n_clusters=k),
        'FBD': Wrapper_FBD(dataset, columns, n_clusters=k)
    }

    for m_name, model in models.items():
        try:
            labels, _ = model.fit()
            # 计算 CA
            score = clustering_accuracy(columns.y_true, labels)
            scores[m_name] = score
        except Exception as e:
            if run_idx == 0:
                print(f"    [Worker-{run_idx}] {m_name} Error: {e}")
            scores[m_name] = 0.0

    # --- B. 运行 HARR (V & M) ---
    if harr_model_base is not None:
        # HARR-V
        try:
            Q_v, _ = harr_model_base.fitV()
            scores['HARR-V'] = clustering_accuracy(columns.y_true, Q_v)
        except Exception as e:
            scores['HARR-V'] = 0.0

        # HARR-M
        try:
            Q_m, _ = harr_model_base.fitM()
            scores['HARR-M'] = clustering_accuracy(columns.y_true, Q_m)
        except Exception as e:
            scores['HARR-M'] = 0.0

    return scores


# ==========================================
# 主流程
# ==========================================
def generate_table6():
    # 1. 获取数据
    harr_ds = HARRDataSet()
    print("正在加载数据集 (Categorical Datasets)...")
    try:
        raw_data = harr_ds.get_all_data_2()
    except AttributeError:
        raw_data = harr_ds.get_all_data()

    # 筛选 Table 6 所需的数据集
    all_data = {name: raw_data[name] for name in TABLE6_DATASET_NAMES if name in raw_data}

    # 2. 初始化结果存储
    final_results = {ds: {m: [] for m in METHODS_LIST} for ds in all_data.keys()}

    # 3. 设置并行参数
    n_repeats = 20
    n_jobs = max(1, cpu_count() - 2)
    print(f"检测到 CPU 核心数: {cpu_count()}，将使用 {n_jobs} 个进程并行计算。")

    # === 遍历数据集 ===
    for dataset_name, (dataset, columns) in all_data.items():
        k = DATASET_CONFIG.get(dataset_name, 2)
        print(f"\n====== 处理数据集: {dataset_name} (k={k}, Rows={len(dataset)}) ======")

        # --- HARR 预处理 (主进程执行一次) ---
        harr_model_base = HARR(
            dataset, n_clusters=k,
            numerical_cols=columns.numerical_columns,
            nominal_cols=columns.nominal_columns,
            ordinal_cols=columns.ordinal_columns
        )
        try:
            print("  正在执行 HARR 预处理...")
            start_time = time.time()
            harr_model_base.preprocess()
            print(f"  预处理完成，耗时: {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"  HARR 预处理失败: {e}")
            harr_model_base = None

        # --- 动态调整并发数 (防止大内存溢出) ---
        current_jobs = n_jobs
        if dataset_name == 'MR' and len(dataset) > 5000:
            current_jobs = min(n_jobs, 4)
            print(f"  注意: 检测到大数据集 {dataset_name}，并发数限制为 {current_jobs}")

        print(f"  正在并行运行 {n_repeats} 次实验 (Metric: CA)...")

        # 并行执行
        parallel_results = Parallel(n_jobs=current_jobs, verbose=5)(
            delayed(run_single_experiment_ca)(
                i, dataset, columns, k, harr_model_base
            ) for i in range(n_repeats)
        )

        # 收集结果
        for single_run_scores in parallel_results:
            for m_name, score in single_run_scores.items():
                final_results[dataset_name][m_name].append(score)

    # 4. 生成表格 DataFrame
    print("\n====== 生成 Table 6 结果 (CA Performance) ======")
    table_data = {}

    for m in METHODS_LIST:
        row_values = []
        for ds in TABLE6_DATASET_NAMES:
            if ds not in final_results:
                row_values.append("-")
                continue

            scores = final_results[ds][m]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                # 格式: Mean ± Std
                cell_str = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                cell_str = "N/A"
            row_values.append(cell_str)
        table_data[m] = row_values

    df_table6 = pd.DataFrame(table_data, index=TABLE6_DATASET_NAMES).T
    print(df_table6)

    # 保存
    output_file = '../outputs/Table6_Result_CA(HARR_V3结果).csv'
    df_table6.to_csv(output_file, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_file}")


if __name__ == '__main__':
    generate_table6()