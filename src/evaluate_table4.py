import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import warnings
from joblib import Parallel, delayed, cpu_count
import time

# 请确保这些模块在你的路径下
from HARRDataset import HARRDataSet
# 确保使用的是优化后的 HARR 类
from HARR_V3_opt import HARR
from src.OTHERS import *

# ==========================================
# 配置部分
# ==========================================
DATASET_CONFIG = {
    'SB': 15, 'SF': 6, 'T3': 2, 'HR': 3,
    'LG': 4, 'MR': 2, 'LE': 5, 'SW': 4
}
TABLE4_DATASET_NAMES = ['SB', 'SF', 'T3', 'HR', 'LG', 'MR', 'LE', 'SW']

# 想要运行的方法列表
METHODS_LIST = [
    'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
    'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
    'HARR-V', 'HARR-M'
]


def run_single_experiment(run_idx, dataset, columns, k, harr_model_base):
    """
    执行单次实验的 Worker 函数
    run_idx: 当前是第几次重复
    """
    scores = {m: 0.0 for m in METHODS_LIST}

    # --- A. 运行对比方法 ---
    # 在这里实例化模型，确保每次运行的随机种子不同(如果模型内部有随机性)
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
            # 大多数 Wrapper 内部会有随机初始化，如果看起来结果一样，可能需要手动设种子
            # np.random.seed(int(time.time() + run_idx * 1000))
            labels, _ = model.fit()
            score = adjusted_rand_score(columns.y_true, labels)
            scores[m_name] = score
        except Exception as e:
            # 只有第一次重复时打印错误，避免并行时控制台刷屏
            if run_idx == 0:
                print(f"    [Worker-{run_idx}] {m_name} Error: {e}")
            scores[m_name] = 0.0

    # --- B. 运行 HARR (V & M) ---
    if harr_model_base is not None:
        # HARR-V
        try:
            # fitV 内部使用了 random.sample 或 np.random，多进程下通常是安全的
            Q_v, _ = harr_model_base.fitV()
            scores['HARR-V'] = adjusted_rand_score(columns.y_true, Q_v)
        except Exception as e:
            scores['HARR-V'] = 0.0

        # HARR-M
        try:
            Q_m, _ = harr_model_base.fitM()
            scores['HARR-M'] = adjusted_rand_score(columns.y_true, Q_m)
        except Exception as e:
            scores['HARR-M'] = 0.0

    return scores


def generate_table4_parallel():
    # 1. 获取数据
    harr_ds = HARRDataSet()
    print("正在加载数据集...")
    try:
        raw_data = harr_ds.get_all_data_2()
    except AttributeError:
        raw_data = harr_ds.get_all_data()

    all_data = {name: raw_data[name] for name in TABLE4_DATASET_NAMES if name in raw_data}

    # 2. 初始化结果存储
    # results[dataset][method] = [score1, score2...]
    final_results = {ds: {m: [] for m in METHODS_LIST} for ds in all_data.keys()}

    # 3. 设置重复次数
    n_repeats = 20

    # 自动检测 CPU 核心数，留 1-2 个核给系统
    n_jobs = max(1, cpu_count() - 2)
    print(f"检测到 CPU 核心数: {cpu_count()}，将使用 {n_jobs} 个进程并行计算。")

    # === 外层循环：数据集 (串行处理数据集，防止内存撑爆) ===
    for dataset_name, (dataset, columns) in all_data.items():
        k = DATASET_CONFIG.get(dataset_name, 2)
        print(f"\n====== 处理数据集: {dataset_name} (k={k}, Rows={len(dataset)}) ======")

        # --- HARR 预处理 (只做一次) ---
        # 预处理是确定性的，不需要在循环里做
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
            harr_ready = True
        except Exception as e:
            print(f"  HARR 预处理失败: {e}")
            harr_ready = False
            harr_model_base = None

        # --- 并行执行 n_repeats 次实验 ---
        # 对于大数据集 (如 MR)，如果内存吃紧，可以将 n_jobs 调小，例如 n_jobs=4
        current_jobs = n_jobs
        if dataset_name == 'MR' and len(dataset) > 5000:
            # MR 数据集较大，为了防止 10 个进程同时申请大内存导致 OOM，可以适当降低并发
            current_jobs = min(n_jobs, 4)
            print(f"  注意: 检测到大数据集 {dataset_name}，并发数限制为 {current_jobs}")

        print(f"  正在并行运行 {n_repeats} 次实验...")

        # Joblib 并行入口
        parallel_results = Parallel(n_jobs=current_jobs, verbose=5)(
            delayed(run_single_experiment)(
                i, dataset, columns, k, harr_model_base
            ) for i in range(n_repeats)
        )

        # --- 收集结果 ---
        # parallel_results 是一个 list，包含 n_repeats 个字典
        for single_run_scores in parallel_results:
            for m_name, score in single_run_scores.items():
                final_results[dataset_name][m_name].append(score)

    # 4. 生成表格
    print("\n====== 生成 Table 4 结果 ======")
    table_data = {}
    for m in METHODS_LIST:
        row_values = []
        for ds in TABLE4_DATASET_NAMES:
            if ds not in final_results:
                row_values.append("-")
                continue
            scores = final_results[ds][m]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                cell_str = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                cell_str = "N/A"
            row_values.append(cell_str)
        table_data[m] = row_values

    df_table4 = pd.DataFrame(table_data, index=TABLE4_DATASET_NAMES).T
    print(df_table4)

    output_file = '../outputs/Table4_Result_Parallel.csv'
    df_table4.to_csv(output_file, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_file}")


if __name__ == '__main__':
    # Windows 下使用多进程必须放在 if __name__ == '__main__': 之下
    generate_table4_parallel()