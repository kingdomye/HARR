import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import warnings
import time
import copy
from tqdm import tqdm  # 建议安装 tqdm: pip install tqdm

# ==========================================
# 1. 导入模块
# ==========================================
# 确保这些文件在你的 Python 路径中
from HARRDataset import HARRDataSet
from HARR_v3 import HARR
from src.OTHERS import *

# ==========================================
# 2. 配置部分
# ==========================================
DATASET_CONFIG = {
    'SB': 15, 'SF': 6, 'T3': 2, 'HR': 3,
    'LG': 4, 'MR': 2, 'LE': 5, 'SW': 4
}
# 你想要运行的数据集列表
TABLE4_DATASET_NAMES = ['SB', 'SF', 'T3', 'HR', 'LG', 'MR', 'LE', 'SW']

# 想要运行的方法列表
METHODS_LIST = [
    'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
    'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
    'HARR-V', 'HARR-M'
]


# ==========================================
# 3. 单次实验逻辑
# ==========================================
def run_single_experiment(run_idx, dataset, columns, k, harr_model_preprocessed):
    """
    执行单次实验
    run_idx: 当前是第几次重复
    harr_model_preprocessed: 已经做完 preprocess 的 HARR 对象
    """
    scores = {m: 0.0 for m in METHODS_LIST}

    # --- A. 运行对比方法 ---
    # 实例化模型
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
            # 运行 fit 获取标签
            labels, _ = model.fit()
            score = adjusted_rand_score(columns.y_true, labels)
            scores[m_name] = score
        except Exception as e:
            # 仅在第一次运行时报错，避免刷屏
            if run_idx == 0:
                print(f"    [Warning] {m_name} 运行失败: {e}")
            scores[m_name] = 0.0

    # --- B. 运行 HARR (V & M) ---
    if harr_model_preprocessed is not None:
        # HARR-V
        try:
            # 【关键】必须使用 deepcopy！
            # 因为是单线程，如果不拷贝，第二次运行会使用第一次训练后的权重，导致实验不独立。
            # 拷贝后，我们保留了 preprocess 的结果，但重置了聚类状态。
            model_v = copy.deepcopy(harr_model_preprocessed)

            # 这里的 fitV 应该包含随机初始化中心点的逻辑
            Q_v, _ = model_v.fitV()
            scores['HARR-V'] = adjusted_rand_score(columns.y_true, Q_v)
        except Exception as e:
            if run_idx == 0:
                print(f"    [Warning] HARR-V 运行失败: {e}")
            scores['HARR-V'] = 0.0

        # HARR-M
        try:
            model_m = copy.deepcopy(harr_model_preprocessed)
            Q_m, _ = model_m.fitM()
            scores['HARR-M'] = adjusted_rand_score(columns.y_true, Q_m)
        except Exception as e:
            if run_idx == 0:
                print(f"    [Warning] HARR-M 运行失败: {e}")
            scores['HARR-M'] = 0.0

    return scores


# ==========================================
# 4. 主流程 (单线程版)
# ==========================================
def generate_table4_linear():
    warnings.filterwarnings("ignore")  # 忽略警告保持清爽

    # 1. 获取数据
    harr_ds = HARRDataSet()
    print("正在加载数据集...")
    try:
        raw_data = harr_ds.get_all_data_2()  # 假设分类数据集在 data_2
    except AttributeError:
        # 兼容性处理
        raw_data = harr_ds.get_all_data()

    # 过滤出需要的数据集
    all_data = {name: raw_data[name] for name in TABLE4_DATASET_NAMES if name in raw_data}

    # 2. 初始化结果存储
    final_results = {ds: {m: [] for m in METHODS_LIST} for ds in all_data.keys()}

    # 3. 设置重复次数
    n_repeats = 20

    print(f"开始单线程执行，每个数据集重复 {n_repeats} 次...")

    # === 循环数据集 ===
    for dataset_name, (dataset, columns) in all_data.items():
        k = DATASET_CONFIG.get(dataset_name, 2)
        print(f"\n====== 处理数据集: {dataset_name} (k={k}, Rows={len(dataset)}) ======")

        # --- HARR 预处理 (只做一次，节省时间) ---
        harr_model_base = HARR(
            dataset, n_clusters=k,
            numerical_cols=columns.numerical_columns,
            nominal_cols=columns.nominal_columns,
            ordinal_cols=columns.ordinal_columns
        )

        try:
            print("  [Init] 正在执行 HARR 预处理 (Projection)...")
            start_time = time.time()
            harr_model_base.preprocess()
            print(f"  [Init] 预处理完成，耗时: {time.time() - start_time:.2f}s")
            harr_ready = True
        except Exception as e:
            print(f"  [Error] HARR 预处理失败: {e}")
            harr_ready = False
            harr_model_base = None

        # --- 循环实验 (单线程) ---
        # 使用 tqdm 显示进度条
        for i in tqdm(range(n_repeats), desc=f"  Running {dataset_name}"):
            # 调用单次实验函数
            scores = run_single_experiment(
                run_idx=i,
                dataset=dataset,
                columns=columns,
                k=k,
                harr_model_preprocessed=harr_model_base if harr_ready else None
            )

            # 收集结果
            for m_name, score in scores.items():
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
                # 格式化输出: Mean ± Std
                cell_str = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                cell_str = "N/A"
            row_values.append(cell_str)
        table_data[m] = row_values

    df_table4 = pd.DataFrame(table_data, index=TABLE4_DATASET_NAMES).T
    print(df_table4)

    # 保存
    output_file = '../outputs/Table4_Result_SingleThread.csv'
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_table4.to_csv(output_file, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_file}")


if __name__ == '__main__':
    generate_table4_linear()