import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import warnings

from HARRDataset import HARRDataSet
from HARR_v3 import HARR
from src.OTHERS import *

# ==========================================
# 1. 论文 Table 2 定义的真实簇数 (k*)
# ==========================================
DATASET_CONFIG = {
    # Mixed Datasets (Table 3)
    'DS': 2,
    'HF': 2,
    'AA': 2,
    'AP': 2,
    'DT': 6,
    'AC': 2,

    # Categorical Datasets (Table 4, 备用)
    'SB': 15,
    'SF': 6,
    'T3': 2,
    'HR': 3,
    'LG': 4,
    'MR': 2,
    'LE': 5,
    'SW': 4
}


def generate_table3():
    # 1. 获取数据
    harr_ds = HARRDataSet()
    # 假设 get_all_data_1 返回的是 Table 3 涉及的 6 个混合数据集
    all_data = harr_ds.get_all_data_1()

    # 2. 定义方法列表 (对应表格的行)
    methods_list = [
        'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
        'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
        'HARR-V', 'HARR-M'
    ]

    # 3. 初始化存储结构
    # results[dataset_name][method_name] = [score1, score2, ...]
    dataset_names = list(all_data.keys())
    results = {ds: {m: [] for m in methods_list} for ds in dataset_names}

    # 4. 设置实验重复次数 (论文通常是 20，调试时建议改小，比如 5)
    n_repeats = 20
    print(f"开始运行 Table 3 复现实验 (重复 {n_repeats} 次)...")

    # === 外层循环：遍历数据集 ===
    for dataset_name, (dataset, columns) in all_data.items():
        # 获取论文指定的 k 值
        k = DATASET_CONFIG.get(dataset_name)
        if k is None:
            print(f"警告: 数据集 {dataset_name} 未在配置中找到 k 值，默认使用 2")
            k = 2

        print(f"\n正在处理数据集: {dataset_name} (k={k})...")

        # 预处理 HARR (因为 preprocess 是确定性的，只需做一次)
        # 注意：HARR 类的初始化可能需要 k，也可能不需要，取决于你的实现
        # 这里假设 fitV/fitM 时才用到 k，或者 init 时传入
        harr_model_base = HARR(dataset, n_clusters=k, numerical_cols=columns.numerical_columns,
                               nominal_cols=columns.nominal_columns, ordinal_cols=columns.ordinal_columns)
        # 确保传入的是分类列名列表
        harr_model_base.preprocess()
        harr_ready = True

        # === 内层循环：重复实验以计算 Mean ± Std ===
        for i in range(n_repeats):
            # 打印进度 (每5次打印一次)
            if (i + 1) % 5 == 0:
                print(f"  -> Run {i + 1}/{n_repeats}")

            # --- A. 运行 10 个对比方法 ---
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
                    score = adjusted_rand_score(columns.y_true, labels)
                    results[dataset_name][m_name].append(score)
                except Exception as e:
                    # 仅在第一次出错时打印，防止刷屏
                    if i == 0: print(f"    [{m_name}] 失败: {e}")
                    results[dataset_name][m_name].append(0.0)

            # --- B. 运行 HARR (V & M) ---
            if harr_ready:
                # 运行 HARR-V
                try:
                    # fitV 内部应该包含随机初始化逻辑
                    Q_v, w_v = harr_model_base.fitV()
                    score_v = adjusted_rand_score(columns.y_true, Q_v)
                    results[dataset_name]['HARR-V'].append(score_v)
                except Exception as e:
                    if i == 0: print(f"    [HARR-V] 失败: {e}")
                    results[dataset_name]['HARR-V'].append(0.0)

                # 运行 HARR-M
                try:
                    Q_m, w_m = harr_model_base.fitM()
                    score_m = adjusted_rand_score(columns.y_true, Q_m)
                    results[dataset_name]['HARR-M'].append(score_m)
                except Exception as e:
                    if i == 0: print(f"    [HARR-M] 失败: {e}")
                    results[dataset_name]['HARR-M'].append(0.0)

    # 5. 生成最终表格 DataFrame
    print("\n====== 正在生成结果表格 ======")

    table_data = {}

    for m in methods_list:
        row_values = []
        for ds in dataset_names:
            scores = results[ds][m]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                # 格式化: 0.1234 ± 0.0567
                cell_str = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                cell_str = "N/A"
            row_values.append(cell_str)
        table_data[m] = row_values

    # 创建 DataFrame: 行是算法，列是数据集
    # 论文 Table 3 的格式是：行=算法，列=数据集
    df_table3 = pd.DataFrame(table_data, index=dataset_names).T

    print(df_table3)

    # 保存 CSV
    df_table3.to_csv('../outputs/Table3_Result_v3_(HARR_V3_簇中心优化)).csv', encoding='utf-8-sig')
    print("\n结果已保存至 Table3_Result_v3_(HARR_V3_簇中心优化).csv")


if __name__ == '__main__':
    generate_table3()
