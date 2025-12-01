import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import warnings

from sklearn.preprocessing import LabelEncoder

# 请确保这些模块在你的路径下
from HARRDataset import HARRDataSet
from HARR_v3 import HARR
from src.OTHERS import *

# ==========================================
# 论文 Table 2 定义的真实簇数 (k*)
# Table 5 涉及的数据集 (Mixed Data): DS, HF, AA, AP, DT, AC
# ==========================================
DATASET_CONFIG = {
    'DS': 2,
    'HF': 2,
    'AA': 2,
    'AP': 2,
    'DT': 6,
    'AC': 2
}

# Table 5 的数据集顺序 (列顺序)
TABLE5_DATASET_NAMES = ['DS', 'HF', 'AA', 'AP', 'DT', 'AC']


def clustering_accuracy(y_true, y_pred):
    """
    计算聚类准确率 (Clustering Accuracy / Purity)
    自动处理字符串类型的标签
    """
    # 1. 使用 LabelEncoder 将真实标签 (如 'NO', 'YES') 转换为整数 (0, 1)
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    # 2. 确保预测标签也是整数
    y_pred = np.array(y_pred).astype(np.int64)

    # 3. 确保两者长度一致
    if y_pred.size != y_true.size:
        raise ValueError("y_true and y_pred must have the same length.")

    # 4. 构造混淆矩阵 (Contingency Matrix)
    # 维度取两者最大类别ID + 1，防止越界
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        # y_pred 是行 (预测簇), y_true 是列 (真实类)
        w[y_pred[i], y_true[i]] += 1

    # 5. 匈牙利算法求解最大匹配
    # scipy 实现的是最小权匹配，所以用 max() - w 将最大化问题转换为最小化问题
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # 6. 计算准确率
    return w[row_ind, col_ind].sum() / y_pred.size


def generate_table5():
    # 1. 获取数据
    harr_ds = HARRDataSet()

    print("正在加载数据集 (Table 5 - Mixed Datasets)...")
    # Table 5 使用混合数据集，通常对应 get_all_data_1
    try:
        raw_data = harr_ds.get_all_data_1()
    except AttributeError:
        print("警告: 未找到 get_all_data_1()，尝试使用 get_all_data() 并筛选...")
        raw_data = harr_ds.get_all_data()

    # 筛选并排序
    all_data = {name: raw_data[name] for name in TABLE5_DATASET_NAMES if name in raw_data}

    if len(all_data) < len(TABLE5_DATASET_NAMES):
        missing = set(TABLE5_DATASET_NAMES) - set(all_data.keys())
        print(f"警告: 以下数据集未加载到: {missing}")

    # 2. 定义方法列表
    methods_list = [
        'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
        'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
        'HARR-V', 'HARR-M'
    ]

    # 3. 初始化存储结构
    results = {ds: {m: [] for m in methods_list} for ds in all_data.keys()}

    # 4. 设置实验重复次数 (论文为 20)
    n_repeats = 20
    print(f"开始运行 Table 5 (Clustering Accuracy) 复现实验 (重复 {n_repeats} 次)...")

    # === 外层循环：遍历数据集 ===
    for dataset_name, (dataset, columns) in all_data.items():
        k = DATASET_CONFIG.get(dataset_name)
        if k is None:
            print(f"警告: 数据集 {dataset_name} 未配置 k 值，默认 k=2")
            k = 2

        print(f"\n正在处理数据集: {dataset_name} (k={k}, Rows={len(dataset)})...")

        # 预处理 HARR
        harr_model_base = HARR(
            dataset,
            n_clusters=k,
            numerical_cols=columns.numerical_columns,
            nominal_cols=columns.nominal_columns,
            ordinal_cols=columns.ordinal_columns
        )
        try:
            harr_model_base.preprocess()
            harr_ready = True
        except Exception as e:
            print(f"  [HARR Preprocess Error]: {e}")
            harr_ready = False

        # === 内层循环：重复实验 ===
        for i in range(n_repeats):
            if (i + 1) % 5 == 0:
                print(f"  -> Run {i + 1}/{n_repeats}")

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
                    # 关键修改：使用 clustering_accuracy 而不是 adjusted_rand_score
                    score = clustering_accuracy(columns.y_true, labels)
                    results[dataset_name][m_name].append(score)
                except Exception as e:
                    if i == 0: print(f"    [{m_name}] 失败: {e}")
                    results[dataset_name][m_name].append(0.0)

            # --- B. 运行 HARR ---
            if harr_ready:
                # HARR-V
                try:
                    Q_v, w_v = harr_model_base.fitV()
                    score_v = clustering_accuracy(columns.y_true, Q_v)
                    results[dataset_name]['HARR-V'].append(score_v)
                except Exception as e:
                    if i == 0: print(f"    [HARR-V] 失败: {e}")
                    results[dataset_name]['HARR-V'].append(0.0)

                # HARR-M
                try:
                    Q_m, w_m = harr_model_base.fitM()
                    score_m = clustering_accuracy(columns.y_true, Q_m)
                    results[dataset_name]['HARR-M'].append(score_m)
                except Exception as e:
                    if i == 0: print(f"    [HARR-M] 失败: {e}")
                    results[dataset_name]['HARR-M'].append(0.0)

    # 5. 生成表格
    print("\n====== 生成 Table 5 结果 (CA) ======")
    table_data = {}

    for m in methods_list:
        row_values = []
        for ds in TABLE5_DATASET_NAMES:
            if ds not in results:
                row_values.append("Missing")
                continue

            scores = results[ds][m]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                # 格式: Mean ± Std
                cell_str = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                cell_str = "N/A"
            row_values.append(cell_str)
        table_data[m] = row_values

    df_table5 = pd.DataFrame(table_data, index=TABLE5_DATASET_NAMES).T
    print(df_table5)

    # 保存
    output_file = '../outputs/Table5_Result_CA（HARR_V3簇中心优化结果）.csv'
    df_table5.to_csv(output_file, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_file}")


if __name__ == '__main__':
    generate_table5()
