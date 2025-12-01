import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# 假设你已正确实现以下导入的类和函数
from HARRDataset import HARRDataSet
from HARR_v3 import HARR
from OTHERS import Wrapper_SBC, Wrapper_GBD, Wrapper_FBD  # 确保这些类已正确导入

# ==========================================
# 0. 全局设置与工具函数
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial']  # 统一字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def convert_numerical_columns(df, numerical_cols):
    """强制转换数值列为数值类型，处理缺失值"""
    if not numerical_cols:
        return df

    df_copy = df.copy()
    for col in numerical_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # 必须在所有转换完成后再删除，以保证索引一致性
    df_copy = df_copy.dropna(subset=numerical_cols).reset_index(drop=True)
    return df_copy


def to_list(cols):
    if cols is None:
        return []
    if isinstance(cols, (int, float, str, np.integer)):
        return [cols]
    if len(cols) == 0:
        return []
    return list(cols)


# ==========================================
# 1. 评价指标 (Metrics)
# ==========================================
def clustering_accuracy(y_true, y_pred):
    """计算聚类精度 (Purity/ACC)"""
    y_true_le = LabelEncoder().fit_transform(y_true)
    y_pred_le = LabelEncoder().fit_transform(y_pred)

    if y_pred_le.size != y_true_le.size:
        return 0.0

    D = max(y_pred_le.max(), y_true_le.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred_le.size):
        w[y_pred_le[i], y_true_le[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred_le.size


# ==========================================
# 2. 数据集加载类 (已修复)
# ==========================================
class DataColumns:
    def __init__(self, numerical_columns, nominal_columns, ordinal_columns, y_true):
        self.numerical_columns = numerical_columns
        self.nominal_columns = nominal_columns
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = self.nominal_columns + self.ordinal_columns
        self.y_true = y_true


# ==========================================
# 3. 实验运行逻辑 (保持不变)
# ==========================================
def run_all_experiments():
    dataset_loader = HARRDataSet()
    datasets = ['AP', 'DT', 'AC', 'HR', 'LG']

    # 你的 Wrapper 算法 + HARR的两个变体（共5个分类方法）
    wrapper_methods = {
        "SBC": Wrapper_SBC,
        "GBD": Wrapper_GBD,
        "FBD": Wrapper_FBD
    }

    results = {}
    n_runs = 20  # 为快速演示设为5次，正式复现请用20

    print(f"{'=' * 60}\nRunning all experiments (n_runs={n_runs})\n{'=' * 60}")

    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")
        df, cols = dataset_loader.get_dataset_by_name(ds_name)
        if df is None:
            continue

        y_true = cols.y_true
        # 合并cols.numerical_columns cols.categorical_columns
        combined_cols = to_list(cols.numerical_columns) + to_list(cols.categorical_columns)
        df_features = df[combined_cols]
        n_clusters = len(np.unique(y_true))

        results[ds_name] = {}

        # --- 运行 Wrapper 算法 ---
        for name, wrapper_class in wrapper_methods.items():
            aris, cas = [], []
            for i in range(n_runs):
                model = wrapper_class(df_features, cols, n_clusters)
                try:
                    labels, _ = model.fit()
                    # 确保标签长度与真实标签对齐
                    if len(labels) == len(y_true):
                        aris.append(adjusted_rand_score(y_true, labels))
                        cas.append(clustering_accuracy(y_true, labels))
                except Exception as e:
                    print(f"  Error in {name} run {i}: {e}")
            results[ds_name][name] = {'ari': np.mean(aris), 'ca': np.mean(cas)}
            print(f"  > {name}: ARI={np.mean(aris):.3f}, CA={np.mean(cas):.3f}")

        # --- 运行 HARR 消融实验 ---
        # 模式1: Distinguish (消融前)
        aris_v_d, cas_v_d, aris_m_d, cas_m_d = [], [], [], []
        # 模式2: Not Distinguish (消融后)
        aris_v_nd, cas_v_nd, aris_m_nd, cas_m_nd = [], [], [], []

        for i in range(n_runs):
            # Distinguish
            model_d = HARR(df_features, n_clusters, cols.numerical_columns, cols.nominal_columns, cols.ordinal_columns)
            model_d.preprocess()
            lv_d, _ = model_d.fitV()
            lm_d, _ = model_d.fitM()
            if len(lv_d) == len(y_true):
                aris_v_d.append(adjusted_rand_score(y_true, lv_d))
                cas_v_d.append(clustering_accuracy(y_true, lv_d))
                aris_m_d.append(adjusted_rand_score(y_true, lm_d))
                cas_m_d.append(clustering_accuracy(y_true, lm_d))

            # Not Distinguish
            fake_nom = to_list(cols.nominal_columns) + to_list(cols.ordinal_columns)
            model_nd = HARR(df_features, n_clusters, cols.numerical_columns, fake_nom, [])
            model_nd.preprocess()
            lv_nd, _ = model_nd.fitV()
            lm_nd, _ = model_nd.fitM()
            if len(lv_nd) == len(y_true):
                aris_v_nd.append(adjusted_rand_score(y_true, lv_nd))
                cas_v_nd.append(clustering_accuracy(y_true, lv_nd))
                aris_m_nd.append(adjusted_rand_score(y_true, lm_nd))
                cas_m_nd.append(clustering_accuracy(y_true, lm_nd))

        # 存储 HARR 结果（5个分类方法：SBC/GBD/FBD/HARR-V/HARR-M）
        results[ds_name]['HARR-V'] = {
            'ari_d': np.mean(aris_v_d), 'ca_d': np.mean(cas_v_d),  # 消融前 (D)
            'ari_nd': np.mean(aris_v_nd), 'ca_nd': np.mean(cas_v_nd)  # 消融后 (ND)
        }
        results[ds_name]['HARR-M'] = {
            'ari_d': np.mean(aris_m_d), 'ca_d': np.mean(cas_m_d),  # 消融前 (D)
            'ari_nd': np.mean(aris_m_nd), 'ca_nd': np.mean(cas_m_nd)  # 消融后 (ND)
        }
        print(f"  > HARR-V(D): ARI={np.mean(aris_v_d):.3f}, HARR-V(ND): ARI={np.mean(aris_v_nd):.3f}")
        print(f"  > HARR-M(D): ARI={np.mean(aris_m_d):.3f}, HARR-M(ND): ARI={np.mean(aris_m_nd):.3f}")

    return results


# ==========================================
# 5. 绘图 (重点修改部分：生成10个图)
# ==========================================
def plot_ablation_comparison(data):
    """
    生成10个图：5个数据集 × 2个指标（ARI/CA）
    每个图：横轴=5个分类方法，每个位置2条柱形（蓝色=消融前，橙色=消融后）
    """
    datasets = ['AP', 'DT', 'AC', 'HR', 'LG']  # 固定数据集顺序
    methods = ['SBC', 'GBD', 'FBD', 'HARR-V', 'HARR-M']  # 5个分类方法
    metrics = ['ARI', 'CA']  # 2个评价指标
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色（消融前）、橙色（消融后）
    bar_width = 0.35  # 柱形宽度
    x = np.arange(len(methods))  # 横轴位置

    # 创建2×5的子图布局（2个指标 × 5个数据集）
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    fig.suptitle('Ablation Experiment Results (Blue: Distinguish, Orange: Not Distinguish)', fontsize=16, y=0.98)

    # 遍历每个指标和数据集
    for metric_idx, metric in enumerate(metrics):
        for ds_idx, ds_name in enumerate(datasets):
            ax = axes[metric_idx, ds_idx]

            # 准备每个方法的消融前/后数据
            pre_ablation = []  # 消融前（Distinguish）
            post_ablation = []  # 消融后（Not Distinguish）

            for method in methods:
                if method in ['SBC', 'GBD', 'FBD']:
                    # Wrapper方法没有消融，前后数据一致
                    val = data[ds_name][method][metric.lower()]
                    pre_ablation.append(val)
                    post_ablation.append(val)
                elif method == 'HARR-V':
                    # HARR-V的消融前/后数据
                    pre_val = data[ds_name][method][f'{metric.lower()}_d']
                    post_val = data[ds_name][method][f'{metric.lower()}_nd']
                    pre_ablation.append(pre_val)
                    post_ablation.append(post_val)
                elif method == 'HARR-M':
                    # HARR-M的消融前/后数据
                    pre_val = data[ds_name][method][f'{metric.lower()}_d']
                    post_val = data[ds_name][method][f'{metric.lower()}_nd']
                    pre_ablation.append(pre_val)
                    post_ablation.append(post_val)

            # 绘制双柱形图
            bars1 = ax.bar(x - bar_width / 2, pre_ablation, bar_width, label='Distinguish (Pre)', color=colors[0],
                           alpha=0.8)
            bars2 = ax.bar(x + bar_width / 2, post_ablation, bar_width, label='Not Distinguish (Post)', color=colors[1],
                           alpha=0.8)

            # 设置子图标题和标签
            ax.set_title(f'{ds_name} Dataset', fontsize=12)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)

            # 设置y轴范围（根据数据动态调整，确保可读性）
            ax.set_ylim(0, min(1.0, max(max(pre_ablation), max(post_ablation)) + 0.1))

            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # 只在第一个子图添加图例（避免重复）
            if metric_idx == 0 and ds_idx == 0:
                ax.legend(loc='upper right', fontsize=9)

    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # 保存图片（可选，建议保存为高分辨率格式）
    plt.savefig('../outputs/Figure8.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 1. 运行实验
    all_results = run_all_experiments()
    # 2. 绘制消融实验对比图（10个子图）
    plot_ablation_comparison(all_results)
