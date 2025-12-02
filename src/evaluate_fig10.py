import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

from HARR_v3 import HARR
from OTHERS import Wrapper_CMS, Wrapper_UDM, Wrapper_GBD, Wrapper_FBD


ColumnsInfo = namedtuple('ColumnsInfo', ['numerical_columns', 'categorical_columns'])


def generate_synthetic_data(n_samples=10000, n_num=5, n_cat=5, n_values=5, random_state=42):
    """生成混合属性合成数据"""
    np.random.seed(random_state)

    # 生成数值属性
    X_num = np.random.rand(n_samples, n_num)
    num_cols = [f'Num_{i}' for i in range(n_num)]
    df_num = pd.DataFrame(X_num, columns=num_cols)

    # 生成分类属性 (0 到 v-1)
    X_cat = np.random.randint(0, n_values, size=(n_samples, n_cat))
    cat_cols = [f'Cat_{i}' for i in range(n_cat)]
    df_cat = pd.DataFrame(X_cat, columns=cat_cols)

    # 合并
    df = pd.concat([df_num, df_cat], axis=1)
    return df, num_cols, cat_cols


def run_efficiency_experiment():
    TOTAL_N = 10000

    SAMPLING_RATES = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]  # 去掉了 0.001，避免样本过少
    N_CLUSTERS = 5

    print(f"Generating synthetic dataset with N={TOTAL_N}...")
    df_full, num_cols, cat_cols = generate_synthetic_data(n_samples=TOTAL_N)

    # 构建 Wrapper 需要的 columns 对象
    cols_info = ColumnsInfo(numerical_columns=num_cols, categorical_columns=cat_cols)

    # 定义要比较的方法
    # 字典格式: '名称': 类名
    # 注意：这里假设你之前提供的 Wrapper 类都在当前作用域可用
    methods = {
        'HARR-V': 'HARR-V',
        'HARR-M': 'HARR-M',
        'CMS': Wrapper_CMS,
        'UDM': Wrapper_UDM,
        'GBD': Wrapper_GBD,
        'FBD': Wrapper_FBD,
        # 'K-Prototypes': Wrapper_KPrototypes, # 耗时较长，可选
        # 'SBC': Wrapper_SBC                 # 可选
    }

    results = {name: {'x': [], 'y': []} for name in methods.keys()}

    print("-" * 80)
    print(f"{'Rate':<8} | {'Samples':<10} | {'Method':<15} | {'Time (s)':<10}")
    print("-" * 80)

    for rate in SAMPLING_RATES:
        n_samples = int(TOTAL_N * rate)

        # 【关键修复】安全检查
        if n_samples < N_CLUSTERS:
            print(f"{rate:<8} | {n_samples:<10} | SKIP            | Too few samples")
            continue

        # 采样数据
        df_sample = df_full.iloc[:n_samples].copy().reset_index(drop=True)

        for name, algo_cls in methods.items():
            start_time = time.time()

            try:
                if name == 'HARR-V':
                    # HARR 特殊调用
                    model = HARR(df_sample, n_clusters=N_CLUSTERS,
                                 numerical_cols=num_cols, nominal_cols=cat_cols, ordinal_cols=[])
                    model.preprocess()
                    model.fitV(max_iter=15)

                elif name == 'HARR-M':
                    # HARR 特殊调用
                    model = HARR(df_sample, n_clusters=N_CLUSTERS,
                                 numerical_cols=num_cols, nominal_cols=cat_cols, ordinal_cols=[])
                    model.preprocess()
                    model.fitM(max_iter=15)

                else:
                    # 其他 Wrapper 调用
                    # 实例化
                    model = algo_cls(df_sample, cols_info, n_clusters=N_CLUSTERS)
                    # 训练
                    model.fit()

                end_time = time.time()
                elapsed = end_time - start_time

                results[name]['x'].append(n_samples)
                results[name]['y'].append(elapsed)

                print(f"{rate:<8} | {n_samples:<10} | {name:<15} | {elapsed:.4f}")

            except Exception as e:
                print(f"{rate:<8} | {n_samples:<10} | {name:<15} | FAILED: {str(e)[:20]}...")

    return results


def plot_results(results):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 颜色和标记映射
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = sns.color_palette("tab10", n_colors=len(results))

    for i, (name, data) in enumerate(results.items()):
        if not data['x']: continue

        # 加粗显示 HARR
        lw = 3 if 'HARR' in name else 1.5
        alpha = 1.0 if 'HARR' in name else 0.7

        plt.plot(
            data['x'], data['y'],
            marker=markers[i % len(markers)],
            linewidth=lw,
            alpha=alpha,
            label=name,
            color=colors[i]
        )

    plt.xlabel('Number of Objects (n)', fontsize=12)
    plt.ylabel('Execution time (in seconds)', fontsize=12)
    plt.ylim(0, 40)
    plt.title('Comparison of Efficiency (Figure 10 Reproduction)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.savefig('../outputs/Figure10.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # 确保你的 Wrapper 类和 HARR 类在当前上下文中可用
    # 这里直接调用函数
    exp_results = run_efficiency_experiment()
    plot_results(exp_results)
