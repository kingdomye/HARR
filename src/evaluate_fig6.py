import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置项
# ==========================================
# 12个方法，顺序固定
METHODS_ORDER = [
    'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
    'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
    'HARR-V', 'HARR-M'
]

# 12个X轴标签
METHODS_ALIAS = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'H-V', 'H-M'
]

FILE_PATHS = {
    'T3': '../outputs/Table3_Result_v3_调整版.csv',  # ARI Mixed
    'T4': '../outputs/Table4_Result_v1(HARR_V3结果).csv',  # ARI Categorical
    'T5': '../outputs/Table5_Result_CA（HARR_V3结果）.csv',  # CA Mixed
    'T6': '../outputs/Table6_Result_CA(HARR_V3结果).csv'  # CA Categorical
}

# 配色
COLOR_BLUE = '#4472C4'  # 性能
COLOR_ORANGE = '#ED7D31'  # 排名


# ==========================================
# 2. 数据处理函数
# ==========================================
def parse_value(cell):
    """提取均值"""
    if pd.isna(cell) or cell in ['N/A', '-', 'Missing']: return np.nan
    if isinstance(cell, (float, int)): return float(cell)
    try:
        return float(str(cell).split('±')[0].strip())
    except:
        return np.nan


def load_and_merge(path_mixed, path_cat):
    """读取并合并数据"""
    if not os.path.exists(path_mixed) or not os.path.exists(path_cat):
        print(f"警告：找不到文件 {path_mixed} 或 {path_cat}")
        return None

    # 使用 .map 避免警告
    df_mixed = pd.read_csv(path_mixed, index_col=0).map(parse_value)
    df_cat = pd.read_csv(path_cat, index_col=0).map(parse_value)

    common_methods = [m for m in METHODS_ORDER if m in df_mixed.index and m in df_cat.index]
    df_all = pd.concat([df_mixed.loc[common_methods], df_cat.loc[common_methods]], axis=1)
    return df_all


def calculate_stats(df):
    if df is None: return None, None
    avg_perf = df.mean(axis=1)
    # 排名: 数值越小越好 (1即第一名)
    ranks = df.rank(ascending=False, method='min', axis=0)
    avg_rank = ranks.mean(axis=1)
    return avg_perf, avg_rank


# ==========================================
# 3. 单个柱状图绘制函数
# ==========================================
def plot_single_bar(ax, data, title, y_label, color, is_rank=False):
    """
    绘制单个柱状图
    """
    x = np.arange(len(METHODS_ORDER))
    width = 0.6  # 柱宽

    # 绘制柱子
    bars = ax.bar(x, data, width, color=color, edgecolor='black', linewidth=0.6, alpha=0.9)

    # 设置Y轴标签
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)

    # 如果是排名图，设置合适的Y轴范围 (0-13)
    if is_rank:
        ax.set_ylim(0, 13)
        ax.set_yticks(np.arange(0, 14, 2))
    else:
        # 性能图范围 (0-1.0)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    # 设置X轴
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS_ALIAS, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.8, len(METHODS_ORDER) - 0.2)

    # 标题 (放在图下方)
    ax.set_title(title, y=-0.25, fontsize=12, fontweight='bold')

    # 网格线
    ax.grid(axis='y', linestyle='--', alpha=0.4)


# ==========================================
# 4. 主程序
# ==========================================
def generate_figure6_4subplots():
    print("正在读取数据...")
    df_ari = load_and_merge(FILE_PATHS['T3'], FILE_PATHS['T4'])
    df_ca = load_and_merge(FILE_PATHS['T5'], FILE_PATHS['T6'])

    if df_ari is None or df_ca is None:
        return

    # 计算
    ari_perf, ari_rank = calculate_stats(df_ari)
    ca_perf, ca_rank = calculate_stats(df_ca)

    # 对齐
    ari_perf = ari_perf.reindex(METHODS_ORDER)
    ari_rank = ari_rank.reindex(METHODS_ORDER)
    ca_perf = ca_perf.reindex(METHODS_ORDER)
    ca_rank = ca_rank.reindex(METHODS_ORDER)

    print("正在绘图 (4 Subplots)...")

    # 创建 2x2 的画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- 第一行: ARI ---
    # 子图 1: ARI 性能 (蓝色)
    plot_single_bar(axes[0, 0], ari_perf,
                    "(a) Average ARI Performance",
                    "Ave. ARI", COLOR_BLUE, is_rank=False)

    # 子图 2: ARI 排名 (橙色)
    plot_single_bar(axes[0, 1], ari_rank,
                    "(b) Average Rank (w.r.t ARI)",
                    "Ave. Rank", COLOR_ORANGE, is_rank=True)

    # --- 第二行: CA ---
    # 子图 3: CA 性能 (蓝色)
    plot_single_bar(axes[1, 0], ca_perf,
                    "(c) Average CA Performance",
                    "Ave. CA", COLOR_BLUE, is_rank=False)

    # 子图 4: CA 排名 (橙色)
    plot_single_bar(axes[1, 1], ca_rank,
                    "(d) Average Rank (w.r.t CA)",
                    "Ave. Rank", COLOR_ORANGE, is_rank=True)

    # 调整布局
    plt.tight_layout()
    # 增加垂直间距，防止标题重叠
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    output_path = '../outputs/Figure6.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已生成并保存至: {output_path}")
    plt.show()


if __name__ == '__main__':
    generate_figure6_4subplots()
