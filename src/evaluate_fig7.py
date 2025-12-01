import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置与样式
# ==========================================
METHODS_ORDER = [
    'KMD/KPT', 'OHE+OC', 'SBC', 'JDM', 'CMS',
    'UDM', 'HOD', 'GWD', 'GBD', 'FBD',
    'HARR-V', 'HARR-M'
]

# 对应论文中的标记形状和颜色
MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', 'X', 'd']
# 使用 Tab20 颜色板，保证颜色区分度
COLORS = plt.cm.get_cmap('tab20', len(METHODS_ORDER))

FILE_PATHS = {
    'T3': '../outputs/Table3_Result_v3_调整版.csv',  # ARI Mixed
    'T4': '../outputs/Table4_Result_v1(HARR_V3结果).csv',  # ARI Categorical
    'T5': '../outputs/Table5_Result_CA（HARR_V3结果）.csv',  # CA Mixed
    'T6': '../outputs/Table6_Result_CA(HARR_V3结果).csv'  # CA Categorical
}

# 临界值
CD_95 = 3.8158
CD_90 = 3.4069


# ==========================================
# 2. 数据加载
# ==========================================
def parse_value(cell):
    if pd.isna(cell) or cell in ['N/A', '-', 'Missing']: return np.nan
    try:
        return float(str(cell).split('±')[0].strip())
    except:
        return np.nan


def load_and_merge(path_mixed, path_cat):
    if not os.path.exists(path_mixed) or not os.path.exists(path_cat):
        return None
    df_mixed = pd.read_csv(path_mixed, index_col=0).map(parse_value)
    df_cat = pd.read_csv(path_cat, index_col=0).map(parse_value)
    common = [m for m in METHODS_ORDER if m in df_mixed.index and m in df_cat.index]
    return pd.concat([df_mixed.loc[common], df_cat.loc[common]], axis=1)


def get_average_ranks(df):
    if df is None: return None
    ranks = df.rank(ascending=False, method='min', axis=0)
    return ranks.mean(axis=1)


# ==========================================
# 3. 精致版 CD 图绘制核心函数
# ==========================================
def plot_beautiful_bd(ax, ranks, title):
    # 1. 基础设置
    # Rank 越小越好，X轴通常显示 1 到 12
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1.2)  # 留出足够的高度给标签

    # 隐藏 Y 轴和四周的框
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 加粗底部 X 轴
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_position(('data', 0))  # X轴位置在 y=0

    # 设置 X 轴刻度
    ax.set_xticks(np.arange(1, 13))
    ax.tick_params(axis='x', labelsize=10, width=1.5, length=5)
    ax.set_xlabel("Average Rank", fontsize=12, fontweight='bold', labelpad=10)

    # 2. 准备数据
    sorted_ranks = ranks.sort_values()
    control_name = sorted_ranks.idxmin()  # 通常是 HARR-M
    control_rank = sorted_ranks.min()

    # 3. 绘制每个方法的 Marker 和 Label (错位排布)
    # 定义 4 个高度层级，防止文字重叠
    text_levels = [0.2, 0.4, 0.6, 0.8]

    for i, (method, rank) in enumerate(sorted_ranks.items()):
        orig_idx = METHODS_ORDER.index(method)
        marker = MARKERS[orig_idx]
        color = COLORS(orig_idx)

        # A. 在轴上画点
        ax.plot(rank, 0, marker=marker, color=color, markersize=9,
                markeredgecolor='black', markeredgewidth=0.8, zorder=10, clip_on=False)

        # B. 确定标签高度 (循环使用层级)
        # 逻辑：根据排序后的索引 i 决定高度，避免相邻的重叠
        level_idx = i % 4
        text_y = text_levels[level_idx]

        # C. 绘制引线 (从点到文字)
        ax.plot([rank, rank], [0, text_y], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # D. 绘制文字框
        # 如果是 Control Method (HARR-M)，加粗显示
        font_weight = 'bold' if method == control_name else 'normal'
        box_color = 'red' if method == control_name else 'black'

        ax.text(rank, text_y + 0.02, f"{method}\n{rank:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight=font_weight, color=box_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # 4. 绘制 CD 区间 (放在最上方，类似一把尺子)
    cd_y_pos = 1.05  # 放在图的最顶端

    # CD 95% 线
    limit_95 = control_rank + CD_95
    ax.hlines(y=cd_y_pos, xmin=control_rank, xmax=limit_95, color='#D62728', linewidth=2.5, zorder=5)
    # 两端竖线
    ax.plot([control_rank, control_rank], [cd_y_pos - 0.03, cd_y_pos + 0.03], color='#D62728', linewidth=2.5)
    ax.plot([limit_95, limit_95], [cd_y_pos - 0.03, cd_y_pos + 0.03], color='#D62728', linewidth=2.5)

    # CD 95% 文字
    mid_point = (control_rank + limit_95) / 2
    ax.text(mid_point, cd_y_pos - 0.08, f"CD (95%): {CD_95:.2f}",
            ha='center', va='top', color='#D62728', fontsize=10, fontweight='bold')

    # CD 90% 线 (虚线，画在 95% 下方一点点)
    limit_90 = control_rank + CD_90
    # 稍微错开一点高度，防止重叠
    ax.plot([limit_90, limit_90], [cd_y_pos - 0.03, cd_y_pos], color='#1F77B4', linewidth=2, linestyle=':')
    ax.text(limit_90, cd_y_pos + 0.02, "90%", ha='center', va='bottom', color='#1F77B4', fontsize=8)

    # 标题
    ax.set_title(title, y=-0.15, fontsize=13, fontweight='bold')


# ==========================================
# 4. 主程序
# ==========================================
def generate_figure7_beautiful():
    print("读取数据...")
    df_ari = load_and_merge(FILE_PATHS['T3'], FILE_PATHS['T4'])
    df_ca = load_and_merge(FILE_PATHS['T5'], FILE_PATHS['T6'])

    if df_ari is None or df_ca is None: return

    rank_ari = get_average_ranks(df_ari)
    rank_ca = get_average_ranks(df_ca)

    print("正在绘图 (Beautiful BD Test)...")

    # 调整画布大小，确保有足够的高度给分层标签
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    plot_beautiful_bd(ax1, rank_ari, "(a) BD Test on ARI results")
    plot_beautiful_bd(ax2, rank_ca, "(b) BD Test on CA results")

    plt.tight_layout()
    # 增加子图间距
    plt.subplots_adjust(hspace=0.3)

    output_path = '../outputs/Figure7_Beautiful.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    plt.show()


if __name__ == '__main__':
    generate_figure7_beautiful()