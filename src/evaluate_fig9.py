from HARR_v3 import HARR
from HARRDataset import HARRDataSet
import matplotlib.pyplot as plt

harr_dataset = HARRDataSet()
target_dataset_names = ['DS', 'HF', 'AA', 'AP', 'DT', 'AC']

all_Z_v = {key: [] for key in target_dataset_names}
all_Z_m = {key: [] for key in target_dataset_names}


for dataset_name in target_dataset_names:
    print(f'Processing dataset {dataset_name}')
    dataset, columns = harr_dataset.get_dataset_by_name(dataset_name)
    model = HARR(
        dataset,
        n_clusters=2,
        numerical_cols=columns.numerical_columns,
        nominal_cols=columns.nominal_columns,
        ordinal_cols=columns.ordinal_columns
    )
    model.preprocess()
    Q_v, weights_v, Q_history_v, W_history_v, Z_history_v = model.fitV_with_history()
    Q_m, weights_m, Q_history_m, W_history_m, Z_history_m = model.fitM_with_history()

    all_Z_v[dataset_name] = Z_history_v
    all_Z_m[dataset_name] = Z_history_m

# 绘制Z- history，对于六个数据集绘制六个子图
# 每个子图包含V和M的Z_history
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i, dataset_name in enumerate(target_dataset_names):
    axs[i].plot(all_Z_v[dataset_name], label='HARR-V')
    axs[i].plot(all_Z_m[dataset_name], label='HARR-M')
    axs[i].set_xlabel(' Number of Iterations')
    axs[i].set_ylabel('Z')
    axs[i].set_title(f'Dataset {dataset_name}')
    axs[i].legend()
plt.show()

# 保存
fig.savefig('../outputs/Figure9.png')
