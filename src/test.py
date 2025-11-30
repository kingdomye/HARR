import numpy as np

from HARRDataset import HARRDataSet
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


haar_dataset = HARRDataSet()
DS_dataset = haar_dataset.get_DS()

X_raw = DS_dataset.iloc[:, 0:6]
y_raw_d1 = DS_dataset.iloc[:, 6]
y_raw_d2 = DS_dataset.iloc[:, 7]

y_true = np.where((y_raw_d1 == 'no') & (y_raw_d2 == 'no'), 0, 1)

DS_dataset = X_raw.copy()
DS_dataset[0] = (DS_dataset[0] - DS_dataset[0].min()) / (DS_dataset[0].max() - DS_dataset[0].min())
DS_dataset = pd.get_dummies(DS_dataset, columns=DS_dataset.columns[1: 6], dtype='int')
DS_dataset.columns = DS_dataset.columns.astype(str)

kmeans = KMeans(n_clusters=2, n_init=20)
y_pred = kmeans.fit_predict(DS_dataset)

ARI = adjusted_rand_score(y_true, y_pred)
print(f"ARI: {ARI:.4f}")
