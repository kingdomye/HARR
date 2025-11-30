from sklearn.metrics import adjusted_rand_score

from HARRDataset import HARRDataSet
from HARR import HARR
from src.OTHERS import *

harr = HARRDataSet()

all_data = harr.get_all_data_1()
ARI_scores = {}

for i in range(2):
    for dataset_name, (dataset, columns) in all_data.items():
        # 定义分类器
        kmd = Wrapper_KPrototypes(dataset, columns)
        ohe = Wrapper_OHE_KMeans(dataset, columns)
        SBC = Wrapper_SBC(dataset, columns)
        jdm = Wrapper_JDM(dataset, columns)
        cms = Wrapper_CMS(dataset, columns)
        harrv = HARR(dataset)
        harrv.preprocess(columns.numerical_columns, columns.categorical_columns)

        # 分类
        kmd_labels, _ = kmd.fit()
        ohe_labels, _ = ohe.fit()
        SBC_labels, _ = SBC.fit()
        jdm_labels, _ = jdm.fit()
        cms_labels, _ = cms.fit()
        Q, w = harrv.fitM()

        # 计算ARI
        harrv_ari = adjusted_rand_score(columns.y_true, Q)
        kmd_ari = adjusted_rand_score(columns.y_true, kmd_labels)
        ohe_ari = adjusted_rand_score(columns.y_true, ohe_labels)
        SBC_ari = adjusted_rand_score(columns.y_true, SBC_labels)
        jdm_ari = adjusted_rand_score(columns.y_true, jdm_labels)
        cms_ari = adjusted_rand_score(columns.y_true, cms_labels)

        print(f'{dataset_name} KMD ARI: {kmd_ari}')
        print(f'{dataset_name} OHE+OC ARI: {ohe_ari}')
        print(f'{dataset_name} SBC ARI: {SBC_ari}')
        print(f'{dataset_name} JDM ARI: {jdm_ari}')
        print(f'{dataset_name} CMS ARI: {cms_ari}')
        print(f'{dataset_name} HARR_V ARI: {harrv_ari}')
        print("=====")

print(ARI_scores)
