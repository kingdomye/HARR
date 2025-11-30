from pickle import APPENDS

import pandas as pd
import numpy as np
import arff
from scipy.io import arff


class DataColumns:
    def __init__(self, numerical_columns, categorical_columns, y_true):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.y_true = y_true


class HARRDataSet:
    def __init__(self):
        self.dataset_path = '../datasets/'
        self.DS = None
        self.HF = None
        self.AA = None
        self.AP = None
        self.DT = None
        self.AC = None
        self.SB = None
        self.SF = None
        self.T3 = None
        self.HR = None
        self.LG = None
        self.MR = None
        self.LE = None
        self.SW = None

        self.all_data_1 = {}
        self.all_data_2 = {}

    def get_all_data_1(self):
        if self.all_data_1:
            return self.all_data_1

        self.all_data_1['DS'] = self.get_DS()
        self.all_data_1['HF'] = self.get_HF()
        self.all_data_1['AA'] = self.get_AA()
        self.all_data_1['AP'] = self.get_AP()
        self.all_data_1['DT'] = self.get_DT()
        self.all_data_1['AC'] = self.get_AC()

        return self.all_data_1

    def get_all_data_2(self):
        if self.all_data_2:
            return self.all_data_2

        self.all_data_2['SB'] = self.get_SB()
        self.all_data_2['SF'] = self.get_SF()
        self.all_data_2['T3'] = self.get_T3()
        self.all_data_2['HR'] = self.get_HR()
        self.all_data_2['LG'] = self.get_LG()
        self.all_data_2['MR'] = self.get_MR()
        self.all_data_2['LE'] = self.get_LE()
        self.all_data_2['SW'] = self.get_SW()

        return self.all_data_2

    def get_DS(self):
        DS_file = self.dataset_path + 'DS/diagnosis.data'
        DS = pd.read_csv(
            DS_file,
            header=None,
            sep="\t",
            encoding='utf-16',
            thousands=None
        )
        DS[0] = DS[0].astype(str).str.replace(',', '.').astype(float)

        numerical_columns = DS.columns[0]
        categorical_columns = DS.columns[1: 6]
        y1 = DS[DS.columns[6]]
        y2 = DS[DS.columns[7]]
        y_true = np.where((y1 == 'no') & (y2 == 'no'), 0, 1)

        DS_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.DS = DS

        return DS, DS_columns

    def get_HF(self):
        HF_file = self.dataset_path + 'HF/heart_failure_clinical_records_dataset.csv'
        HF = pd.read_csv(HF_file)

        numerical_columns = HF.columns[: 7]
        categorical_columns = HF.columns[7: 12]
        y_true = HF[HF.columns[12]]

        HF_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.HF = HF

        return HF, HF_columns

    def get_AA(self):
        AA_file = self.dataset_path + 'AA/Autism-Adolescent-Data.arff'
        data, meta = arff.loadarff(AA_file)
        AA = pd.DataFrame(data)

        object_columns = AA.select_dtypes(include=['object']).columns
        for col in object_columns:
            AA[col] = AA[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        numerical_cols = ['age', 'result']
        categorical_cols = [
            'gender', 'ethnicity', 'jundice', 'austim',
            'contry_of_res', 'used_app_before', 'relation'
        ]
        label_col = 'Class/ASD'

        AA = AA[numerical_cols + categorical_cols + [label_col]]

        numerical_columns = AA.columns[0: 2]
        categorical_columns = AA.columns[2: 9]
        y_true = AA[AA.columns[9]]
        AA_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.AA = AA

        return AA, AA_columns

    def get_AP(self):
        AP_file = self.dataset_path + 'AP/dataset.csv'

        AP = pd.read_csv(AP_file, sep=';', skiprows=1)

        numerical_cols_name = ['SR', 'NR']
        categorical_cols_name = [
            'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR',
            'OR', 'RR', 'BR', 'MR', 'CR'
        ]

        label_col_name = 'Green frogs'

        selected_cols = numerical_cols_name + categorical_cols_name + [label_col_name]
        AP = AP[selected_cols]

        numerical_columns = AP.columns[0: 2]
        categorical_columns = AP.columns[2: 14]
        y_true = AP[label_col_name]

        AP_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.AP = AP

        return AP, AP_columns

    def get_DT(self):
        DT_file = self.dataset_path + 'DT/dermatology.data'

        DT = pd.read_csv(DT_file, header=None, na_values='?')

        DT = DT.dropna().reset_index(drop=True)

        cols = list(DT.columns)

        new_col_order = [33] + list(range(33)) + [34]

        DT = DT[new_col_order]

        feature_names = ['Age'] + [f'Cat_{i}' for i in range(33)] + ['Class']
        DT.columns = feature_names

        numerical_columns = DT.columns[0: 1]
        categorical_columns = DT.columns[1: 34]
        y_true = DT[DT.columns[34]]

        DT_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.DT = DT

        return DT, DT_columns

    def get_AC(self):
        AC_file = self.dataset_path + 'AC/australian.dat'
        AC = pd.read_csv(AC_file, sep=' ', header=None)

        num_indices = [1, 2, 6, 9, 12, 13]
        cat_indices = [0, 3, 4, 5, 7, 8, 10, 11]
        label_index = [14]

        AC = AC[num_indices + cat_indices + label_index]

        numerical_columns = AC.columns[0: 6]
        categorical_columns = AC.columns[6: 14]
        y_true = AC[AC.columns[14]]

        AC_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.AC = AC

        return AC, AC_columns

    def get_SB(self):
        SB_file = self.dataset_path + 'SB/soybean-large.data'
        SB = pd.read_csv(SB_file, header=None)

        # 将 Class (第一列) 移到最后一列
        cols = list(SB.columns)
        new_cols = cols[1:] + [cols[0]]
        SB = SB[new_cols]

        numerical_columns = []
        categorical_columns = SB.columns[0: 35]
        y_true = SB[SB.columns[35]]

        SB_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.SB = SB

        return SB, SB_columns

    def get_SF(self):
        SF_file = self.dataset_path + 'SF/flare.data1'
        SF = pd.read_csv(SF_file, sep='\s+', skiprows=1, header=None)

        numerical_columns = []
        categorical_columns = SF.columns[0: 9]

        y_true = SF[SF.columns[10]]

        SF_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.SF = SF

        return SF, SF_columns

    def get_T3(self):
        T3_file = self.dataset_path + 'T3/tic-tac-toe.data'
        T3 = pd.read_csv(T3_file, header=None)

        numerical_columns = []
        categorical_columns = T3.columns[0: 9]
        y_true = T3[T3.columns[9]]

        T3_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.T3 = T3

        return T3, T3_columns

    def get_HR(self):
        HR_file = self.dataset_path + 'HR/hayes-roth.data'
        HR = pd.read_csv(HR_file, header=None)

        HR = HR.iloc[:, 1:6]

        numerical_columns = []
        categorical_columns = HR.columns[0: 4]
        y_true = HR[HR.columns[4]]

        HR_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.HR = HR

        return HR, HR_columns

    def get_LG(self):
        LG_file = self.dataset_path + 'LG/lymphography.data'
        LG = pd.read_csv(LG_file, header=None)

        cols = list(LG.columns)
        new_cols = cols[1:] + [cols[0]]
        LG = LG[new_cols]

        LG.columns = range(LG.shape[1])

        numerical_columns = []
        categorical_columns = LG.columns[0: 18]
        y_true = LG[LG.columns[18]]

        LG_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.LG = LG

        return LG, LG_columns

    def get_MR(self):
        MR_file = self.dataset_path + 'MR/agaricus-lepiota.data'
        MR = pd.read_csv(MR_file, header=None)

        if MR[16].nunique() == 1:
            MR = MR.drop(columns=[16])

        if 11 in MR.columns:
            MR = MR.drop(columns=[11])

        cols = list(MR.columns)
        new_cols = cols[1:] + [cols[0]]
        MR = MR[new_cols]

        MR.columns = range(MR.shape[1])

        numerical_columns = []
        categorical_columns = MR.columns[0: 20]
        y_true = MR[MR.columns[20]]

        MR_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.MR = MR

        return MR, MR_columns

    def get_LE(self):
        LE_file = self.dataset_path + 'LE/LEV.arff'

        with open(LE_file, 'r', encoding='cp1252') as f:
            data, meta = arff.loadarff(f)
        LE = pd.DataFrame(data)

        numerical_columns = []
        categorical_columns = LE.columns[0: 4]
        y_true = LE[LE.columns[4]]

        LE_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.LE = LE

        return LE, LE_columns

    def get_SW(self):
        SW_file = self.dataset_path + 'SW/SWD.arff'
        with open(SW_file, 'r', encoding='cp1252') as f:
            data, meta = arff.loadarff(f)

        SW = pd.DataFrame(data)

        numerical_columns = []
        categorical_columns = SW.columns[0: 10]
        y_true = SW[SW.columns[10]]

        SW_columns = DataColumns(numerical_columns, categorical_columns, y_true)

        self.SW = SW

        return SW, SW_columns


# 测试
if __name__ == '__main__':
    harr = HARRDataSet()
    print(harr.get_SW())
