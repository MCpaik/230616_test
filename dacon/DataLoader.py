import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, target, transform, is_test=False):
        self.df = df
        self.target = target # HLM or MLM
        self.is_test = is_test # train,valid / test

        self.feature_select = transform
        if not self.is_test: 
            self.fp = self.feature_select.fit_transform(np.stack(df['FPs']))
        else: # valid or test
            self.fp = self.feature_select.transform(np.stack(df['FPs']))

    def __getitem__(self, index):
        
        if not self.is_test: # test가 아닌 경우(label 존재)
            fp = self.fp[index]
            label = self.df[self.target][index]
            return torch.tensor(fp).float(), torch.tensor(label).float().unsqueeze(dim=-1) # feature, label

        else: # test인 경우
            fp = self.fp[index]
            return torch.tensor(fp).float() # feature
        
    def __len__(self):
        return len(self.df)

class ReCustomDataset(Dataset):
    def __init__(self, df, target, transform, is_test=False):
        self.df = df
        self.target = target # HLM or MLM
        self.is_test = is_test # train,valid / test
        self.AlogP_list = np.stack(df['AlogP']).reshape(len(np.stack(df['AlogP'])), 1)
        self.MW_list = np.stack(df['Molecular_Weight']).reshape(len(np.stack(df['Molecular_Weight'])), 1)
        self.NHA_list = np.stack(df['Num_H_Acceptors']).reshape(len(np.stack(df['Num_H_Acceptors'])), 1)
        self.NHD_list = np.stack(df['Num_H_Donors']).reshape(len(np.stack(df['Num_H_Donors'])), 1)
        self.NRB_list = np.stack(df['Num_RotatableBonds']).reshape(len(np.stack(df['Num_RotatableBonds'])), 1)
        self.MP_list = np.stack(df['Molecular_PolarSurfaceArea']).reshape(len(np.stack(df['Molecular_PolarSurfaceArea'])), 1)

        self.feature_select = transform
        if not self.is_test:
            self.FP = self.feature_select.fit_transform(np.stack(df['FPs']))
            self.AP = self.feature_select.fit_transform(self.AlogP_list)
            self.MW = self.feature_select.fit_transform(self.MW_list)
            self.NHA = self.feature_select.fit_transform(self.NHA_list)
            self.NHD = self.feature_select.fit_transform(self.NHD_list)
            self.NRB = self.feature_select.fit_transform(self.NRB_list)
            self.MP = self.feature_select.fit_transform(self.MP_list)

        else: # valid or test
            self.fp = self.feature_select.transform(np.stack(df['FPs']))

    def __getitem__(self, index):
        
        if not self.is_test: # test가 아닌 경우(label 존재)
            # fp = self.FP[index]
            fp = np.block([self.FP[index], self.AP[index], self.MW[index], self.NHA[index], self.NHD[index], self.NRB[index], self.MP[index]])
            # print(fp)
            label = self.df[self.target][index]
            # print(torch.tensor(fp).float(), torch.tensor(label).float().unsqueeze(dim=-1))
            return torch.tensor(fp).float(), torch.tensor(label).float().unsqueeze(dim=-1) # feature, label

        else: # test인 경우
            fp = self.fp[index]
            return torch.tensor(fp).float() # feature
        
    def __len__(self):
        return len(self.df)
    
