import random
import os

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from rdkit import DataStructs
from rdkit.Chem import PandasTools, AllChem

import model
from DataLoader import CustomDataset, ReCustomDataset

import yaml

with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

seed_everything(42) 

train = pd.read_csv('./csv_file/train.csv')
test = pd.read_csv('./csv_file/test.csv')

PandasTools.AddMoleculeColumnToFrame(train,'SMILES','Molecule')
PandasTools.AddMoleculeColumnToFrame(test,'SMILES','Molecule')

def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 6, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar

train["FPs"] = train.Molecule.apply(mol2fp)
# test["FPs"] = test.Molecule.apply(mol2fp)

# train = train[['FPs','MLM', 'HLM']]
# test = test[['FPs']]

# transform = transforms.Compose([VarianceThreshold(threshold=0.05),
#                                 transforms.Normalize])
transform = VarianceThreshold(threshold=0.05)
train_MLM = CustomDataset(df=train, target='MLM', transform=transform, is_test=False)
train_HLM = CustomDataset(df=train, target='HLM', transform=transform, is_test=False)

# input_size = train_MLM.fp.shape[1]

# CFG = {'BATCH_SIZE': 256,
#        'EPOCHS': 1000,
#        'INPUT_SIZE': input_size,
#        'HIDDEN_SIZE': 1024,
#        'OUTPUT_SIZE': 1,
#        'DROPOUT_RATE': 0.8,
#        'LEARNING_RATE': 0.001}

train_MLM_dataset, valid_MLM_dataset = train_test_split(train_MLM, test_size=0.2, random_state=42)
train_HLM_dataset, valid_HLM_dataset = train_test_split(train_HLM, test_size=0.2, random_state=42)

train_MLM_loader = DataLoader(dataset=train_MLM_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=True)

valid_MLM_loader = DataLoader(dataset=valid_MLM_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=False)


train_HLM_loader = DataLoader(dataset=train_HLM_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=True)

valid_HLM_loader = DataLoader(dataset=valid_HLM_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=False)

model_MLM = model.Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['DROPOUT_RATE'], config['OUTPUT_SIZE'])
model_HLM = model.Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['DROPOUT_RATE'], config['OUTPUT_SIZE'])

criterion = nn.MSELoss()
optimizer_MLM = torch.optim.Adam(model_MLM.parameters(), lr=config['LEARNING_RATE'])
optimizer_HLM = torch.optim.Adam(model_HLM.parameters(), lr=config['LEARNING_RATE'])

def train(train_loader, valid_loader, model, criterion, optimizer, epochs, train_type):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, targets in train_loader:
            # print(input)
            # torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            output = model(inputs)
            # output = torch.nan_to_num(output)
            loss = criterion(output, targets)
            # loss = torch.nan_to_num(loss)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if epoch % 100 == 0:
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    output = model(inputs)
                    # output = torch.nan_to_num(output)
                    loss = criterion(output, targets)
                    # loss = torch.nan_to_num(loss)

                    valid_loss += loss.item()
                    
            print(f'Epoch: {epoch}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Valid Loss: {valid_loss/len(valid_HLM_loader)}')
            
            model.train()
    if train_type == 'MLM':
        torch.save(model_MLM, './weights/230831_1825_MLM.pth')
    else:
        torch.save(model_HLM, './weights/230831_1825_HLM.pth')

    return model

print("Training Start: MLM")
model_MLM = train(train_MLM_loader, valid_MLM_loader, model_MLM, criterion, optimizer_MLM, epochs=config['EPOCHS'], train_type='MLM')

print("Training Start: HLM")
model_HLM = train(train_HLM_loader, valid_HLM_loader, model_HLM, criterion, optimizer_HLM, epochs=config['EPOCHS'], train_type='HLM')  

print('End training')

