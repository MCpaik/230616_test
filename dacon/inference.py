from DataLoader import CustomDataset
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
import model
import numpy as np
from rdkit.Chem import PandasTools, AllChem
from rdkit import DataStructs
from torch.utils.data import DataLoader
import random
import os
import torch.nn as nn

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
test["FPs"] = test.Molecule.apply(mol2fp)
# train = train[['FPs','MLM', 'HLM']]
test = test[['FPs']]

transform = VarianceThreshold(threshold=0.05)

train_MLM = CustomDataset(df=train, target='MLM', transform=transform, is_test=False)
train_HLM = CustomDataset(df=train, target='HLM', transform=transform, is_test=False)

test_MLM = CustomDataset(df=test, target=None, transform=transform, is_test=True)
test_HLM = CustomDataset(df=test, target=None, transform=transform, is_test=True)

test_MLM_loader = DataLoader(dataset=test_MLM,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=False)

test_HLM_loader = DataLoader(dataset=test_HLM,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=False)

def inference(test_loader, model):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for inputs in test_loader:
            output = model(inputs)
            preds.extend(output.cpu().numpy().flatten().tolist())
    
    return preds


model_MLM = model.Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['DROPOUT_RATE'], config['OUTPUT_SIZE'])
model_HLM = model.Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['DROPOUT_RATE'], config['OUTPUT_SIZE'])

criterion = nn.MSELoss()
optimizer_MLM = torch.optim.Adam(model_MLM.parameters(), lr=config['LEARNING_RATE'])
optimizer_HLM = torch.optim.Adam(model_HLM.parameters(), lr=config['LEARNING_RATE'])

# model_MLM.load_state_dict(torch.load('./weights/230831_1825_MLM.pth', map_location='cpu'))
# model_HLM.load_state_dict(torch.load('./weights/230831_1825_HLM.pth', map_location='cpu'))
model_MLM = torch.load('./weights/230831_1825_MLM.pth', map_location='cpu')
model_HLM = torch.load('./weights/230831_1825_HLM.pth', map_location='cpu')
# model_MLM.load_state_dict(torch.load('./dacon/weights/model_MLM.pth', map_location='cpu'))
# model_HLM.load_state_dict(torch.load('./dacon/weights/230828_HLM.pth', map_location='cpu'))
predictions_MLM = inference(test_MLM_loader, model_MLM)
predictions_HLM = inference(test_HLM_loader, model_HLM)
submission = pd.read_csv('./result/sample_submission.csv')

submission['MLM'] = predictions_MLM
submission['HLM'] = predictions_HLM

submission.to_csv('./result/230831_submission.csv', index=False)
print('finish')
# submission
