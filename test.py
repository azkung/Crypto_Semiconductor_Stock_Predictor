# Description: Test the model on the test dataset
# Written by: Alexander Kung

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LSTM2, LSTM1
import json
import matplotlib.pyplot as plt
import joblib
import numpy as np
from copy import deepcopy as dc
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('results'):
    os.makedirs('results')

with open('config.json') as f:
    config = json.load(f)

hidden_size = config['hidden_size']
num_layers = config['num_layers']

batch_size = config['batch_size']
n_steps = config['lookback']

model = LSTM2(1, hidden_size=hidden_size, num_layers=num_layers).to(device)

train_dataset = torch.load('formatted/train_dataset.pt')
test_dataset = torch.load('formatted/test_dataset.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

model.eval()
model.load_state_dict(torch.load('weights/best.pt'))

with torch.no_grad():
    for i, (X1, X2, y) in enumerate(train_loader):
        X1 = X1.reshape(-1, n_steps, 1).to(device)
        X2 = X2.reshape(-1, n_steps, 1).to(device)
        y_pred = model(X1, X2)
        y = y.reshape(-1, 1).to(device)

print(y_pred.shape)
print(y.shape)

y_pred = y_pred.cpu().numpy()
y = y.cpu().numpy()

y_pred = y_pred.reshape(-1, 1)
y = y.reshape(-1, 1)

plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Train')
plt.savefig('results/train_scaled.png')
plt.show()

loss_function = nn.MSELoss(reduction='mean')

with torch.no_grad():
    for i, (X1, X2, y) in enumerate(test_dataloader):
        X1 = X1.reshape(-1, n_steps, 1).to(device)
        X2 = X2.reshape(-1, n_steps, 1).to(device)
        y_pred = model(X1, X2)
        y = y.reshape(-1, 1).to(device)
        loss = loss_function(y_pred, y)


print(y_pred.shape)
print(y.shape)
print(loss)

y_pred = y_pred.cpu().numpy()
y = y.cpu().numpy()

y_pred = y_pred.reshape(-1, 1)
y = y.reshape(-1, 1)

plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Test')
plt.savefig('results/test_scaled.png')
plt.show()


print(y_pred.shape)
print(y.shape)

NVDA_Scaler = joblib.load('formatted/NVDA_Scaler.pkl')
# BTC_Scaler = joblib.load('scalers/BTC_Scaler.pkl')
# per_Scaler = joblib.load('formatted/per_Scaler.pkl')

lookback = config['lookback']


dummies = np.zeros((len(test_dataset), lookback+1))
dummies[:, 0] = y.flatten()
dummies = NVDA_Scaler.inverse_transform(dummies)

y = dc(dummies[:, 0])

dummies = np.zeros((len(test_dataset), lookback+1))
dummies[:, 0] = y_pred.flatten()
dummies = NVDA_Scaler.inverse_transform(dummies)

y_pred = dc(dummies[:, 0])

plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Test (Real Values)')

plt.savefig('results/test_unscaled.png')
plt.show()

