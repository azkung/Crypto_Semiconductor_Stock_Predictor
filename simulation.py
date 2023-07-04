# Description: This file is used to simulate the trading of the model on the test set.
# Trading is done by buying when the predicted price is higher than the previous day's 
# price and selling when the predicted price is lower than the previous day's price.
# The amount of money made is then calculated and plotted.
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('config.json') as f:
    config = json.load(f)

hidden_size = config['hidden_size']
num_layers = config['num_layers']


with open('config.json') as f:
    config = json.load(f)

batch_size = config['batch_size']
n_steps = config['lookback']

model = LSTM2(1, hidden_size=hidden_size, num_layers=num_layers).to(device)

train_dataset = torch.load('formatted/train_dataset.pt')
test_dataset = torch.load('formatted/test_dataset.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

model.eval()
model.load_state_dict(torch.load('weights/best.pt'))

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

principal = 1000
money = principal
money_history = [money]
percent_invested = 1

for i in range(1, len(y)):
    if y_pred[i] > y[i-1]:
        money += percent_invested*money*(y[i]-y[i-1])/y[i-1]
    money_history.append(money)


plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Test (Scaled)')
plt.show()

print(((money-principal)/principal)*100, '%')

plt.plot(money_history)
plt.title('Money History')
plt.show()
