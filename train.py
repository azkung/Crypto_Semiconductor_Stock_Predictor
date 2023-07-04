# Description: Trains the model
# Written by: Alexander Kung

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
from model import LSTM1, LSTM2
import os

def main():
    with open('config.json') as f:
        config = json.load(f)

    n_steps = config['lookback']
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    train_percent = config['train_percent']
    weights = config['weights']
    momentum = config['momentum']

    optimizer = config['optimizer']

    # check if weights folder exists
    if not os.path.exists('weights'):
        os.makedirs('weights')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load('formatted/train_dataset.pt')
    test_dataset = torch.load('formatted/test_dataset.pt')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)


    model = LSTM2(1, hidden_size=hidden_size, num_layers=num_layers)

    model.to(device)

    best_loss = float('inf')


    loss_function = nn.MSELoss(reduction='mean')

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        print('Invalid optimizer')
        exit()

    # test_loss_function = nn.L1Loss(reduction='sum')

    if(weights != ""):
        model.load_state_dict(torch.load(weights))
        with torch.no_grad():
            for i, (X1, X2, y) in enumerate(test_loader):
                X1 = X1.reshape(-1, n_steps, 1).to(device)
                X2 = X2.reshape(-1, n_steps, 1).to(device)
                y = y.reshape(-1, 1).to(device)
                y_pred = model(X1, X2)
                loss = loss_function(y_pred, y)
                best_loss = loss

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (X1, X2, y) in enumerate(train_loader):
            # reshape to (batch_size, seq, input_size)
            X1 = X1.reshape(-1, n_steps, 1).to(device)
            X2 = X2.reshape(-1, n_steps, 1).to(device)
            y_pred = model(X1, X2)
            y = y.reshape(-1, 1).to(device)
            
            loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'epoch: {epoch:3} loss: {running_loss/len(train_loader):10.8f}')

        torch.save(model.state_dict(), 'weights/last.pt')

        if(len(test_dataset) == 0):
            if(running_loss/len(train_loader) < best_loss):
                best_loss = running_loss/len(train_loader)
                torch.save(model.state_dict(), 'weights/best.pt')
                print('Saved model')
            continue

        # validation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for i, (X1, X2, y) in enumerate(test_loader):
                X1 = X1.reshape(-1, n_steps, 1).to(device)
                X2 = X2.reshape(-1, n_steps, 1).to(device)
                y_pred = model(X1, X2)
                y = y.reshape(-1, 1).to(device)
                test_loss += loss_function(y_pred, y)

            test_loss /= len(test_loader)
            print(f'Test loss: {test_loss:.8f}')
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), 'weights/best.pt')
                print('Saved model')


if __name__ == '__main__':
    main()
