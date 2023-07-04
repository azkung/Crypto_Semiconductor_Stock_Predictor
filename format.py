import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import numpy as np
import joblib

def main():
    
    with open('config.json') as f:
        config = json.load(f)


    n_steps = config['lookback']
    train_percent = config['train_percent']


    df_BTC = pd.read_csv('datasets/BTC-USD.csv')
    df_NVDA = pd.read_csv('datasets/NVDA.csv')


    df_BTC.drop('Volume', axis=1, inplace=True)
    df_BTC.drop('Adj Close', axis=1, inplace=True)

    df_NVDA.drop('Volume', axis=1, inplace=True)
    df_NVDA.drop('Adj Close', axis=1, inplace=True)


    df = pd.merge(df_BTC, df_NVDA, on='Date', how='inner')


    df = df[['Date', 'Close_y', 'Close_x']]

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close_y(t-{i})'] = df['Close_y'].shift(i)
    
    for i in range(1, n_steps+1):
        df[f'Close_x(t-{i})'] = df['Close_x'].shift(i)

    df.dropna(inplace=True)

    df.drop('Close_x', axis=1, inplace=True)

    print(df.head())


    data = df.to_numpy()

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # data = scaler.fit_transform(data)

    NVDA = data[:, 0:n_steps+1]
    BTC = data[:, n_steps+1:]

    NVDA_Scaler = MinMaxScaler(feature_range=(-1, 1))
    BTC_Scaler = MinMaxScaler(feature_range=(-1, 1))

    NVDA = NVDA_Scaler.fit_transform(NVDA)
    BTC = BTC_Scaler.fit_transform(BTC)

    X1 = NVDA[:, 1:]
    X2 = BTC[:, 0:]
    y = NVDA[:, 0]

    print(X1.shape, X2.shape, y.shape)

    print(X1)
    print(X2)
    print(y)


    X1 = dc(np.flip(X1, axis=1))
    X2 = dc(np.flip(X2, axis=1))

    print(X1)
    print(X2)
    print(y)


    print(X1.shape, X2.shape, y.shape)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X1).float(), torch.from_numpy(X2).float(), torch.from_numpy(y).float())

    train_values = int(len(dataset)*train_percent)


    train_dataset = torch.utils.data.Subset(dataset, range(0, train_values))
    test_dataset = torch.utils.data.Subset(dataset, range(train_values, len(dataset)))



    # save datasets
    torch.save(train_dataset, 'formatted/train_dataset.pt')
    torch.save(test_dataset, 'formatted/test_dataset.pt')

    # save scaler
    joblib.dump(NVDA_Scaler, 'formatted/NVDA_Scaler.pkl') # NVDA
    joblib.dump(BTC_Scaler, 'formatted/BTC_Scaler.pkl') # BTC


if __name__ == '__main__':
    main()