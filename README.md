# Crypto_Semiconductor_Stock_Predictor

There are many claims that the value of semiconductor company stock is partially dependent on the value of cryptocurrencies, as these companies such as AMD and Nvidia create the chips used for mining these cryptocurrencies.  

The following repository is a time-series LSTM model that predicts the stock price of Nvidia from the time-series data of its own stock, but of popular cryptocurrencies as well.

## Results

![Train-results](https://github.com/azkung/Crypto_Semiconductor_Stock_Predictor/blob/main/results/train_scaled.png)

![Test-results](https://github.com/azkung/Crypto_Semiconductor_Stock_Predictor/blob/main/results/test_scaled.png)

![Predictions](https://github.com/azkung/Crypto_Semiconductor_Stock_Predictor/blob/main/results/test_unscaled.png)

![Simulation](https://github.com/azkung/Crypto_Semiconductor_Stock_Predictor/blob/main/results/simulation.png)

### Simulation Statistics

- Starting Principal: $1000
- Final Value: $2331.39
- Total Return: 133.14%
- Time Period: 186 Business Days (Start of 2023 to the middle of 2023)

## Installation

```bash
pip install -r requirements.txt
```

Must perform a local installation of PyTorch

## Usage

Edit the config file to change hyperparameters. Also, add the Nvidia and crypto data to the datasets folder. The format script uses the yahoo finance CSV format.

```bash
python format.py
python train.py
python test.py
python simulation.py
```
