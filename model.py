import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM1(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.linear1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.leakyrelu = nn.LeakyReLU() 
        self.linear2 = nn.Linear(int(hidden_size/2), 1)
    
    def forward(self, input_seq1):
        h0 = torch.zeros(self.num_layers, input_seq1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_seq1.size(0), self.hidden_size).to(device)

        lstm_out1, _ = self.lstm1(input_seq1, (h0, c0))

        lstm_out = lstm_out1[:, -1, :]

        predictions = self.linear1(lstm_out)
        predictions = self.leakyrelu(predictions)
        predictions = self.linear2(predictions)
        
        
        return predictions


class LSTM2(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU() 
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, input_seq1, input_seq2):
        h0 = torch.zeros(self.num_layers, input_seq1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input_seq1.size(0), self.hidden_size).to(device)

        h1 = torch.zeros(self.num_layers, input_seq2.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(self.num_layers, input_seq2.size(0), self.hidden_size).to(device)

        lstm_out1, _ = self.lstm1(input_seq1, (h0, c0))
        lstm_out2, _ = self.lstm2(input_seq2, (h1, c1))
        
        lstm_out = torch.cat((lstm_out1[:, -1, :], lstm_out2[:, -1, :]), dim=1)

        predictions = self.linear1(lstm_out)
        predictions = self.leakyrelu(predictions)
        predictions = self.linear2(predictions)
        
        return predictions
