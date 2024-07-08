import torch
import torch.nn as nn
import numpy as np


class LSTM_Hedging(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(LSTM_Hedging, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        for weight in self.lstm.parameters():
            nn.init.normal_(weight, mean=0.0, std=1/np.sqrt(hidden_size))

    def forward(self, data, PnL, price_loc):
        batch_size, seq_size, input_size = data.shape
        input_size += 1
        h_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size))
        delta, Hedge_PnL = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
        delta_his = torch.reshape(delta.clone(), (batch_size, 1, 1))
        for t in range(seq_size):
            price = torch.reshape(data[:, t, price_loc], (batch_size, 1))
            data_sec = torch.cat([data[:, t, :], delta], dim=1)
            data_sec = torch.reshape(data_sec, (batch_size, 1, input_size))
            lstm_out, h_state = self.lstm(data_sec, h_state)
            delta = self.fc(lstm_out)
            delta = torch.reshape(delta, (batch_size, 1))
            Hedge_PnL += (delta_his[:, -1, :]-delta)*price
            delta_his = torch.cat((delta_his, torch.reshape(delta, (batch_size, 1, 1))), dim=1)
        Hedge_PnL += delta_his[:, -1, :]*torch.reshape(data[:, -1, 0], (batch_size, 1))
        TotalPnL = Hedge_PnL+torch.reshape(PnL, (batch_size, 1))
        return delta_his, TotalPnL

    def test(self, sec_data, h_state):
        lstm_out, h_state = self.lstm(sec_data, h_state)
        delta = self.fc(lstm_out)
        return delta, h_state


class GRU_Hedging(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(GRU_Hedging, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        for weight in self.gru.parameters():
            nn.init.normal_(weight, mean=0.0, std=1/np.sqrt(hidden_size))

    def forward(self, data, PnL, price_loc):
        batch_size, seq_size, input_size = data.shape
        input_size += 1
        h_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size))
        delta, Hedge_PnL = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
        delta_his = torch.reshape(delta.clone(), (batch_size, 1, 1))
        for t in range(seq_size):
            price = torch.reshape(data[:, t, price_loc], (batch_size, 1))
            data_sec = torch.cat([data[:, t, :], delta], dim=1)
            data_sec = torch.reshape(data_sec, (batch_size, 1, input_size))
            gru_out, h_state = self.gru(data_sec, h_state)
            delta = self.fc(gru_out)
            delta = torch.reshape(delta, (batch_size, 1))
            Hedge_PnL += (delta_his[:, -1, :]-delta)*price
            delta_his = torch.cat((delta_his, torch.reshape(delta, (batch_size, 1, 1))), dim=1)
        Hedge_PnL += delta_his[:, -1, :]*torch.reshape(data[:, -1, 0], (batch_size, 1))
        TotalPnL = Hedge_PnL+torch.reshape(PnL, (batch_size, 1))
        return delta_his, TotalPnL

    def test(self, sec_data, h_state):
        gru_out, h_state = self.gru(sec_data, h_state)
        delta = self.fc(gru_out)
        return delta, h_state