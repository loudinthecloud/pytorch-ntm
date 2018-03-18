"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


class FFWController(nn.Module):
    """An NTM controller based on FeedFoward Network."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(FFWController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.ffw = FFW(input_size=num_inputs,
                       hidden_size=num_outputs,
                       num_layers=num_layers)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        return None, None

    def reset_parameters(self):
        for p in self.ffw.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                nn.init.xavier_uniform(p, 1)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp = self.ffw(x)
        return outp.squeeze(0), None


# Neural Network Model
class FFW(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FFW, self).__init__()
        self.fc = nn.Sequential()
        for ii in range(num_layers-1):
            self.fc.add_module(str(ii), nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size

        self.fc.add_module('last', nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh()))

    def forward(self, x):
        return self.fc(x)
