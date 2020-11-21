import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        ##### please implment here
        self.linear_1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_2 = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, d, hidden):
        
        ##### please implment here
        print(d.shape)
        hidden = torch.cat(d.shape[0] * [hidden]).reshape(d.shape[0], 1, -1)
        print(hidden.shape)

        hidden_input = torch.cat((d, hidden), 2)
        hidden = self.linear_1(hidden_input)
        output = self.linear_2(hidden_input)
        output = self.softmax(output)
        return output

