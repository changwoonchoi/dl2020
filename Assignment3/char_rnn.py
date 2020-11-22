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
        output = []
        for i in range(d.shape[0]):
            hidden_input = torch.cat((d[i], hidden), 1)
            hidden = self.linear_1(hidden_input)
            output = self.linear_2(hidden_input)
            output = self.softmax(output)
            '''
            output_temp = self.linear_2(hidden_input)
            output_temp = self.softmax(output_temp)
            output.append(output_temp)
            '''
        # output = torch.cat(output, dim=0)

        return output

