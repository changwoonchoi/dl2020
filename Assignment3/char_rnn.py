import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        ##### please implment here
        

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, d, hidden):
        
        ##### please implment here
        
        output = self.softmax(output)
        return output

