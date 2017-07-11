import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, nhid, ntoken, dropout=0.5):
        super(Softmax, self).__init__()
        
        self.decoder = nn.Linear(nhid, ntoken)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input):
        input = self.drop(input)
        decoded = self.decoder(input.view(-1, input.size(-1)))
        decoded = self.logsoftmax(decoded)
        return decoded