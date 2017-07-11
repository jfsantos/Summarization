import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.encoder = getattr(nn, rnn_type+'Cell')(ninp, nhid)
            self.decoder = getattr(nn, rnn_type+'Cell')(ninp, nhid)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.encoder = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)
            self.decoder = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def set_mask(self, inputmask):
        self.inputmask = inputmask.float()[:, :, None]

    def forward(self, input):
        emb = self.drop(input)
        hidden = self.init_hidden(input.size(1))
        output_encoder = []
        for i in range(input.size(0)):
            wemb = emb[i]
            new_hidden = self.encoder(wemb, hidden)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0] * self.inputmask[i] + hidden[0] * (1 - self.inputmask[i])
                c = new_hidden[1] * self.inputmask[i] + hidden[1] * (1 - self.inputmask[i])
                output_encoder.append(h)
                new_hidden = (h,c)
            else:
                new_hidden = new_hidden * self.inputmask[i] + hidden * (1 - self.inputmask[i])
                output_encoder.append(new_hidden)
            hidden = new_hidden
        output_encoder = torch.stack(output_encoder, dim=0)

        output_decoder = []
        for i in range(input.size(0)):
            wemb = emb[i]
            new_hidden = self.decoder(wemb, hidden)
            if self.rnn_type == 'LSTM':
                output_decoder.append(new_hidden[0])
            else:
                output_decoder.append(new_hidden)
            hidden = new_hidden
        output_decoder = torch.stack(output_decoder, dim=0)

        return output_decoder.view(output_decoder.size(0)*output_decoder.size(1), output_decoder.size(2))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())