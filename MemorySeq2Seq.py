import torch
import torch.nn as nn
from torch.autograd import Variable

class Attention(nn.Module):

    def __init__(self, ndim):
        super(Attention, self).__init__()
        self.lin = nn.Linear(ndim, ndim, bias=False)
        self.softmax = nn.Softmax()
        self.dist = None

    def forward(self, input, memory):
        t = self.lin(input)
        e = torch.bmm(memory, t[:, :, None]).squeeze(2)
        p = self.softmax(e)
        self.dist = p
        output = torch.bmm(p[:, None, :], memory).squeeze(1)

        return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, max_nslots=3):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.encoder_forward = getattr(nn, rnn_type + 'Cell')(ninp, nhid)
            self.encoder_backward = getattr(nn, rnn_type + 'Cell')(ninp, nhid)
            self.recall = getattr(nn, rnn_type + 'Cell')(ninp, nhid)
            self.decoder = getattr(nn, rnn_type + 'Cell')(ninp, nhid)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.encoder = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)
            self.recall = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)
            self.decoder = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)

        self.write_attention = Attention(nhid)
        self.read_attention = Attention(nhid)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.max_nslots = max_nslots

    def set_mask(self, inputmask):
        self.inputmask = inputmask.float()[:, :, None]

    def forward(self, input):
        bsz = input.size(1)
        emb = self.drop(input)

        hidden_forward = self.init_hidden(bsz)
        output_encoder_forward = []
        for i in range(input.size(0)):
            wemb = emb[i]
            new_hidden = self.encoder_forward(wemb, hidden_forward)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0] * self.inputmask[i] + hidden_forward[0] * (1 - self.inputmask[i])
                c = new_hidden[1] * self.inputmask[i] + hidden_forward[1] * (1 - self.inputmask[i])
                output_encoder_forward.append(h)
                new_hidden = (h, c)
            else:
                new_hidden = new_hidden * self.inputmask[i] + hidden_forward * (1 - self.inputmask[i])
                output_encoder_forward.append(new_hidden)
            hidden_forward = new_hidden
        output_encoder_forward = torch.stack(output_encoder_forward, dim=1)

        hidden_backward = self.init_hidden(bsz)
        output_encoder_backward = []
        for i in range(input.size(0)-1, -1, -1):
            wemb = emb[i]
            new_hidden = self.encoder_backward(wemb, hidden_backward)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0] * self.inputmask[i] + hidden_backward[0] * (1 - self.inputmask[i])
                c = new_hidden[1] * self.inputmask[i] + hidden_backward[1] * (1 - self.inputmask[i])
                output_encoder_backward.append(h)
                new_hidden = (h, c)
            else:
                new_hidden = new_hidden * self.inputmask[i] + hidden_backward * (1 - self.inputmask[i])
                output_encoder_backward.append(new_hidden)
            hidden_backward = new_hidden
        output_encoder_backward = torch.stack(output_encoder_backward[::-1], dim=1)

        if self.rnn_type == 'LSTM':
            hidden = (hidden_forward[0] + hidden_backward[0],
                      hidden_forward[1] + hidden_backward[1])
        else:
            hidden = hidden_forward + hidden_backward
        encoder_hidden = hidden

        output_encoder = output_encoder_forward + output_encoder_backward

        memory = []
        nslots = min(input.size(0) / 2, self.max_nslots)
        m = Variable(next(self.parameters()).data.new(bsz, self.nhid).zero_())
        for i in range(nslots):
            new_hidden = self.recall(m, hidden)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0]
            else:
                h = new_hidden
            m = self.write_attention(h, output_encoder)
            memory.append(m)
            hidden = new_hidden
        memory = torch.stack(memory, dim=1)

        output_decoder = []
        dist = []
        hidden = self.init_hidden(bsz)
        # hidden = encoder_hidden
        for i in range(input.size(0)):
            wemb = emb[i]
            new_hidden = self.decoder(wemb, hidden)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0]
            else:
                h = new_hidden
            m = self.read_attention(h, memory)
            dist.append(self.read_attention.dist)
            output_decoder.append(m + h)
            hidden = new_hidden
        output_decoder = torch.stack(output_decoder, dim=0)
        self.read_attention_dist = torch.stack(dist, dim=0)

        return output_decoder.view(output_decoder.size(0) * output_decoder.size(1), output_decoder.size(2))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(bsz, self.nhid).zero_())
