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
    """Container module with an reader, a recall module, and a writer."""

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
            self.encoder_forward = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)
            self.encoder_backward = nn.RNNCell(ninp, nhid, nonlinearity=nonlinearity)
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

        init_hidden = self.init_hidden(bsz)

        def masked_step(wemb, mask, hidden, mask_hidden, cell):
            new_hidden = cell(wemb, hidden)
            if self.rnn_type == 'LSTM':
                h = new_hidden[0] * mask + mask_hidden[0] * (1 - mask)
                c = new_hidden[1] * mask + mask_hidden[1] * (1 - mask)
                new_hidden = (h, c)
            else:
                h = new_hidden[0] * mask + mask_hidden[0] * (1 - mask)
                new_hidden = h

            return new_hidden, h

        # Forward read
        hidden = self.init_hidden(bsz)
        output_encoder_forward = []
        for i in range(input.size(0)):
            hidden, h = masked_step(emb[i], self.inputmask[i], hidden, init_hidden, self.encoder_forward)
            output_encoder_forward.append(h)
        output_encoder_forward = torch.stack(output_encoder_forward, dim=1)

        # Backward read
        hidden = self.init_hidden(bsz)
        output_encoder_backward = []
        for i in range(input.size(0) - 1, -1, -1):
            hidden, h = masked_step(emb[i], self.inputmask[i], hidden, init_hidden, self.encoder_backward)
            output_encoder_backward.append(h)
        output_encoder_backward = torch.stack(output_encoder_backward[::-1], dim=1)

        output_encoder = output_encoder_forward + output_encoder_backward

        # Recall
        memory = []
        hidden = self.init_hidden(bsz)
        m = Variable(next(self.parameters()).data.new(bsz, self.nhid).zero_())
        for i in range(self.max_nslots):
            hidden, h = masked_step(m, 1, hidden, init_hidden, self.recall)
            m = self.write_attention(h, output_encoder)
            memory.append(m)
        memory = torch.stack(memory, dim=1)

        # Rewrite
        output_decoder = []
        dist = []
        hidden = self.init_hidden(bsz)
        # hidden = encoder_hidden
        for i in range(input.size(0)):
            hidden, h = masked_step(emb[i], self.inputmask[i], hidden, init_hidden, self.decoder)
            m = self.read_attention(h, memory)
            dist.append(self.read_attention.dist)
            output_decoder.append(m + h)
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
