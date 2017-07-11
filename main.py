import argparse
import time
import math
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import softmax
from collections import OrderedDict

import data
from MemorySeq2Seq import RNNModel

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

emb_layer = nn.Embedding(ntokens, args.emsize, sparse=True)
seq2seq = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
softmax_layer = softmax.Softmax(args.nhid, ntokens, args.dropout)

if args.tied:
    if args.nhid != args.ninp:
        raise ValueError('When using the tied flag, nhid must be equal to emsize')
        softmax_layer.decoder.weight = emb_layer.weight

model = nn.Sequential(OrderedDict([('emb', emb_layer), ('seq2seq', seq2seq), ('softmax', softmax_layer)]))
if args.cuda:
    model.cuda()


def criterion(input, targets, targets_mask):
    targets_mask = targets_mask.float()
    input = input.view(-1, ntokens)
    loss = torch.gather(input, 1, targets[:, None]).view(-1)
    loss = (-loss * targets_mask).sum() / targets_mask.sum()
    return loss


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    bsz = min((args.batch_size, len(source) - i))
    subset = source[i:i + bsz]

    maxlen = max([len(t) for t in subset]) + 1

    data = numpy.zeros((maxlen, args.batch_size), dtype='int64')
    mask = numpy.zeros((maxlen, args.batch_size), dtype='int64')
    for j in range(bsz):
        data[:subset[j].shape[0], j] = subset[j]
        mask[:subset[j].shape[0], j] = 1
    data = torch.LongTensor(data)
    mask = torch.LongTensor(mask)
    if args.cuda:
        data = data.cuda()
        mask = mask.cuda()

    input = Variable(data[:-1], volatile=evaluation)
    input_mask = Variable(mask[:-1], volatile=evaluation)
    target = Variable(data[1:].view(-1))
    target_mask = Variable(mask[1:].view(-1))
    return input, input_mask, target, target_mask


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for i in range(0, len(data_source) - 1, args.batch_size):
        data, data_mask, targets, targets_mask = get_batch(data_source, i, evaluation=True)
        model.seq2seq.set_mask(data_mask)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets, targets_mask).data
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, len(corpus.train) - 1, args.batch_size)):
        data, data_mask, targets, targets_mask = get_batch(corpus.train, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        model.seq2seq.set_mask(data_mask)
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets, targets_mask)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.seq2seq.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(corpus.train) // args.batch_size, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(corpus.valid)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(corpus.test)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
