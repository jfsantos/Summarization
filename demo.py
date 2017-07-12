import argparse

import numpy
import torch
from torch.autograd import Variable

import data

numpy.set_printoptions(precision=3, suppress=True, linewidth=500)

def criterion(input, targets, targets_mask):
    targets_mask = targets_mask.float()
    input = input.view(-1, ntokens)
    loss = torch.gather(input, 1, targets[:, None]).view(-1)
    loss = (-loss * targets_mask).sum() / targets_mask.sum()
    return loss

parser = argparse.ArgumentParser(description='News summarization model')

# Model parameters.
parser.add_argument('--data', type=str, default='../Data/ParaNews/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

while True:
    s = raw_input('Input a sentence:')
    words = ['<s>'] + s.strip().split(' ') + ['</s>']
    x = [corpus.dictionary[w] for w in words]
    l = len(x)

    data = numpy.zeros((l+1, 1), dtype='int64')
    mask = numpy.zeros((l+1, 1), dtype='int64')
    data[:l, 0] = x
    mask[:l, 0] = 1

    data = torch.LongTensor(data)
    mask = torch.LongTensor(mask)
    if args.cuda:
        data = data.cuda()
        mask = mask.cuda()

    input = Variable(data[:-1], volatile=True)
    input_mask = Variable(mask[:-1], volatile=True)
    target = Variable(data[1:].view(-1))
    target_mask = Variable(mask[1:].view(-1))

    model.seq2seq.set_mask(input_mask)
    output = model(input)
    output_flat = output.view(-1, ntokens)
    loss = criterion(output_flat, target, target_mask).data

    print 'Perplexity', numpy.exp(loss.numpy()[0])
    attention_dist = model.seq2seq.read_attention_dist.data.squeeze().numpy()
    print attention_dist.T