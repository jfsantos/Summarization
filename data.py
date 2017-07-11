import os
import torch
import numpy

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def get_id(self, w):
        if self.word2idx.has_key(w):
            return self.word2idx[w]
        else:
            return self.word2idx['<unk>']


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, sort_by_len=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<s>'] + line.split() + ['</s>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sens = []
        with open(path, 'r') as f:
            for line in f:
                words = ['<s>'] + line.split() + ['</s>']
                ids = numpy.zeros(len(words), dtype='int32')
                for i, word in enumerate(words):
                    ids[i] = self.dictionary.word2idx[word]
                if len(ids) > 3:
                    sens.append(ids)

        if sort_by_len:
            sorted_index = sorted(range(len(sens)), key=lambda x: len(sens[x]))
            sens = [sens[i] for i in sorted_index]

        return sens
