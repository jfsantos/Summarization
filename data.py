import os
import numpy


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0, '<s>': 1, '</s>': 2}
        self.idx2word = ['<unk>', '<s>', '</s>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=5):
        self.word2idx = {'<unk>': 0, '<s>': 1, '</s>': 2}
        self.idx2word = ['<unk>', '<s>', '</s>']

        for k, v in self.word2frq.iteritems():
            if v >= thd:
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.add_words(os.path.join(path, 'train.txt'))
        self.dictionary.rebuild_by_freq()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def add_words(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                sub_line = line.strip().split('\t')
                words = []
                for s in sub_line:
                    words += ['<s>'] + s.strip().split() + ['</s>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        return tokens

    def tokenize(self, path, sort_by_len=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        sens = []
        with open(path, 'r') as f:
            for line in f:
                sub_line = line.strip().split('\t')
                words = []
                for s in sub_line:
                    words += ['<s>'] + s.strip().split() + ['</s>']
                ids = numpy.zeros(len(words), dtype='int32')
                for i, word in enumerate(words):
                    ids[i] = self.dictionary[word]
                if 150 > len(ids) > 6:
                    sens.append(ids)

        if sort_by_len:
            sorted_index = sorted(range(len(sens)), key=lambda x: len(sens[x]))
            sens = [sens[i] for i in sorted_index]

        return sens
