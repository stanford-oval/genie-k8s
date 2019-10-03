#!/usr/bin/env python3

import sys
import os
import numpy as np

def process_file(filename):
    vocab = set()

    N = 0
    with open(filename, 'r') as fp:
        for line in fp:
            _id, sentence, target_code = line.strip().split('\t')
            vocab.update(sentence.split(' '))
            N += 1

    vocab = list(vocab)
    vocab.sort()
    vocab.insert(0, '</s>')
    vocab.insert(0, '<s>')
    vocab = dict(((w,i) for i,w in enumerate(vocab)))

    start_id = vocab['<s>']
    eos_id = vocab['</s>']

    V = len(vocab)
    bigrams = np.zeros((V,V), dtype=np.int32)

    with open(filename, 'r') as fp:
        for line in fp:
            _id, sentence, target_code = line.strip().split('\t')
            words = sentence.split(' ')

            for i in range(len(words)+1):
                if i >= len(words):
                    curr = eos_id
                else:
                    curr = vocab[words[i]]
                if i == 0:
                    prev = start_id
                else:
                    prev = vocab[words[i-1]]
                bigrams[prev][curr] += 1

    total_bigrams = np.sum(bigrams)
    nonzero_bigrams = bigrams[bigrams != 0].flatten().astype(np.float32)
    nonzero_bigrams /= total_bigrams
    num_bigrams = len(nonzero_bigrams)

    bigram_entropy = - np.sum(nonzero_bigrams * np.log(nonzero_bigrams))

    print(os.path.basename(filename), N, V, num_bigrams, bigram_entropy, sep='\t')

for f in sys.argv[1:]:
    process_file(f)
