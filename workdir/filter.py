#!/usr/bin/python3

from collections import defaultdict
import random

random.seed(12345)

def coin(p):
    return random.random() < p

needed = set()
for line in open('synthetic-contextual-sampled.tsv'):
    id, context, sentence, program = line.strip().split('\t')
    needed.add(context)
    
found = defaultdict(list)
for line in open('synthetic.tsv'):
    id, sentence, program = line.strip().split('\t')
    if program in needed or coin(1e-4):
        found[program].append((id, sentence))
        
for p, val in found.items():
    for id, s in val:
        print(id, s, p, sep='\t')
