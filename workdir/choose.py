#!/usr/bin/python3

import sys
import random

random.seed(12345)

def coin(p):
    return random.random() < p

# N: total size of dataset to choose from
# T: target size
N = 0
T = int(sys.argv[1])

buffer = [None] * T
for line in sys.stdin:
    line = line.strip()
    if N < T:
        buffer[N] = line
    elif coin(T/N):
        pos = random.randrange(0, T)
        buffer[pos] = line
    N += 1
    
for el in buffer:
    print(el)
        
