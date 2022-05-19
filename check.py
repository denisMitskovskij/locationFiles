
from __future__ import absolute_import
import pandas as pd
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from ast import literal_eval
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, BertConfig, BertModel, BertTokenizer)
target = []
with open("resultBertProcessedTestRes/test_0.gold", encoding="utf-8") as f:
    for line in f:
        line = line.replace('\t', ' ')
        line = line.replace('\n', '')
        arrline = line.split(' ')
        target.append(arrline)
totalTarget = len(target)

output = []
with open("resultBertProcessedTestRes/test_0.output", encoding="utf-8") as f:
    for line in f:
        line = line.replace('\t', ' ')
        line = line.replace('\n', '')
        arrline = line.split(' ')
        output.append(arrline)

partHits = 0
fullHits = 0

for i in range(0, totalTarget):
    arrOut = output[i]
    arrTgt = target[i]
    sizeTarget = len(arrTgt)
    sizeOutput = len(arrOut)
    hit = 0
    for j in range(0, sizeOutput):
        if (arrTgt.__contains__(arrOut[j])):
            hit = hit + 1
    if (hit > 1):
        partHits = partHits + 1
        if (hit == sizeTarget and sizeOutput == sizeTarget):
            fullHits = fullHits + 1
precisionFull = fullHits/totalTarget * 100
precisionPart = partHits/totalTarget * 100
print("Full precision = ", precisionFull, "%")
print("Part precision = ", precisionPart, "%")





