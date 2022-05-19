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
import csv
target = []
data_header = ['cap', 'location']

def processFile(filename, outname):
    data = pd.read_csv(filename)
    texts = data["cap"]
    locations = data["location"]
    train_data = []
    for i in range(0, len(texts)):
        locs = literal_eval(locations[i])
        if (len(locs)>1):
            for j in range(0, len(locs)):
                train_data.append([texts[i], str([locs[j]])])
        else:
            train_data.append([texts[i], locations[i]])

    with open(outname, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerow(data_header)
        writer.writerows(train_data)
processFile("COCO.csv", "train-upd.csv")
processFile("COCO-dev.csv", "eval-upd.csv")
processFile("COCO-test1.csv", "test1-upd.csv")
processFile("COCO-test2.csv", "test2-upd.csv")
processFile("COCO-test3.csv", "test3-upd.csv")






