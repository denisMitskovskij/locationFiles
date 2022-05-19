from __future__ import absolute_import
import pandas as pd
from io import open
from ast import literal_eval
import csv
target = []
data_header = ['cap', 'location']

def processBookFile():
    data = pd.read_csv("book1.csv")
    texts = data["cap"]
    locations = data["location"]
    test_data = pd.read_csv("book4probBeta.csv")
    test_texts = test_data["cap"]
    test_locations = test_data["location"]
    train_data = []
    for i in range(0, len(texts)):
        if (texts[i] not in test_texts):
            train_data.append([texts[i], [locations[i]]])

    with open('book1-upd.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerow(data_header)
        writer.writerows(train_data)

def processGeneratedFile():
    data = pd.read_csv("train.csv")
    texts = data["cap"]
    locations = data["location"]
    test_data = pd.read_csv("test.csv")
    test_texts = test_data["cap"].values
    test_locations = test_data["location"]
    eval_data = []
    train_data = []
    unique_locs = []
    loc_counts = dict()
    loc_put = dict()
    for i in range(0, len(locations)):
        locs = literal_eval(locations[i])
        for j in range(0, len(locs)):
            if (not unique_locs.__contains__(locs[j])):
                unique_locs.append(locs[j])
                loc_put[locs[j]] = 0
                loc_counts[locs[j]] = 0
            loc_counts[locs[j]] = loc_counts[locs[j]] + 1
    for i in range(0, len(locations)):
        locs = literal_eval(locations[i])
        for j in range(0, len(locs)):
            if (not test_texts.__contains__(texts[i])):
                loc_put[locs[j]] = loc_put[locs[j]] + 1
                if (loc_put[locs[j]] < 1/2*loc_counts[locs[j]]):
                    train_data.append([texts[i], locations[i]])
                else:
                    train_data.append([texts[i], locations[i]])
    '''
    with open('test.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerow(data_header)
        writer.writerows(eval_data)
    '''
    with open('train-test.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerow(data_header)
        writer.writerows(train_data)

processBookFile()
processGeneratedFile()
