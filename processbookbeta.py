from __future__ import absolute_import
import pandas as pd
from io import open
from ast import literal_eval
import csv
import re
import enchant
from collections import Counter
dictionary = enchant.Dict("en_US")
stopWords = ['and','the', 'does', 'don\'t','she','it',
             'they','her','his','him','its','you','with','was','were','have','has','had','where','when','then','there','that']
data_header = ['cap', 'location']

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
data_header = ['cap', 'location']
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def processFile():
    data = ""
    with open('book1.txt', 'r') as file:
        data = file.read()
    sentences = split_into_sentences(data)
    tr_data = pd.read_csv("COCO.csv")
    texts = tr_data["cap"]
    locations = tr_data["location"]
    dataset_data = []
    unique_locs = []
    loc_texts = dict()
    pre_sent = dict()
    post_sent = dict()
    for i in range(0, len(locations)):
        locs = literal_eval(locations[i])
        for j in range(0, len(locs)):
            if (not unique_locs.__contains__(locs[j])):
                unique_locs.append(locs[j])

    for i in range(0, len(unique_locs)):
        loc_texts[unique_locs[i]]=[]
    for i in range(0, len(locations)):
        locs = literal_eval(locations[i])
        for j in range(0, len(locs)):
            loc_texts[locs[j]].append(texts[i])

    for i in range(0, len(unique_locs)):
        loc_sentences = []
        for j in range(1, len(sentences)):
            if (sentences[j].__contains__(unique_locs[i]) & (not sentences[j-1].__contains__(unique_locs[i]))):
                sentence = sentences[j-1].replace('\"', "")
                sentArr = sentence.split(" ")
                if (sentArr.__len__()>10):
                    loc_sentences.append(sentence)
        pre_sent[unique_locs[i]]=loc_sentences

    for i in range(0, len(unique_locs)):
        loc_sentences = []
        for j in range(0, len(sentences)-1):
            if (sentences[j].__contains__(unique_locs[i]) & (not sentences[j+1].__contains__(unique_locs[i]))):
                sentence = sentences[j+1].replace('\"', "")
                sentArr = sentence.split(" ")
                if (sentArr.__len__()>10):
                    loc_sentences.append(sentence)
        post_sent[unique_locs[i]]=loc_sentences

    for i in range(0, len(unique_locs)):
        loc_sentences_cur = pre_sent[unique_locs[i]]
        existingSentences = loc_texts[unique_locs[i]]
        text = " ".join(existingSentences)
        words = re.findall('\w+', text)
        counter = Counter(words)
        most_occur = counter.most_common(n=50)
        most_occur_filtered = []
        for y in range(0, len(most_occur)):
            if (len(most_occur[y][0]) > 2 and (not stopWords.__contains__(most_occur[y][0].lower()))):
                most_occur_filtered.append(most_occur[y][0])
        for j in range(0, len(loc_sentences_cur)):
                sentence = loc_sentences_cur[j].replace('\"', "")
                allowed = False
                hits = 0
                for x in range(0, len(most_occur_filtered)):
                    if (re.search(most_occur_filtered[x], sentence, re.IGNORECASE)):
                        hits = hits + 1
                    if (hits>4):
                        allowed = True
                        break
                if (allowed):
                    dataset_data.append([sentence, [unique_locs[i]]])

    for i in range(0, len(unique_locs)):
        loc_sentences_cur = post_sent[unique_locs[i]]
        existingSentences = loc_texts[unique_locs[i]]
        text = " ".join(existingSentences)
        words = re.findall('\w+', text)
        counter = Counter(words)
        most_occur = counter.most_common(n=50)
        most_occur_filtered = []
        for y in range(0, len(most_occur)):
            if (len(most_occur[y][0]) > 2 and (not stopWords.__contains__(most_occur[y][0].lower()))):
                most_occur_filtered.append(most_occur[y][0])
        for j in range(0, len(loc_sentences_cur)):
                sentence = loc_sentences_cur[j].replace('\"', "")
                allowed = False
                hits = 0
                for x in range(0, len(most_occur_filtered)):
                    if (re.search(most_occur_filtered[x], sentence, re.IGNORECASE)):
                        hits = hits + 1
                    if (hits>4):
                        allowed = True
                        break
                if (allowed):
                    dataset_data.append([sentence, [unique_locs[i]]])

    with open("book4probBeta.csv", 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerow(data_header)
        writer.writerows(dataset_data)
processFile()





