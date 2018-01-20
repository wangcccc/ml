"""
Use multinomial Naive Bayes model to classify spam comments.
Data set is downloaded from http://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
"""

import csv
import re
from math import log
from collections import Counter
from collections import defaultdict


def load_data_set(fname):
    data_set = []
    classes = [0, 1]
    with open(fname) as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            comment = list(filter(None, re.split('[^a-zA-Z0-9]', row['CONTENT'])))
            label = int(row['CLASS'])
            data_set.append((comment, label))
    return data_set, classes


def train_mnb(data_set, classes):
    vocabulary = set()
    concat_comment = defaultdict(list)
    class_counter = defaultdict(int)
    for data in data_set:
        vocabulary.update(data[0])
        concat_comment[data[1]] += data[0]
        class_counter[data[1]] += 1
    num_vocabulary = len(vocabulary)
    num_comment = len(data_set)
    prior = {}
    condprob = defaultdict(dict)
    for c in classes:
        prior[c] = class_counter[c] / num_comment
        word_counter = Counter(concat_comment[c])
        num_word = len(concat_comment[c])
        for word in vocabulary:
            condprob[c][word] = (word_counter[word] + 1) / (num_word + num_vocabulary)
    return vocabulary, prior, condprob


def classify(classes, vocabulary, prior, condprob, sample):
    sample = list(set(vocabulary) & set(sample))
    score = {}
    for c in classes:
        score[c] = log(prior[c])
        for word in sample:
            score[c] += log(condprob[c][word])
    return max(score.items(), key=lambda kv: kv[1])[0]


data_set, classes = load_data_set('data/YouTube-Spam-Collection-v1/Youtube01-Psy.csv')
vocabulary, prior, condprob = train_mnb(data_set, classes)
data_set, classes = load_data_set('data/YouTube-Spam-Collection-v1/Youtube04-Eminem.csv')
num_comment = len(data_set)
error_count = 0
for d in data_set:
    c = classify(classes, vocabulary, prior, condprob, d[0])
    if c != d[1]:
        error_count += 1
print('The error rate is: %f' % (float(error_count) / num_comment))

