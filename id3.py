"""
An implementation of ID3 algorithm
Training data are downloaded from: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
"""


from math import log
from collections import Counter
from collections import defaultdict


def calc_entropy(data_set):
    entropy = 0.0
    total_count = len(data_set)
    label_count = Counter(feature[-1] for feature in data_set)
    for key in label_count:
        prob = float(label_count[key]) / total_count
        entropy -= prob * log(prob, 2)
    return entropy


def group_by(data_set, axis):
    groups = defaultdict(list)
    for feature in data_set:
        sub_feature = feature[:]
        del sub_feature[axis]
        groups[feature[axis]].append(sub_feature)
    return groups


def get_best_feature(data_set):
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    total_count = len(data_set)
    feature_count = len(data_set[0]) - 1
    for i in range(feature_count):
        new_entropy = 0.0
        for group in group_by(data_set, i).values():
            prob = float(len(group)) / total_count
            new_entropy += prob * calc_entropy(group)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_tree(data_set, labels):
    # every element in the subset belongs to the same class
    groups = group_by(data_set, -1)
    if len(groups) == 1:
        return data_set[0][-1]
    # no more attributes to be selected, return the most common class
    if len(data_set[0]) == 1:
        label_count = Counter(data_set)
        return label_count.most_common(1)[0][0]
    # partition by attribute
    best_feature = get_best_feature(data_set)
    label = labels[best_feature]
    tree = {label: {}}
    groups = group_by(data_set, best_feature)
    for key in groups:
        sub_label = labels[:]
        del sub_label[best_feature]
        tree[label][key] = create_tree(groups[key], sub_label)
    return tree


def classify(tree, labels, sample):
    label = list(tree.keys())[0]
    label_index = labels.index(label)
    key = sample[label_index]
    value = tree[label][key]
    if isinstance(value, dict):
        class_label = classify(value, labels, sample)
    else:
        class_label = value
    return class_label


def load_data_set():
    labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'evaluation']
    fr = open('data/car.data')
    data_set = []
    for line in fr.readlines():
        data_set.append(line.strip().split(','))
    fr.close()
    return data_set, labels


if __name__ == '__main__':
    data_set, labels = load_data_set()
    tree = create_tree(data_set, labels)
    sample = data_set[100]
    prediction = classify(tree, labels, sample)
    answer = sample[-1]
    print('Prediction is: %s, real answer is: %s' % (prediction, answer))
