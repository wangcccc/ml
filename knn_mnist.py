"""
kNN solution for the recognition of the MNIST database of handwritten digits
Training and test set are downloaded from: http://yann.lecun.com/exdb/mnist/

Result when set train_size to 60,000 and test_size to 3,000:
The error rate is: 0.045667
Time used: 1719.804910s
"""

import numpy as np
import cv2
import time

X_SKIP = 16
Y_SKIP = 8
IMG_W = 28
IMG_H = 28
IMG_SIZE = IMG_W * IMG_H


# use cv2.waitKey() to stop window from closing
def display_nth_digit(fname, n):
    fr = open(fname, 'rb')
    fr.seek(X_SKIP + IMG_SIZE * n)
    bytes = fr.read(IMG_SIZE)
    fr.close()
    img = np.frombuffer(bytes, np.uint8).reshape(IMG_W, IMG_H)
    cv2.imshow(str(n), img)


# load normalized input data
def load_x(fname, size):
    fr = open(fname, 'rb')
    fr.seek(X_SKIP)
    bytes = fr.read(IMG_SIZE * size)
    fr.close()
    array = np.frombuffer(bytes, np.uint8).reshape(size, IMG_SIZE)
    return array / 255


def load_y(fname, size):
    fr = open(fname, 'rb')
    fr.seek(Y_SKIP)
    bytes = fr.read(size)
    fr.close()
    array = np.frombuffer(bytes, np.uint8)
    return array


def knn_classify(x, train_x, train_y, k):
    diffs = train_x - x
    dists = (diffs ** 2).sum(axis=1)
    sorted_dist_indices = dists.argsort()
    label_count = {}
    for i in range(k):
        label = train_y[sorted_dist_indices[i]]
        label_count[label] = label_count.get(label, 0) + 1
    sorted_label_count = sorted(label_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_label_count[0][0]


def test_viewer():
    display_nth_digit('train-images-idx3-ubyte', 100)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train_size = 60000
    test_size = 3000
    train_x = load_x('train-images-idx3-ubyte', train_size)
    train_y = load_y('train-labels-idx1-ubyte', train_size)
    test_x = load_x('t10k-images-idx3-ubyte', test_size)
    test_y = load_y('t10k-labels-idx1-ubyte', test_size)

    start_time = time.time()
    error_count = 0
    for i in range(len(test_x)):
        label = knn_classify(test_x[i], train_x, train_y, 10)
#        print('The prediction is: %d, the real answer is: %d' % (label, test_y[i]))
        if label != test_y[i]:
            error_count += 1
    print('The error rate is: %f' % (float(error_count) / test_size))
    print('Time used: %fs' % (time.time() - start_time))
