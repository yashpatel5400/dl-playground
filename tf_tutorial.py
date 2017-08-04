import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
num_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for fn in (pos, neg):
        with open(fn, "r") as f:
            contents = f.readlines()
            for line in contents:
                lexicon += list(word_tokenize(line.lower()))
    lexicon = [lemmatizer.lemmatize(word) for word in lexicon]
    word_counts = Counter(lexicon)
    LOW_THRESH  = 25
    HIGH_THRESH = 1000
    filtered_words = [word for word in word_counts if
          LOW_THRESH < word_counts[word] < HIGH_THRESH]
    return filtered_words

def sample_handling(sample, lexicon, classification):
    feature_set = []
    with open(sample, "r") as f:
        contents = f.readlines()
        for l in contents:
            words = word_tokenize(l.lower())
            feature = np.zeros(len(lexicon))
            for word in words:
                if word in lexicon:
                    ind = lexicon.index(word)
                    feature[ind] += 1
            feature = list(feature)
            feature_set.append([feature, classification])
    return feature_set

def create_feature_sets_and_labels(pos,neg,test_prop=0.1):
    lexicon = create_lexicon(pos,neg)
    samples = []
    samples += sample_handling(pos, lexicon, [1,0])
    samples += sample_handling(neg, lexicon, [0,1])
    random.shuffle(samples)
    samples = np.array(samples)
    test_len = int(len(samples) * test_prop)
    trainX, trainY = list(samples[:,0][:-test_len]), list(samples[:,1][:-test_len])
    testX, testY = list(samples[:,0][-test_len:]), list(samples[:,1][-test_len:])
    return trainX, trainY, testX, testY

def dcreate_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y


def create_model(lexicon_size, output_size):
    x = tf.placeholder(tf.float32, shape=[None, lexicon_size])
    y = tf.placeholder(tf.float32, shape=[None, output_size])

    h1_size = 64
    h2_size = 128
    h3_size = 256

    h1 = {
        "weights": tf.Variable(tf.random_normal(([lexicon_size, h1_size]))),
        "biases": tf.Variable(tf.random_normal(([h1_size])))
    }

    h2 = {
        "weights": tf.Variable(tf.random_normal(([h1_size, h2_size]))),
        "biases": tf.Variable(tf.random_normal(([h2_size])))
    }

    h3 = {
        "weights": tf.Variable(tf.random_normal(([h2_size, h3_size]))),
        "biases": tf.Variable(tf.random_normal(([h3_size])))
    }

    output = {
        "weights": tf.Variable(tf.random_normal(([h3_size, output_size]))),
        "biases": tf.Variable(tf.random_normal(([output_size])))
    }

    l1 = tf.add(tf.matmul(x, h1["weights"]), h1["biases"])
    u1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(u1, h2["weights"]), h2["biases"])
    u2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(u2, h3["weights"]), h3["biases"])
    u3 = tf.nn.relu(l3)
    prediction = tf.add(tf.matmul(u3, output["weights"]), output["biases"])
    return x, y, prediction

def run():
    trainX, trainY, testX, testY = pickle.load(open("sentiment_set.pickle", "rb"))
    # trainX, trainY, testX, testY = create_feature_sets_and_labels("pos.txt", "neg.txt")
    lexicon_size = len(trainX[0])
    output_size = 2

    x, y, prediction = create_model(lexicon_size, output_size)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    avg_cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer().minimize(avg_cost)

    num_epochs = 30
    batch_size = 100

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(num_epochs):
            epoch_cost = 0
            for batch_num in range(int(len(trainX)/batch_size)):
                start_batch = batch_num * batch_size
                sampleX = trainX[start_batch:start_batch+batch_size]
                sampleY = trainY[start_batch:start_batch + batch_size]
                _, c = sess.run([optimizer, avg_cost], feed_dict={
                    y: np.array(sampleY),
                    x: np.array(sampleX)
                })
                epoch_cost += c
            print("Epoch {} completed w/ cost: {}".format(epoch, epoch_cost))

        accuracy = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc_cost = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        acc_val  = acc_cost.eval({
            x: testX,
            y: testY
        })
        print("Completed w/ accuracy: {}".format(acc_val))

if __name__ == "__main__":
    trainX, trainY, testX, testY = \
        create_feature_sets_and_labels("pos.txt", "neg.txt")
    with open("sentiment_set.pickle", "wb") as f:
        pickle.dump([trainX, trainY, testX, testY], f)
    run()