import tensorflow as tf
from tensorflow.python.ops import rnn_cell

num_epochs = 10
rnn_size   = 128
num_classes = 2
batch_size = 128

look_back = 28
num_chunks = 28

def recurrent_neural_network(x):
    layer = {
        "weights": tf.Variable(tf.random_normal([rnn_size, num_classes])),
        "biases ": tf.Variable(tf.random_normal([num_classes]))
    }

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, look_back])
    x = tf.split(0, num_chunks, x)

    output = tf.add(tf.matmul(x, layer["weights"]), layer["biases"])
    return output