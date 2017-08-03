import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# preprocessing
train_X, train_Y = mnist.train.images, mnist.train.labels
test_X, test_Y = mnist.test.images, mnist.test.labels

img_width, img_height = 28, 28

train_X = train_X.reshape(-1, img_width, img_height)
test_X  = test_X.reshape(-1, img_width, img_height)

# finished preprocessing -> now network
n_nodes_hidden_1 = 250
n_nodes_hidden_2 = 500
n_nodes_hidden_3 = 750

img_size = 784
n_classes = 10
batch_size = 100

# define the class
x = tf.placeholder(tf.float32, [None, img_size])
y = tf.placeholder(tf.float32, [None, n_classes])

hidden_1 = {
	"weights": tf.Variable(tf.random_normal([img_size, n_nodes_hidden_1])),
	"biases": tf.Variable(tf.random_normal([n_nodes_hidden_1]))
}

hidden_2 = {
	"weights": tf.Variable(tf.random_normal([n_nodes_hidden_1, n_nodes_hidden_2])),
	"biases": tf.Variable(tf.random_normal([n_nodes_hidden_2]))
}

hidden_3 = {
	"weights": tf.Variable(tf.random_normal([n_nodes_hidden_2, n_nodes_hidden_3])),
	"biases": tf.Variable(tf.random_normal([n_nodes_hidden_3]))
}

output = {
	"weights": tf.Variable(tf.random_normal([n_nodes_hidden_3, n_classes])),
	"biases": tf.Variable(tf.random_normal([n_classes]))
}

l1 = tf.add(tf.matmul(x, hidden_1["weights"]), hidden_1["biases"])
u1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(u1, hidden_2["weights"]), hidden_2["biases"])
u2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(u2, hidden_3["weights"]), hidden_3["biases"])
u3 = tf.nn.relu(l3)

output = tf.add(tf.matmul(u3, output["weights"]), output["biases"])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=output, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

num_epochs = 5
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for epoch in range(num_epochs):
		epoch_loss = 0
		for _ in range(mnist.train.num_examples/batch_size):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={
				x: epoch_x,
				y: epoch_y
			})
			epoch_loss += c
		print("Epoch completed {}/{} - loss: {}".format(
			epoch, num_epochs, epoch_loss))
	
	correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	print("Test accuracy: {}".format(accuracy.eval({
		x : mnist.test.images,
		y : mnist.test.labels
	})))