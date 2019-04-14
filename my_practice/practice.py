from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow
import tensorflow as tf
# import MNIST (MNIST is a dataset containing correctly labeled images of handwritten numbers)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.int64, shape=(None))

hidden1 = tf.layers.dense(x, 300, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu)
output = tf.layers.dense(hidden2, 10)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
loss = tf.reduce_mean(cross_entropy)

optimizer=tf.train.AdamOptimizer(0.001)
training_Op=optimizer.minimize(loss)


with tf.name_scope("eval"):
    # correct variable is the number of correct labels
    correct = tf.nn.in_top_k(output, y, 1)
    # count up the number of correct labels
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

'''session keeps values of graph'''
with tf.Session() as session:
    tf.global_variables_initializer().run()

    for step in range(3000):
        x_batch, y_batch = mnist.train.next_batch(batch_size=120)

        session.run(training_Op, feed_dict={x: x_batch, y: y_batch})

        accuracy_val = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(step, "Test accuracy:", accuracy_val)



