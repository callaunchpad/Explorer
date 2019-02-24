import tensorflow as tf
import numpy as np
import os

class Net:

	def __init__(self):
		self.input_size = 10
		self.output_size = 10
		self.batch_size = 1
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_size])

		# Model 1: Fixed network
		self.num_units1 = 128
		with tf.variable_scope("fixed"):
			self.m1_l1 = tf.layers.dense(self.x, self.num_units1, activation=tf.nn.relu, trainable=False)
			self.truth = tf.layers.dense(self.m1_l1, self.output_size, activation=tf.nn.relu, trainable=False)

		self.weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.m1_l1.name)[0] + '/kernel:0')

		# Model 2: Prediction network
		self.num_units2 = 128
		with tf.variable_scope("trainable"):
			self.m2_l1 = tf.layers.dense(self.x, self.num_units2, activation=tf.nn.relu)
			self.pred = tf.layers.dense(self.m2_l1, self.output_size, activation=tf.nn.relu)

		self.losses = tf.losses.mean_squared_error(labels=self.truth, predictions=self.pred)
		self.total_loss = tf.reduce_mean(self.losses)
		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.005),0.995).minimize(self.total_loss)

	def train(self, x):
		# print(tf.trainable_variables())
		# last_weights = None
		while True:
			loss, weights, _ = self.sess.run([self.total_loss, self.weights, self.optimizer], feed_dict={self.x: x})
			print("loss: %f", loss)
			# print("weights: ", weights == last_weights)
			# last_weights = weights

net = Net()
test = np.array([np.random.rand(net.input_size)])
net.train(test)