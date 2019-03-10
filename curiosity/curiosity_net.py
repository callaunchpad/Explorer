import tensorflow as tf
import numpy as np
import os

class CuriosityNet:
	"""Basic Curiosity Module with Random Network Distillation

	TODO:
		Modularize it to take any network structure and build it
		for both the fixed and trainable networks.

	@Authors:
		An Wang
		Arsh Zahed
	"""

	def __init__(self, input_size, output_size, batch_size):
		"""
		Builds and initializes parameters	
		"""
		self.input_size = input_size
		self.output_size = output_size
		self.batch_size = batch_size
		self.sess = tf.Session()
		self._build_model()
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		"""
		Builds the graph for the curiosity module.
		"""
		self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_size])

		# Model 1: Fixed network
		self.num_units1 = 32
		with tf.variable_scope("fixed"):
			self.m1_l1 = tf.layers.dense(self.x, 
										self.num_units1, 
										activation=tf.nn.relu,
										kernel_initializer=tf.initializers.random_normal,
										bias_initializer=tf.initializers.random_normal,
										trainable=False)
			self.truth = tf.layers.dense(self.m1_l1, 
										self.output_size, 
										activation=tf.nn.relu, 
										kernel_initializer=tf.initializers.random_normal,
										bias_initializer=tf.initializers.random_normal,
										trainable=False)

		self.weights = tf.get_default_graph().get_tensor_by_name(os.path.split(
										self.m1_l1.name)[0] + '/kernel:0')

		# Model 2: Prediction network
		self.num_units2 = 32
		with tf.variable_scope("trainable"):
			self.m2_l1 = tf.layers.dense(self.x, 
										self.num_units2, 
										activation=tf.nn.relu,
										kernel_initializer=tf.initializers.random_normal,
										bias_initializer=tf.initializers.random_normal
										)
			self.pred = tf.layers.dense(self.m2_l1, 
										self.output_size, 
										activation=tf.nn.relu,
										kernel_initializer=tf.initializers.random_normal,
										bias_initializer=tf.initializers.random_normal
										)

		self.losses = tf.losses.mean_squared_error(labels=self.truth, 
												predictions=self.pred)
		self.total_loss = tf.reduce_mean(self.losses)
		self.optimizer = tf.train.RMSPropOptimizer(tf.constant(0.005),
												0.995).minimize(self.total_loss)

	def train(self, x):
		"""
		Train Loop, NOT FINAL!!!!!!
		"""
		while True:
			loss = self.step(x)
			print("loss: %f", loss)

	def step(self, x):
		"""
		Does a single optimization step

		Args:
			x: Array input to the curiosity module
		"""
		loss, _ = self.sess.run([self.total_loss, 
								self.optimizer], 
								feed_dict={self.x: x})
		return loss


if __name__ == "__main__":
	net = CuriosityNet(10, 10, 1)
	test = np.array([np.random.rand(net.input_size)])
	net.train(test)

