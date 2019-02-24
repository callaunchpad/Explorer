import tensorflow as tf

class Net:

	def __init__(self):
		self.sess = tf.Session()

	def _build_model(self):
		