import tensorflow as tf


class Agent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_x = tf.placeholder(tf.float32, shape=[None, input_size])
        self.input_actions = tf.placeholder(tf.int32, shape=[None, ])
        self.input_rewards = tf.placeholder(tf.float32, shape=[None, ])
        self.hidden1 = tf.layers.dense(self.input_x, 30, tf.nn.leaky_relu)
        self.hidden2 = tf.layers.dense(self.hidden1, 30, tf.nn.leaky_relu)
        self.output = tf.layers.dense(self.hidden2, self.output_size, tf.nn.leaky_relu)

        self.softmax_output = tf.nn.softmax(self.output)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.input_actions, self.output_size), logits=self.output) * self.input_rewards
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.train_operator = self.optimizer.minimize(self.loss)

    def eval(self, state, session):
        feed_dict = {self.input_x: state}
        return session.run(self.softmax_output, feed_dict=feed_dict)

    def train(self, states, actions, rewards, session):
        feed_dict = {self.input_actions: actions, self.input_rewards: rewards, self.input_x: states}
        loss, _ = session.run([self.loss, self.train_operator], feed_dict=feed_dict)
        return loss
