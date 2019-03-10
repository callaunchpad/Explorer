import tensorflow as tf
import numpy as np
import gym
from sys import argv
from curiosity.curiosity_net import CuriosityNet

class Rollout:
	def __init__(self):
		self.observations = []
		self.actions = []
		self.rewards = []

	def update(self, observation, action, reward):
		self.observations.append(observation)
		self.actions.append(action)
		self.rewards.append(reward)
	
	def generate_processed_rewards(self):
		# Apply discounted accumulation
		self.processed_rewards = self.rewards[:]
		future_reward = 0
		scale_factor = 0.99
		for i in reversed(range(len(self.processed_rewards))):
			future_reward = self.processed_rewards[i] + future_reward * scale_factor
			self.processed_rewards[i] = future_reward
		# Normalize
		self.processed_rewards -= np.mean(self.processed_rewards)
		self.processed_rewards /= np.std(self.processed_rewards)
	
	def get_rollout(self):
		return (self.observations, self.actions, self.processed_rewards)

class Agent:
	def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.batch_size = 1
		# self.curiosity = CuriosityNet(self.input_size, self.output_size, self.batch_size)
		with tf.variable_scope("policy_grad"):
			self.input_x = tf.placeholder(tf.float32, shape=[None, input_size])
			self.input_actions = tf.placeholder(tf.int32, shape=[None,])
			self.input_rewards = tf.placeholder(tf.float32, shape=[None,])
			self.hidden1 = tf.layers.dense(self.input_x, 30, tf.nn.leaky_relu)
			self.hidden2 = tf.layers.dense(self.hidden1, 30, tf.nn.leaky_relu)
			self.output = tf.layers.dense(self.hidden2, self.output_size, tf.nn.leaky_relu)
			self.softmax_output = tf.nn.softmax(self.output)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.input_actions, self.output_size), logits=self.output) * self.input_rewards
			self.loss = tf.reduce_mean(self.cross_entropy)
			self.train_operator = self.optimizer.minimize(self.loss)
		
	
	def eval(self, state, session):
		feed_dict = {self.input_x: state}
		return session.run(self.softmax_output, feed_dict=feed_dict)

	def train(self, states, actions, rewards, session):
		feed_dict = {self.input_actions: actions, self.input_rewards: rewards, self.input_x: states}
		loss, _ = session.run([self.loss, self.train_operator], feed_dict=feed_dict)
		return loss

def main():
	use_curiosity = False
	use_display = False
	stochastic_action = False
	if len(argv) > 1:
		if "use_curiosity" in argv:
			use_curiosity = True
		if "use_display" in argv:
			use_display = True
		if "stochastic_action" in argv:
			stochastic_action = True
	with tf.Session() as sess:
		env = gym.make("CartPole-v0")
		env._max_episode_steps = 1E3
		input_size = 4
		output_size = 2
		agent = Agent(input_size, output_size)
		if use_curiosity:
			curiosity_discount = 100.0
			curiosity = CuriosityNet(input_size, output_size, 1)
		tf.global_variables_initializer().run()
		for i in range(25000):
			state_obs = env.reset()
			done = False
			rollout = Rollout()
			global_reward = 0
			while (not done):
				old_state_obs = state_obs
				agent_choice = agent.eval(old_state_obs[np.newaxis, :], sess)[0]
				if stochastic_action:
					action = np.random.choice(np.arange(2), p=agent_choice)
				else:
					action = np.argmax(agent_choice)
				state_obs, reward, done, _ = env.step(action)
				global_reward += reward
				if use_curiosity:
					reward += curiosity_discount * curiosity.step(np.array([state_obs]))
				rollout.update(old_state_obs, action, reward)
				if (i % 25 == 0 and use_display):
					env.render()
			rollout.generate_processed_rewards()
			loss = agent.train(*rollout.get_rollout(), sess)
			if (i % 5 == 0):
				print(f"Step {i}: Loss of {loss*100:.2f}%, reward of {global_reward}")
	

if __name__ == "__main__":
	main()
