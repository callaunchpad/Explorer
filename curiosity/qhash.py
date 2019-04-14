import operator
import random


class QHashAgent:
	def __init__(self, actions, start_lr=0.1, eps=.2):
		"""
		Initializes Q Table and saves possible actions
		"""
		self.Q = dict()
		self.actions = actions
		self.eps = eps
		self.start_lr = start_lr
		self.lr = start_lr
		self.num_updates = 1


	def act(self, state):
		"""
		Returns that action that maximizes expected reward
		"""
		if state not in self.Q:
			self.Q[state] = dict()
			for i in range(len(self.actions)):
				self.Q[state][self.actions[i]] = 0

		max_a = max(self.Q[state].keys(), key=lambda x: self.Q[state][x])

		return max_a


	def update_val(self, state, action, reward):
		"""
		Updates a Q value
		"""
		if state not in self.Q:
			self.Q[state] = dict()
			for a in self.actions:
				self.Q[state][a] = 0

		self.Q[state][action] *= (1-self.lr)
		self.Q[state][action] += self.lr * reward
		# self.num_updates +=1
		# self.lr = self.start_lr / self.num_updates


	def eps_greedy(self, state):
		"""
		Epsilon Greedy acion
		"""
		r = random.uniform(0.0, 1.0)
		if state not in self.Q:
			self.Q[state] = dict()
			for i in range(len(self.actions)):
				self.Q[state][self.actions[i]] = 0

		if r<self.eps:
			return random.choice(list(self.Q[state].keys()))
		else:
			return self.act(state)

