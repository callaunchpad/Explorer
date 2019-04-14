import numpy as np

class Maze:

	def __init__(self):
		self.pos = (0, 0)
		self.x_size = 2
		self.y_size = 2
		self.tv_pos = (1, 1)
		self.state_shape = 64

	def get_action_space(self):
		return 9

	def get_state(self, action):
		x_move = action[0]
		y_move = action[1]
		x_pos = self.pos[0]
		y_pos = self.pos[1]

		x_pos += x_move
		y_pos += y_move

		if x_pos < 0:
			x_pos = 0
		if x_pos > self.x_size - 1:
			x_pos = self.x_size - 1
		if y_pos < 0:
			y_pos = 0
		if y_pos > self.y_size - 1:
			y_pos = self.y_size - 1

		self.pos = (x_pos, y_pos)

		if self.pos == self.tv_pos:
			# print("watching TV")
			return (x_pos, y_pos, np.random.normal(0, 10, self.state_shape))
		else:
			# print("exercising")
			return (x_pos, y_pos, (np.zeros(self.state_shape) + 5))

# maze = Maze()
