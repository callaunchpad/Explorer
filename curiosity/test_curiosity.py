import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

from curiosity_net import CuriosityNet
from qhash import QHashAgent
from tv_maze import Maze 

NUM_BATCHES = 30
HORIZON = 100
BATCH_SIZE = 600
discount = .2
move_reward = 1
stay_reward = -1
curiosity_weight = 10

def train_batch(agent, curiosity):
	count_in_tv = 0
	for i in range(BATCH_SIZE):
		maze = Maze()
		states = []
		actions = []
		ep_reward = []
		maze.get_state(agent.eps_greedy(maze.pos))

		for j in range(HORIZON):
			states.append(maze.pos)
			action = agent.eps_greedy(maze.pos)
			if (maze.pos == maze.tv_pos and action == (0,0)):
				count_in_tv += 1
			x, y, obs = maze.get_state(action)

			actions.append(action)

			r_curiosity = curiosity.step(np.array(obs).reshape(1, len(obs)))
			
			if action[0] == 0 and action[1] == 0:
				r_action = stay_reward
			else:
				r_action = move_reward

			step_reward = r_action + curiosity_weight * r_curiosity

			for n in range(len(ep_reward)):
				ep_reward[n] += (discount**(j-n))*step_reward
			ep_reward.append(step_reward)


		for j in range(HORIZON):
			agent.update_val(states[j], actions[j], ep_reward[j])

	print("Num staying in TV:", count_in_tv, 
		  "Num moving:", BATCH_SIZE*HORIZON - count_in_tv)
	return count_in_tv

def show_V(Q, batch_num, num_stay_in_tv):
	im = np.zeros((7, 7))
	im[3,:] = -10
	im[:,3] = -10
	for state in Q.keys():
		for a in Q[state].keys():
			x = 3*int(state[0])+ 1 + int(a[0]) + int(state[0]>0)
			y = 3*int(state[1])+ 1 + int(a[1]) + int(state[1]>0)
			im[x, y] = Q[state][a] # max(Q[state].values())
	
	plt.imshow(im)
	plt.colorbar()
	plt.title("Q-Value at Batch " + str(batch_num) +", # Stay in TV = " + str(num_stay_in_tv))
	plt.savefig('Q_batch'+str(batch_num)+'.png')
	plt.clf()
	# plt.show()

def main():
	# agent and curiosity

	actions = []
	for i in [-1.0, 0.0, 1.0]:
		for j in [-1.0, 0.0, 1.0]:
			actions.append((i, j))
	agent = QHashAgent(actions)
	curiosity = CuriosityNet()
	print("Starting training!\n\n")
	num_stay_in_tv = 0
	for i in range(NUM_BATCHES):
		print("Batch ", i)
		show_V(agent.Q, i, num_stay_in_tv)
		num_stay_in_tv = train_batch(agent, curiosity)
		agent.eps *= .65
		agent.lr *=.95
if __name__ == "__main__":
	main()
