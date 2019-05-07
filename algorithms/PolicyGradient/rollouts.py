import numpy as np
import copy


class Rollouts:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def update(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def stuff(self):
        num_samples = int(len(self.observations) / 2)
        observations = copy.deepcopy(self.observations)
        actions = copy.deepcopy(self.actions)
        rewards = copy.deepcopy(self.rewards)
        np.random.seed(100)
        np.random.shuffle(observations)
        np.random.seed(100)
        np.random.shuffle(actions)
        np.random.seed(100)
        np.random.shuffle(rewards)
        observations = observations[:num_samples]
        actions = actions[:num_samples]
        rewards = rewards[:num_samples]
        return observations, actions, rewards
