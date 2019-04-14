import tensorflow as tf
import gym
import os
import sys
import numpy as np
from algorithms.policygrad import Agent
from algorithms.rollouts import Rollouts

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))


def discount(rewards):
    result = []
    sum = 0
    for i in range(0, len(rewards)):
        sum = 0.99 * sum + rewards[len(rewards) - 1 - i]
        result.insert(0, sum)
    result -= np.mean(result)
    result /= np.std(result)
    return result


def main():
    with tf.Session() as sess:
        env = gym.make("CartPole-v0")
        env._max_episode_steps = 1E3
        agent = Agent(4, 2)
        tf.global_variables_initializer().run()
        count = 0
        rollout = Rollouts()
        for i in range(2000):
            state_obs = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                old_state_obs = state_obs
                action = np.random.choice(np.arange(2), p=agent.eval(old_state_obs[np.newaxis, :], sess)[0])
                state_obs, reward, done, _ = env.step(action)
                states.append(old_state_obs)
                actions.append(action)
                rewards.append(reward)
                if done:
                    print("Number of times trained: ", count, sum(rewards))
                    rewards = discount(rewards)
                    for index in range(len(states)):
                        rollout.update(states[index], actions[index], rewards[index])
                if i % 25 == 0:
                    env.render()
            if i % 10 == 0:
                loss = agent.train(*rollout.stuff(), sess)
                count += 1
                print("Training: ", loss)


if __name__ == "__main__":
    main()
