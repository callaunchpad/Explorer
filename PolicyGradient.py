import tensorflow as tf
import numpy as np
import gym
import copy

def discount(rewards):
    result = []
    sum = 0
    for i in range(0, len(rewards)):
        sum = 0.99 * sum + rewards[len(rewards)-1-i]
        result.insert(0, sum)
    result -= np.mean(result)
    result /= np.std(result)
    return result

class Rollout:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
    
    def update(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def stuff(self):
        num_samples = int(len(self.observations)/2)
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
        return (observations, actions, rewards)

class Agent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
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
    with tf.Session() as sess:
        env = gym.make("CartPole-v0")
        env._max_episode_steps = 1E3
        agent = Agent(4, 2)
        tf.global_variables_initializer().run()
        count = 0
        rollout = Rollout()
        for i in range(2000):
            state_obs = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while (not done):
                old_state_obs = state_obs
                action = np.random.choice(np.arange(2), p=agent.eval(old_state_obs[np.newaxis,:], sess)[0])
                state_obs, reward, done, _ = env.step(action)
                states.append(old_state_obs)
                actions.append(action)
                rewards.append(reward)
                if done:
                    print("Number of times trained: ", count, sum(rewards))
                    rewards = discount(rewards)
                    for index in range(len(states)):
                        rollout.update(states[index], actions[index], rewards[index])
                if (i % 25 == 0):
                    env.render()
            if (i % 10 == 0):
                loss = agent.train(*rollout.stuff(), sess)
                count += 1
                print ("Training: ", loss)
    

if __name__ == "__main__":
    main()