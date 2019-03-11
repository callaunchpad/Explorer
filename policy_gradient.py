import tensorflow as tf
import numpy as np
import gym

import architectures as arch

class Rollout:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
    
    def update(self, state, action, reward, next_state):
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
    
    def __iter__(self):
        return iter([self.observations, self.actions, self.rewards])

# TODO:
# Put the logic from Agent into Actor...
class Actor:
    
    def __init__(self, input_size, output_size, critic):
        self.input_size = input_size
        self.output_size = output_size
        self.critic = critic
        
        self.input_states = tf.placeholder(tf.float32, shape=[None, input_size])
        self.input_actions = tf.placeholder(tf.int32, shape=[None,])
        self.input_rewards = tf.placeholder(tf.float32, shape=[None,])
        
        self.output = arch.feed_forward(
            self.input_actions,
            {
                'output_size': self.output_size,
                'hidden_sizes': [30, 30],
                'activations': [tf.activations.relu, tf.activations.relu],
                'kernel_initializers': [tf.initializers.relu, tf.initializers.relu],
            }
        )
        
        self.output_probabilities = tf.nn.softmax(self.output)
        self.input_actions_one_hot = tf.one_hot(self.input_actions, self.output_size)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_actions_one_hot, logits=self.output)
        self.loss = tf.reduce_mean(self.cross_entropy * self.input_rewards)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
    
    def eval(self, state, session):
        feed_dict = {self.input_state: state}
        return session.run(self.output_probabilities, feed_dict=feed_dict)
    
    def update(self, states, actions, rewards, session):
        feed_dict = {
            self.input_states: states,
            self.input_actions: actions,
            self.input_rewards: rewards
        }
        loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

class Critic:
    # TODO: Implement me
    pass

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
        for i in range(250):
            state_obs = env.reset()
            done = False
            rollout = Rollout()
            while (not done):
                old_state_obs = state_obs
                action = np.random.choice(np.arange(2), p=agent.eval(old_state_obs[np.newaxis,:], sess)[0])
                state_obs, reward, done, _ = env.step(action)
                rollout.update(old_state_obs, action, reward)
                if (i % 25 == 0):
                    env.render()
            rollout.generate_processed_rewards()
            loss = agent.train(*rollout, sess)
            if (i % 5 == 0):
                print(f"Step {i}: Loss of {loss*100:.2f}%, reward of {sum(rollout.rewards)}")
    

if __name__ == "__main__":
    main()