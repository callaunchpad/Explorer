import gym
import tensorflow as tf
import numpy as np
from curiosity import curiosity_net


num_inputs = 4
num_outputs = 2
state_pl = tf.placeholder(tf.float32, shape = [None, num_inputs])

hidden_1 = tf.layers.dense(state_pl, 30, activation=tf.nn.leaky_relu)
hidden_2 = tf.layers.dense(hidden_1, 30, activation = tf.nn.leaky_relu)
output = tf.layers.dense(hidden_2, num_outputs)

p_output = tf.nn.softmax(output)

action_pl = tf.placeholder(tf.int32, shape = [None, ])
reward_pl = tf.placeholder(tf.float32, shape = [None, ])


def discount(arr, discount_rate):
    result = []
    sum = 0
    for i in range(0, len(arr)):
        sum = discount_rate * sum + arr[len(arr) - 1 - i]
        result.insert(0, sum)
    result -= np.mean(result)
    result /= np.std(result)
    return result


with tf.variable_scope("training"):
    one_hot = tf.one_hot(action_pl, num_outputs)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot, logits = output)
    loss = cross_entropy * reward_pl
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    training_op = optimizer.minimize(loss)

episodes = 0
render = False
env = gym.make('CartPole-v0')
env._max_episode_steps = 9001
state = env.reset()
total_reward = 0
with tf.Session() as session:
    tf.global_variables_initializer().run()
    states = []
    actions = []
    rewards = []
    while True:
        if render:
            env.render()
        action = session.run(p_output, feed_dict = {state_pl: state[np.newaxis, :]})[0]
        action = np.random.choice(np.arange(num_outputs), p = action)

        # rand = np.random.random(1)
        # if action[0]>=action[1]:
        #     action = 0
        # elif action[1]>action[0]:
        #     action = 1
        actions.append(action)
        states.append(state)
        #print(action)

        new_state, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward+=reward
        # session.run(training_op, feed_dict = {state_pl: state,
        #                                       action_pl: [action],
        #                                       reward_pl: [reward]})
        state = new_state
        #env.render()
        if done:
            print (episodes, total_reward)
            rewards = discount(rewards, 0.99)
            session.run(training_op, feed_dict = {state_pl:states,
                                                      action_pl:actions,
                                                      reward_pl:rewards})
            if episodes > 100:
                render = True
            episodes += 1
            total_reward = 0
            state = env.reset()

            states = []
            actions = []
            rewards = []
=======
import tensorflow as tf
import numpy as np
import gym

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
    
    def stuff(self):
        return (self.observations, self.actions, self.processed_rewards)

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
            loss = agent.train(*rollout.stuff(), sess)
            if (i % 5 == 0):
                print(f"Step {i}: Loss of {loss*100:.2f}%, reward of {sum(rollout.rewards)}")
    

if __name__ == "__main__":
    main()
