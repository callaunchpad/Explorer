import gym
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PolicyGradient:

    def __init__(self):
        #self.data = []
        self.name = "policy"

    def policy_gradient(self, initializer, num_episodes):

        def discount(array, rate):
            sum = 0
            result = []

            for i in range(0, len(array)):
                sum = sum * rate + array[len(array) - i - 1]
                result.insert(0, sum)

            # NN like rewards between -1 and 1 like their weights
            # normalize

            result -= np.mean(result)
            result /= np.std(result)
            return result

        num_inputs = 4
        num_outputs = 2

        state_pl = tf.placeholder(tf.float32, shape=(None, num_inputs))
        reward_pl = tf.placeholder(tf.float32, shape=[None, ])
        action_pl = tf.placeholder(tf.int32, shape=[None, ])
        hidden1 = tf.layers.dense(state_pl, 30, kernel_initializer=initializer, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(hidden1, 30, kernel_initializer=initializer, activation=tf.nn.leaky_relu)

        '''right or left'''
        output = tf.layers.dense(hidden2, num_outputs)
        p_output = tf.nn.softmax(output)


        with tf.variable_scope("training"):
            one_hot = tf.one_hot(action_pl, num_outputs)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=output)
            loss = cross_entropy * reward_pl
            loss = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer(0.01) #0.01
            training_Op = optimizer.minimize(loss)

        render = False

        '''OpenAI environments'''
        env = gym.make('CartPole-v0')  #'Cartpole'
        env._max_episode_steps = 1000
        state = env.reset()
        total_reward = 0
        data = []

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            states = []
            actions = []
            rewards = []
            episode = 0

            while True:
                if render:
                    env.render()
                action = session.run(p_output, feed_dict={state_pl: state[np.newaxis, :]})[0]
                action = np.random.choice(np.arange(num_outputs), p=action)

                #epsilon = 0.5
                #random_num = np.random.random(1)
                #print(action)

                #if action[0] > action[1]:
                #    action = 0
                #else:
                #    action = 1

                actions.append(action)
                states.append(state)
                new_state, reward, done, info = env.step(action)

                rewards.append(reward)

                total_reward += reward
                #new_state = new_state[np.newaxis, :]
                #session.run(training_Op, feed_dict={state_pl:state, reward_pl:[reward], action_pl:[action]})
                state = new_state
                #env.render()
                if episode > num_episodes:
                    return data
                if done:
                    print(episode, total_reward)
                    #self.data.append((episode, total_reward))
                    data.append((episode, total_reward))

                    rewards = discount(rewards, 0.99)
                    session.run(training_Op, feed_dict={state_pl: states, reward_pl: rewards, action_pl: actions})

                    # if total_reward > 600:
                    #     render = True
                    episode+=1
                    state = env.reset()
                    total_reward = 0

                    states = []
                    rewards = []
                    actions = []

def plot_xy(array):
    x = []
    y = []
    for i in array:
        x.append(i[0])
        y.append(i[1])

    return x, y


pg = PolicyGradient()
random_normal = tf.random_normal_initializer()
xavier = tf.contrib.layers.xavier_initializer()
random_num_data = []
xavier_data = []
range_data = 10

for i in range(range_data):
    random_num_data.append(pg.policy_gradient(random_normal, 80))
    xavier_data.append(pg.policy_gradient(xavier, 80))

random_num_np = np.array(random_num_data)
xavier_np = np.array(xavier_data)
print("Random normal initialization")
print(random_num_np)
print("Xavier initializer")
print(xavier_np)

for i in range(range_data):
    x, y1 = plot_xy(random_num_data[i])
    x, y2 = plot_xy(xavier_data[i])

    plt.subplot(2,1,1)
    plt.plot(y1)
    plt.title("Random Normal and Xavier Comparison")
    plt.ylabel("Random Normal")
    plt.subplot(2,1,2)
    plt.plot(y2)
    plt.ylabel("Xavier")
    plt.xlabel("Episode")

plt.show()
