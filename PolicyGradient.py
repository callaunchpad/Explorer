import gym
import tensorflow as tf
import numpy as np
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

        actions.append(action)
        states.append(state)
        #print(action)

        new_state, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward+=reward
        state = new_state
        if done:
            print (episodes, total_reward)
            rewards = discount(rewards, 0.99)
            session.run(training_op, feed_dict = {state_pl:states,
                                                      action_pl:actions,
                                                      reward_pl:rewards})
            if episodes > 200 or total_reward > 800:
                render = True
            episodes += 1
            total_reward = 0
            state = env.reset()

            states = []
            actions = []
            rewards = []
