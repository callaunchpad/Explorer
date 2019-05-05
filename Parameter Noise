import gym
import tensorflow as tf
import copy
import numpy as np
import scipy


class ParameterPPO:
    def __init__(self, state_dim, action_dim, action_bound, lr, gamma, clip_val):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_val = clip_val
        self.action_bound = action_bound

        self.state_pl = tf.placeholder(tf.float32, [None, ] + list(state_dim), 'states')
        self.action_pl = tf.placeholder(tf.float32, shape=[None, action_dim], name="actions")
        self.return_pl = tf.placeholder(tf.float32, [None, 1], 'return')
        self.advantage_pl = tf.placeholder(tf.float32, [None, 1], 'advantages')

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state_pl,
                                                           "actions": self.action_pl,
                                                           "return": self.return_pl,
                                                           "advantage": self.advantage_pl})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(32)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(10)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()

        pi_probs, self.v_pred, pi_params = self._build_network('pi', batch["state"], False)
        oldpi_probs, _, oldpi_params = self._build_network('old_pi', batch["state"], False)
        pi_eval, self.pi_value, _ = self._build_network('pi', self.state_pl, True)
        self.act_random = tf.squeeze(pi_eval.sample(1), axis=1)
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
        # act_one_hot = tf.one_hot(self.action_pl, self.action_dim)
        # act_prob = tf.reduce_sum(act_one_hot * pi_probs, axis=1)
        # oldact_one_hot = tf.one_hot(self.action_pl, self.action_dim)
        # old_act_prob = tf.reduce_sum(oldact_one_hot * oldpi_probs, axis=1)
        act_prob = pi_probs.log_prob(batch["actions"])
        old_act_probs = oldpi_probs.log_prob(batch["actions"])
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(batch["return"] - self.v_pred))
        with tf.variable_scope('actor_loss'):
            rt = tf.exp(act_prob - old_act_probs)
            clipped_rt = tf.clip_by_value(rt, 1 - self.clip_val, 1 + clip_val)
            self.actor_loss = tf.reduce_mean(tf.minimum(batch["advantage"] * rt,
                                                        batch["advantage"] * clipped_rt))
        # with tf.variable_scope('entropy_loss'):
        #    entropy = -tf.reduce_sum(pi_probs * tf.log(pi_probs + 1e-5), axis=1)
        #    self.entropy_loss = tf.reduce_mean(entropy, axis=0)
        with tf.variable_scope('loss'):
            self.total_loss = -self.actor_loss + self.critic_loss * 0.5  # - 0.01 * self.entropy_loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_params)

    def _build_network(self, scope, state_in, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('policy'):
                hidden1_layer = tf.layers.dense(state_in, 400, activation=tf.nn.relu)
                hidden2_layer = tf.layers.dense(hidden1_layer, 400, activation=tf.nn.relu)
                mu = tf.layers.dense(hidden2_layer, self.action_dim, activation=tf.nn.tanh)
                # probs = tf.nn.softmax(output)
                log_sigma = tf.get_variable("pi_sigma", shape=self.action_dim, initializer=tf.zeros_initializer())
                dist = tf.distributions.Normal(loc=mu * self.action_bound, scale=tf.maximum(tf.exp(log_sigma), 0.0))

            with tf.variable_scope('critic'):
                hidden1_layer = tf.layers.dense(state_in, 400, activation=tf.nn.relu)
                hidden2_layer = tf.layers.dense(hidden1_layer, 400, activation=tf.nn.relu)
                v_pred = tf.layers.dense(hidden2_layer, 1)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return dist, v_pred, params

    def get_action(self, sess, state):
        return sess.run(self.act_random, feed_dict={self.state_pl: state})

    def get_value(self, sess, state):
        return sess.run(self.pi_value, feed_dict={self.state_pl: state})

    def update_params(self, sess):

        # Noisy parameters here
        old_params = []
        for param in self.pi_params:
            old_params.append(param)
            noisy_param = param + np.random.normal(0, self.noise_std, size=param.shape)
            param.assign(noisy_param)
        sess.run(self.update_oldpi_op)
        i = 0
        for param in self.pi_params:
            param.assign(old_params[i])
            i += 1

    def train(self, sess, states, actions, returns, advantages):
        feed_dict = {self.state_pl: states, self.return_pl: returns, self.advantage_pl: advantages,
                     self.action_pl: actions}
        sess.run([self.update_oldpi_op, self.iterator.initializer], feed_dict=feed_dict)
        while True:
            try:
                sess.run(self.train_op)
            except tf.errors.OutOfRangeError:
                break


def discount(x, gamma, terminal_array=None):
    if terminal_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]


# def get_gaes(gamma, rewards, v_preds, v_preds_next):
#     deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
#     gaes = copy.deepcopy(deltas)
#     for t in reversed(range(len(gaes) - 1)):
#         gaes[t] = gaes[t] + gamma * gaes[t + 1]
#     return gaes

def main():
    env = gym.make('Pendulum-v0')
    # env._max_episode_steps = 9000
    env.seed(0)
    gamma = 0.99
    states = []
    actions = []
    rewards = []
    values = []
    terminals = []
    train_step, done = 0, False;
    with tf.Session() as sess:
        state_size = env.observation_space.shape
        action_size = env.action_space.shape[0]
        action_bound = (env.action_space.high - env.action_space.low) / 2
        agent = ParameterPPO(state_size, action_size, action_bound, 1e-4, gamma, 0.2)
        tf.global_variables_initializer().run()
        for iteration in range(10000):
            total_reward = 0
            obs = env.reset()
            while True:
                # if iteration % 100 == 0:
                #   env.render()
                action = agent.get_action(sess, obs[np.newaxis, :])
                value = agent.get_value(sess, obs[np.newaxis, :])[0][0]
                new_obs, reward, new_done, info = env.step(action[0])
                if train_step == 2000:
                    rewards = np.array(rewards)
                    rewards = np.clip(rewards / (np.std(rewards) + 1e-6), -10, 10)
                    valuez = np.array(values + [value * (1 - done)])
                    terminals = np.array(terminals + [done])
                    delta = rewards + gamma * valuez[1:] * (1 - terminals[1:]) - valuez[: -1]
                    advantages = discount(delta, gamma * 0.95, terminals)
                    returns = advantages + np.array(values)
                    advantages -= np.mean(advantages)
                    advantages /= np.std(advantages) + 1e-6
                    actions = np.vstack(actions)
                    returns = np.vstack(returns)
                    advantages = np.vstack(advantages)
                    states = np.array(states)
                    states = np.reshape(states, (train_step,) + agent.state_dim)
                    agent.train(sess, states, actions, returns, advantages)

                    states = []
                    actions = []
                    rewards = []
                    values = []
                    terminals = []
                    train_step = 0

                total_reward += reward
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(done)
                values.append(value)
                done = new_done
                obs = new_obs
                train_step += 1;
                if done:
                    # values_next = values[1:] + [0]
                    print(total_reward, iteration)
                    break

            # gaes = np.array(get_gaes(gamma, rewards, v_preds, v_preds_next))
            # gaes = (gaes - gaes.mean()) / gaes.std()


#             states = np.array(states)
#             actions = np.array(actions).astype(dtype=np.int32)
#             rewards = np.array(rewards).astype(dtype=np.float32)
#             v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
#             agent.update_params(sess)
#             inp = [states, actions, rewards, gaes, v_preds_next]
#             # train
#             for epoch in range(4):
#                 sample_indices = np.random.randint(low=0, high=states.shape[0], size=64)  # indices are in [low, high)
#                 sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
#                 agent.train(sess, sampled_inp[0], sampled_inp[1], sampled_inp[2],
#                             sampled_inp[3], sampled_inp[4])
if __name__ == "__main__":
    main()
