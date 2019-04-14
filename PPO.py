import tensorflow as tf
import numpy as np
from sys import argv
import gym

from curiosity.curiosity_net import CuriosityNet


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

class PPO:
    def __init__(self, state_dim, action_dim, action_bound, lr, epochs, batch_size, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = lr

        self.state_pl = tf.placeholder(tf.float32, [None] + list(self.state_dim), 'state')
        self.action_pl = tf.placeholder(tf.float32, [None, self.action_dim], 'action')
        self.advantage_pl = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.reward_pl = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state_pl, "actions": self.action_pl,
                                                           "rewards": self.reward_pl, "advantage": self.advantage_pl})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(epochs)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()

        pi_old, vf_old, pi_old_params = self._build_network(batch["state"], 'oldpi')
        pi, self.v, pi_params = self._build_network(batch["state"], 'pi')
        pi_eval, self.vf_eval, _ = self._build_network(self.state_pl, 'pi', reuse=True)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.global_step = tf.train.get_or_create_global_step()

        epsilon_decay = tf.train.polynomial_decay(epsilon, self.global_step, 1e5, 0.01, power=0.0)
        with tf.variable_scope('actor_loss'):
            ratio = tf.exp(tf.log(tf.maximum(pi.prob(batch["actions"]), 1e-6)) -
                           tf.maximum(pi_old.prob(batch["actions"]), 1e-6))
            ratio = tf.clip_by_value(ratio, 0, 10)
            surr1 = batch["advantage"] * ratio
            surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
            self.loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))

        with tf.variable_scope('critic_loss'):
            clipped_val = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
            surr1 = tf.squared_difference(clipped_val, batch["rewards"])
            surr2 = tf.squared_difference(self.v, batch["rewards"])
            self.loss_val = tf.reduce_mean(tf.maximum(surr1, surr2)) * 0.5

        with tf.variable_scope('train'):
            self.total_loss = self.loss_pi + self.loss_val
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step, var_list=pi_params)

        with tf.variable_scope('update_old'):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]

    def train(self, sess, s, a, r, adv):
        sess.run([self.update_pi_old_op, self.iterator.initializer],
                 feed_dict={self.state_pl: s, self.action_pl: a, self.reward_pl: r, self.advantage_pl: adv})

        print("====Starting Training====")
        while True:
            try:
                sess.run(self.train_op)
            except tf.errors.OutOfRangeError:
                break
        print("====Finished Training====")

    def _build_network(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(0.001)
        with tf.variable_scope(name, reuse=reuse):
            layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")

            with tf.variable_scope("policy"):
                mu = tf.layers.dense(layer_2, self.action_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
                log_sigma = tf.get_variable(name="pi_sigma", shape=self.action_dim, initializer=tf.zeros_initializer())
                dist = tf.distributions.Normal(loc=mu * self.action_bound, scale=tf.maximum(tf.exp(log_sigma), 0.0))

            with tf.variable_scope("critic"):
                vf = tf.layers.dense(layer_2, 1, kernel_regularizer=w_reg, name="vf_output")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, vf, params

    def get_action(self, sess, state):
        action = sess.run(self.sample_op, {self.state_pl: state[np.newaxis, :]})
        return action[0]

    def get_value(self, sess, state):
        value = sess.run(self.vf_eval, {self.state_pl: state[np.newaxis, :]})
        return np.squeeze(value)


def main():
    use_curiosity = False
    use_display = False
    curiosity_size = 10
    if len(argv) > 1:
        if "use_curiosity" in argv:
            use_curiosity = True
        if "use_display" in argv:
            use_display = True

    env = gym.make("Pendulum-v0")
    # env.seed(0)
    gamma_const = 0.99
    lambda_const = 0.95

    state_size = env.observation_space.shape
    action_size = env.action_space.shape[0]
    action_bound = (env.action_space.high - env.action_space.low) / 2

    train_step, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
    

    # np.random.seed(1)
    # tf.set_random_seed(1)
    with tf.Session() as sess:
        ppo = PPO(state_size, curiosity_size, action_bound, 1e-4, 10, 32, 0.1)
        
        if use_curiosity:
            curiosity_discount = 100.0
            curiosity = CuriosityNet(input_size, output_size, 1)

        tf.global_variables_initializer().run()

        for episode in range(10000):
            obs = env.reset()
            total_reward, num_steps = 0, 0

            while True:
                a = ppo.get_action(sess, obs)
                v = ppo.get_value(sess, obs)

                if train_step == 2000:
                    rewards = np.array(buffer_r)
                    rewards = np.clip(rewards / np.std(rewards), -10, 10)

                    v_final = [v * (1 - terminal)]
                    values = np.array(buffer_v + v_final)
                    terminals = np.array(buffer_terminal + [terminal])

                    delta = rewards + gamma_const * values[1:] * (1 - terminals[1:]) - values[:-1]
                    advantage = discount(delta, gamma_const * lambda_const, terminals)
                    returns = advantage + np.array(buffer_v)
                    advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                    bs, ba, br, badv = np.reshape(buffer_s, (train_step,) + ppo.state_dim), np.vstack(buffer_a), \
                                       np.vstack(returns), np.vstack(advantage)

                    ppo.train(sess, bs, ba, br, badv)
                    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
                    train_step = 0

                buffer_s.append(obs)
                buffer_a.append(a)
                buffer_v.append(v)
                buffer_terminal.append(terminal)

                a = np.clip(a, env.action_space.low, env.action_space.high)
                obs, r, terminal, _ = env.step(a)
                buffer_r.append(r)

                total_reward += r
                num_steps += 1
                train_step += 1

                if terminal:
                    print('Episode: %i' % episode, "| Reward: %.2f" % total_reward, '| Steps: %i' % num_steps)
                    break


if __name__ == '__main__':
    main()