import tensorflow as tf
import numpy as np
import gym
import os
import cv2
import envs

class PPO:
    def __init__(self, state_dim, action_dim, lr, epochs, batch_size, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.state_pl = tf.placeholder(tf.float32, [None] + list(self.state_dim), 'state')
        self.action_pl = tf.placeholder(tf.int32, [None, 1], 'action')
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

        self.epsilon_decay = tf.train.polynomial_decay(epsilon, self.global_step, 1e5, 0.01, power=0.0)
        with tf.variable_scope('actor_loss'):
            ratio = tf.exp(pi.log_prob(batch["actions"]) - pi_old.log_prob(batch["actions"]))
            ratio = tf.clip_by_value(ratio, 0, 10)
            surr1 = batch["advantage"] * ratio
            surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - self.epsilon_decay, 1 + self.epsilon_decay)
            self.loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))

        with tf.variable_scope('critic_loss'):
            clipped_val = vf_old + tf.clip_by_value(self.v - vf_old, -self.epsilon_decay, self.epsilon_decay)
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
        total_loss, total_loss_pi, total_loss_val, counter = 0, 0, 0, 0
        while True:
            try:
                _, _, loss, loss_pi, loss_val = sess.run([self.train_op, self.global_step, self.total_loss, self.loss_pi, self.loss_val])
                total_loss += loss
                total_loss_pi += loss_pi
                total_loss_val += loss_val
                counter += 1
            except tf.errors.OutOfRangeError:
                break
        epsilon_decay = sess.run(self.epsilon_decay)
        print('Total Loss: %4f' % (total_loss / counter), "| Actor Loss: %4f" % (total_loss_pi / counter),
              'Critic Loss: %4f' % (total_loss_val / counter), 'Epsilon Decay: %4f' % epsilon_decay)
        print("====Finished Training====")

    def _build_network(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(0.01)
        with tf.variable_scope(name, reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=state_in, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            state_in = tf.layers.flatten(conv3)

            with tf.variable_scope("policy"):
                layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg)
                layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg)
                # layer_3 = tf.layers.dense(layer_2, 400, tf.nn.relu, kernel_regularizer=w_reg)
                a_logits = tf.layers.dense(layer_2, self.action_dim, kernel_regularizer=w_reg, name="pi_logits")
                dist = tf.distributions.Categorical(logits=a_logits)

            with tf.variable_scope("critic"):
                layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg)
                layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg)
                # layer_3 = tf.layers.dense(layer_2, 400, tf.nn.relu, kernel_regularizer=w_reg)
                vf = tf.layers.dense(layer_2, 1, kernel_regularizer=w_reg, name="vf_output")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, vf, params

    def get_action(self, sess, state):
        action = sess.run(self.sample_op, {self.state_pl: state[np.newaxis, :]})
        return action[0]

    def get_value(self, sess, state):
        value = sess.run(self.vf_eval, {self.state_pl: state[np.newaxis, :]})
        return np.squeeze(value)


def preprocess(input_observation, prev_processed_observation):
    return input_observation
    # obs = cv2.cvtColor(input_observation, cv2.COLOR_RGB2GRAY)
    # obs[obs != 0] = 1
    # obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

    # # subtract the previous frame from the current one so we are only processing on changes in the game
    # if prev_processed_observation is not None:
    #     return np.maximum(obs[:, :, None], 0.6 * prev_processed_observation)
    # return obs[:, :, None]


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
        
        
def main():
    env = envs.GradEnv()
    # env.seed(0)
    gamma_const = 0.99
    lambda_const = 0.95

    state_size = env.observation_space.shape
    action_size = env.action_space.n

    train_step, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []

    # np.random.seed(1)
    # tf.set_random_seed(1)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': True})
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=config) as sess:
        ppo = PPO(state_size, action_size, 0.0001, 10, 32, 0.1)
        tf.global_variables_initializer().run()

        for episode in range(int(1E12)):
            obs = env.reset()
            prep_obs = preprocess(obs, None)
            total_reward, num_steps = 0, 0
            
            if episode % 1000 == 0:
                tf.train.Saver().save(sess, './ye')

            while True:
                if episode % 25 == 0:
                    env.render()
                a = ppo.get_action(sess, prep_obs)
                v = ppo.get_value(sess, prep_obs)

                if train_step == 2000:
                    rewards = np.array(buffer_r)
                    # rewards = np.clip(rewards / np.std(rewards), -10, 10)

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

                buffer_s.append(prep_obs)
                buffer_a.append(a)
                buffer_v.append(v)
                buffer_terminal.append(terminal)

                new_obs, r, terminal, _ = env.step(a)
                buffer_r.append(r)

                total_reward += r
                num_steps += 1
                train_step += 1
                prep_obs = preprocess(new_obs, prep_obs)

                obs = new_obs

                if terminal:
                    print('Episode: %i' % episode, "| Reward: %.2f" % total_reward, '| Steps: %i' % num_steps)
                    break


if __name__ == '__main__':
    main()
