import tensorflow as tf
import numpy as np
import gym
import architectures as arch
import time

class Rollouts:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.steps = []

    def push(self, state, action, reward, next_state):
        self.steps.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
    
    def sample(self):
        return np.random.choice(self.steps)

class CumulativeRollout(Rollouts):
    def __init__(self, discount_factor):
        super().__init__()
        self.discount_factor = discount_factor
        
    def _compute_cumulative_rewards(self):
        cumulative_rewards = [0] * len(self.steps)
        future_reward = 0
        for i in reversed(range(len(self.steps))):
            future_reward = self.steps[i]['reward'] + future_reward * self.discount_factor
            cumulative_rewards[i] = future_reward
        cumulative_rewards -= np.mean(cumulative_rewards)
        cumulative_rewards /= np.std(cumulative_rewards)
        return cumulative_rewards
    
    def close(self):
        cumulative_rewards = self._compute_cumulative_rewards()
        for i in range(len(self.steps)):
            self.steps[i]['cumulative_reward'] = cumulative_rewards[i]

class ACRollout(Rollouts):
    def __init__(self, discount_factor):
        super().__init__()
        self.discount_factor = discount_factor
        self.end = []

    def push(self, state, action, reward, next_state, end):
        super().push(state, action, reward, next_state)
        self.steps[-1]['end'] = end


class ActorCritic:
    HIDDEN_SIZE_1 = 20
    HIDDEN_SIZE_2 = 20

    def __init__(self, state_dim, num_actions, discount_factor=0.99, tau=0.01, learning_rate=0.00025):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau

        # Rollout time step sample
        self.state = tf.placeholder(tf.float32, shape=[1, state_dim])
        self.action = tf.placeholder(tf.int32, shape=None)
        self.reward = tf.placeholder(tf.float32, shape=None)
        self.next_state = tf.placeholder(tf.float32, [1, state_dim])
        self.critic_next_state_V = tf.placeholder(tf.float32, None)
        self.end = tf.placeholder(tf.float32, None)

        # Critic
        self.critic_state_V = tf.squeeze(arch.feed_forward(
            self.state,
            {
                'output_size': 1,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()],
            },
            'critic_state_V'
        ))
        self.params = tf.trainable_variables('critic_state_V')

        self.target_critic_state_V = tf.squeeze(arch.feed_forward(
            self.state,
            {
                'output_size': 1,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()],
            },
            'target_critic_state_V'
        ))
        self.t_params = tf.trainable_variables('target_critic_state_V')

        self.target_init = [self.t_params[i].assign(self.params[i]) for i in range(len(self.t_params))]
        self.update_target_params = \
            [self.t_params[i].assign(self.params[i] * self.tau +
                                     self.t_params[i] * (1. - self.tau)) for i in range(len(self.t_params))]

        self.critic_advantage = self.reward + self.discount_factor * self.critic_next_state_V * (
                1 - self.end) - self.critic_state_V

        self.critic_loss = 0.5 * tf.reduce_mean(tf.square(self.critic_advantage))

        # Actor
        self.actor_logits = arch.feed_forward(
            self.state,
            {
                'output_size': num_actions,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()],
            },
            'actor_logits'
        )

        self.actor_probs = tf.nn.softmax(self.actor_logits)
        self.action_one_hot = tf.one_hot(self.action, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_one_hot, logits=self.actor_logits)
        self.actor_loss = tf.reduce_mean(cross_entropy * tf.stop_gradient(self.critic_advantage))
        self.actor_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.actor_loss)

        self.critic_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.critic_loss)

    def action_probs(self, state, session):
        feed_dict = {self.state: state}
        return session.run(self.actor_probs, feed_dict=feed_dict)

    def update(self, rollout_step, session):
        critic_next_state_v = session.run(self.target_critic_state_V, feed_dict={self.state: [rollout_step['next_state']]})

        feed_dict = {
            self.state: [rollout_step['state']],
            self.action: rollout_step['action'],
            self.reward: rollout_step['reward'],
            self.next_state: [rollout_step['next_state']],
            self.critic_next_state_V: critic_next_state_v,
            self.end: rollout_step['end']
        }
        loss1, loss2, _, _ = session.run([self.actor_loss, self.critic_loss, self.actor_op, self.critic_op], feed_dict=feed_dict)
        return loss1, loss2

    def value(self, state, session):
        feed_dict = {self.state: [state]}
        return session.run(self.critic_state_V, feed_dict=feed_dict)

    def init_target(self, sess):
        sess.run(self.target_init)

    def update_target(self, sess):
        sess.run(self.update_target_params)

def lunar_lander(learning_rate, discount_factor, tau):
    file_dir = "/Users/Praveen/Desktop/" + str(time.time_ns())
    with tf.Session() as sess:
        env = gym.make("LunarLander-v2")
        env._max_episode_steps = int(1E3)
        agent = ActorCritic(state_dim=8, num_actions=4, discount_factor=0.99, tau=0.01, learning_rate=0.00025) 
        tf.global_variables_initializer().run()
        agent.init_target(sess)
        rollout = ACRollout(0.99)
        for i in range(25000):
            if i % 30 == 0:
                rollout.reset()
            total_reward = 0
            next_state_obs = env.reset()
            done = False
            num_steps = 0
            while not done:
                num_steps += 1
                state_obs = next_state_obs
                action = np.random.choice(np.arange(4), p=agent.action_probs(state_obs[np.newaxis, :], sess)[0])
                next_state_obs, reward, done, _ = env.step(action)
                rollout.push(state_obs, action, reward, next_state_obs, int(done))
                total_reward += reward
                if i % 25 == 0:
                    env.render()

            for j in range(100):
                loss = agent.update(rollout.sample(), sess)
                agent.update_target(sess)
            print(f"Step {i}: Loss of {loss[0]}, {loss[1]}, reward of {total_reward}. Took {num_steps} to end.")

def main():
    with tf.Session() as sess:
        env = gym.make("LunarLander-v2")
        env._max_episode_steps = int(1E3)
        agent = ActorCritic(state_dim=8, num_actions=4, discount_factor=0.99, tau=0.01, learning_rate=0.00025) 
        tf.global_variables_initializer().run()
        agent.init_target(sess)
        rollout = ACRollout(0.99)
        for i in range(25000):
            if i % 30 == 0:
                rollout.reset()
            total_reward = 0
            next_state_obs = env.reset()
            done = False
            num_steps = 0
            while not done:
                num_steps += 1
                state_obs = next_state_obs
                action = np.random.choice(np.arange(4), p=agent.action_probs(state_obs[np.newaxis, :], sess)[0])
                next_state_obs, reward, done, _ = env.step(action)
                rollout.push(state_obs, action, reward, next_state_obs, int(done))
                total_reward += reward
                if i % 25 == 0:
                    env.render()

            for j in range(100):
                loss = agent.update(rollout.sample(), sess)
                agent.update_target(sess)
            print(f"Step {i}: Loss of {loss[0]}, {loss[1]}, reward of {total_reward}. Took {num_steps} to end.")


if __name__ == "__main__":
    main()
