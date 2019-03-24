import tensorflow as tf
import numpy as np
import gym
import architectures as arch


class Rollouts:
    """
    Rollouts contains the following arrays
    - states: States from state_0 ... state_n
    - actions: Actions from action_0 ... action_n
    - rewards: Rewards from reward_0 ... reward_n
    - next_states: States from state_1 ... state_n+1 (terminal state)
    """

    def __init__(self, discount_factor):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.cumulative_rewards = []
        self.discount_factor = discount_factor

    def push(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def _compute_cumulative_rewards(self):
        # Apply discounted accumulation
        cumulative_rewards = self.rewards[:]
        future_reward = 0
        for i in reversed(range(len(cumulative_rewards))):
            future_reward = cumulative_rewards[i] + future_reward * self.discount_factor
            cumulative_rewards[i] = future_reward
        # Normalize
        cumulative_rewards -= np.mean(cumulative_rewards)
        cumulative_rewards /= np.std(cumulative_rewards)
        return cumulative_rewards

    # def close(self):
    #     self.cumulative_rewards = self._compute_cumulative_rewards()
    #
    # def sample(self):
    #     """
    #         Returns sampled tuple (state, action, reward, next_state)
    #         or (state, action, reward, next_state, cumulative_reward)
    #     """
    #     i = np.random.randint(len(self.states))
    #     return self.states[i], self.actions[i], self.rewards[i], self.next_states[i]


class ACRollout(Rollouts):
    """
        Rollout which also stores a critic network's value predictions for the next state at each time step
    """

    def __init__(self, discount_factor):
        super().__init__(discount_factor)
        self.end = []

    def push(self, *args):
        state, action, reward, s_, end = args
        super().push(state, action, reward, s_)
        self.end.append(end)

    def sample(self):
        """
            Returns sampled tuple (state, action, reward, next_state, next_state_value_pred)
        """
        i = np.random.randint(0, len(self.states))
        return self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.end[i]


class ActorCritic:
    HIDDEN_SIZE_1 = 20
    HIDDEN_SIZE_2 = 20
    LR = 0.00025

    def __init__(self, state_dim, num_actions, discount_factor, tau):
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
        self.log_prob = tf.log(self.actor_probs[0, self.action] + 1e-6)
        self.actor_loss = -tf.reduce_mean(self.log_prob * tf.stop_gradient(self.critic_loss))
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.actor_loss)

        self.critic_op = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.critic_loss)

    def action_probs(self, state, session):
        feed_dict = {self.state: state}
        return session.run(self.actor_probs, feed_dict=feed_dict)

    def update(self, rollout_sample, session):
        state, action, reward, next_state, end = rollout_sample
        critic_next_state_v = session.run(self.critic_state_V, feed_dict={self.state: [next_state]})

        feed_dict = {
            self.state: [state],
            self.action: action,
            self.reward: reward,
            self.next_state: [next_state],
            self.critic_next_state_V: critic_next_state_v,
            self.end: end
        }
        loss1, loss2, _, _ = session.run([self.actor_loss, self.critic_loss, self.actor_train_op, self.critic_op], feed_dict=feed_dict)
        return loss1, loss2

    def value(self, state, session):
        feed_dict = {self.state: [state]}
        return session.run(self.critic_state_V, feed_dict=feed_dict)


def main():
    with tf.Session() as sess:
        env = gym.make("CartPole-v0")
        env._max_episode_steps = 10000
        agent = ActorCritic(4, 2, 0.99, 0.01)

        tf.global_variables_initializer().run()
        rollout = ACRollout(0.99)
        for i in range(25000):
            total_reward = 0
            next_state_obs = env.reset()
            done = False
            if i % 30 == 0:
                rollout = ACRollout(0.99)

            while not done:
                state_obs = next_state_obs
                action = np.random.choice(np.arange(2), p=agent.action_probs(state_obs[np.newaxis, :], sess)[0])
                next_state_obs, reward, done, _ = env.step(action)
                rollout.push(state_obs, action, reward, next_state_obs, int(done))
                total_reward += reward
                if i % 25 == 0:
                    env.render()

            for j in range(1000):
                loss = agent.update(rollout.sample(), sess)
            print(f"Step {i}: Loss of {loss[0]}, {loss[1]}, reward of {total_reward}")


if __name__ == "__main__":
    main()
