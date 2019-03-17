import tensorflow as tf
import numpy as np
import gym

import architectures as arch

class Agent:
    """
    Abstract class for an Agent
    """
    def action_probs(self, state, session):
        raise NotImplementedError("action_probs not implemented")
    
    def update(self, rollout, session):
        raise NotImplementedError("update not implemented")


class Rollout:
    """
    Rollout contains the following arrays
    - states: States from state_0 ... state_n
    - actions: Actions from action_0 ... action_n
    - rewards: Rewards from reward_0 ... reward_n
    - next_states: States from state_1 ... state_n+1 (terminal state)
    """
    
    def __init__(self, discount_factor=1):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
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
            future_reward = cumulative_rewards[i] + future_reward * discount_factor
            cumulative_rewards[i] = future_reward
        # Normalize
        cumulative_rewards -= np.mean(cumulative_rewards)
        cumulative_rewards /= np.std(cumulative_rewards)
        return cumulative_rewards

    def close(self):
        self.cumulative_rewards = self._compute_cumulative_rewards()

    def sample(self):
        """
            Returns sampled tuple (state, action, reward, next_state)
            or (state, action, reward, next_state, cumulative_reward)
        """
        i = np.random.randint(len(self.states))
        '''if return_cumulative_reward:
            if self.cumulative_rewards is None:
                raise ValueError('Cumulative rewards can only be returned once rollout is closed.')
            return (self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.cumulative_rewards[i])
        else:'''
        return (self.states[i], self.actions[i], self.rewards[i], self.next_states[i])
class ACRollout(Rollout):
    """
        Rollout which also stores a critic network's value predictions for the next state at each time step
    """
    def __init__(self, ac_agent, session, discount_factor=1):
        super().__init__(discount_factor)
        self.critic_next_state_Vs = []
        self.ac_agent = ac_agent
        self.session = session
    def push(self, state, action, reward, next_state):
        super().push(state, action, reward, next_state)
        self.critic_next_state_Vs.append(self.ac_agent.value(next_state, self.session)[0])
    def sample(self):
        """
            Returns sampled tuple (state, action, reward, next_state, next_state_value_pred)
        """
        i = np.random.randint(len(self.states))
        return (self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.critic_next_state_Vs[i])



class ActorCritic(Agent):
    
    HIDDEN_SIZE_1 = 30
    HIDDEN_SIZE_2 = 30
    ACTOR_LEARNING_RATE = 0.001
    CRITIC_LEARNING_RATE = 0.01
    
    def __init__(self, state_dim, num_actions, discount_factor):
        self.discount_factor = discount_factor
        
        # Rollout time step sample
        self.state = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.action = tf.placeholder(tf.int32, shape=[None,])
        self.reward = tf.placeholder(tf.float32, shape=[None,])
        self.next_state = tf.placeholder(tf.float32, [None, state_dim])
        self.critic_next_state_V = tf.placeholder(tf.float32, [None, ])

        # Critic
        self.critic_state_V = arch.feed_forward(
            self.state,
            {
                'output_size': 1,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.initializers.random_normal, tf.initializers.random_normal],
            },
            'critic_state_V'
        )

        self.critic_state_V = tf.reshape(self.critic_state_V, [-1])
        self.critic_advantage = self.reward - (self.critic_state_V - self.critic_next_state_V)
        
        
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_advantage)) # tf.losses.mean_squared_error(self.cumulative_rewards, self.critic_state_V)
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate=self.CRITIC_LEARNING_RATE).minimize(self.critic_loss)
        
        # Advantage
        
        # Actor
        self.actor_logits = arch.feed_forward(
            self.state,
            {
                'output_size': num_actions,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.initializers.random_normal, tf.initializers.random_normal],
            },
            'actor_logits'
        )

        self.actor_probs = tf.nn.softmax(self.actor_logits)
        self.action_one_hot = tf.one_hot(self.action, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_one_hot, logits=self.actor_logits)
        self.actor_loss = tf.reduce_mean(cross_entropy * self.critic_advantage)
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.ACTOR_LEARNING_RATE).minimize(self.actor_loss)
    
    def action_probs(self, state, session):
        feed_dict = {self.state: state}
        return session.run(self.actor_probs, feed_dict=feed_dict)
    
    def update(self, rollout_sample, session):
        state, action, reward, next_state, critic_next_state_V = rollout_sample
        
        feed_dict = {
            self.state: [state],
            self.action: [action],
            self.reward: [reward],
            self.next_state: [next_state],
            self.critic_next_state_V: [critic_next_state_V]
        }
        actor_loss, critic_loss, _, _ = session.run([self.actor_loss, self.critic_loss, self.actor_train_op, self.critic_train_op], feed_dict=feed_dict)
        return actor_loss, critic_loss

    def value(self, state, session):
        feed_dict = {self.state: [state]}
        return session.run(self.critic_state_V, feed_dict=feed_dict)

def main():
    with tf.Session() as sess:
        env = gym.make("CartPole-v0")
        env._max_episode_steps = 1E3
        agent = ActorCritic(4, 2, 0.99)
        tf.global_variables_initializer().run()
        rollout = ACRollout(agent, sess)
        for i in range(25000):
            if i % 5 == 0:
                rollout = ACRollout(agent, sess)
                next_state_obs = env.reset()
                done = False
                while (not done):
                    state_obs = next_state_obs
                    action = np.random.choice(np.arange(2), p=agent.action_probs(state_obs[np.newaxis,:], sess)[0])
                    next_state_obs, reward, done, _ = env.step(action)
                    rollout.push(state_obs, action, reward, next_state_obs)
                    if (i % 25 == 0):
                        env.render()

            loss = agent.update(rollout.sample(), sess)
            if (i % 5 == 0):
                print(f"Step {i}: Loss of (actor: {loss[0]}, critic: {loss[1]}), reward of {sum(rollout.rewards)}")
    

if __name__ == "__main__":
    main()
