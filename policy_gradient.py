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
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
    
    def push(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
    
    def cumulative_rewards(self, discount_factor):
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

class ActorCritic(Agent):
    
    HIDDEN_SIZE_1 = 30
    HIDDEN_SIZE_2 = 30
    ACTOR_LEARNING_RATE = 0.01
    CRITIC_LEARNING_RATE = 0.01
    
    def __init__(self, state_dim, num_actions, discount_factor):
        self.discount_factor = discount_factor
        
        # Rollout
        self.states = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions = tf.placeholder(tf.int32, shape=[None,])
        self.rewards = tf.placeholder(tf.float32, shape=[None,])
        self.cumulative_rewards = tf.placeholder(tf.float32, shape=[None,])
        self.next_states = tf.placeholder(tf.float32, [None, state_dim])
        
        # Critic
        self.critic_state_V = arch.feed_forward(
            self.states,
            {
                'output_size': 1,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.initializers.random_normal, tf.initializers.random_normal],
            },
            'critic_state_V'
        )
        self.critic_state_V = tf.reshape(self.critic_state_V, [-1])
        
        self.critic_next_state_V = tf.concat([self.critic_state_V[1:], tf.zeros(1)], axis=0)
        self.critic_loss = tf.losses.mean_squared_error(self.cumulative_rewards, self.critic_state_V)
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate=self.CRITIC_LEARNING_RATE).minimize(self.critic_loss)
        
        # Advantage
        self.critic_advantage = self.rewards + self.discount_factor * self.critic_next_state_V - self.critic_state_V
        
        # Actor
        self.actor_logits = arch.feed_forward(
            self.states,
            {
                'output_size': num_actions,
                'hidden_sizes': [self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2],
                'activations': [tf.nn.relu, tf.nn.relu],
                'kernel_initializers': [tf.initializers.random_normal, tf.initializers.random_normal],
            },
            'actor_logits'
        )
        self.actor_probs = tf.nn.softmax(self.actor_logits)
        self.actions_one_hot = tf.one_hot(self.actions, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.actions_one_hot, logits=self.actor_logits)
        self.actor_loss = tf.reduce_mean(cross_entropy * self.critic_advantage)
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.ACTOR_LEARNING_RATE).minimize(self.actor_loss)
    
    def action_probs(self, state, session):
        feed_dict = {self.states: state}
        return session.run(self.actor_probs, feed_dict=feed_dict)
    
    def update(self, rollout, session):
        feed_dict = {
            self.states: rollout.states,
            self.actions: rollout.actions,
            self.rewards: rollout.rewards,
            self.cumulative_rewards: rollout.cumulative_rewards(self.discount_factor),
            self.next_states: rollout.next_states
        }
        actor_loss, critic_loss, _, _ = session.run([self.actor_loss, self.critic_loss, self.actor_train_op, self.critic_train_op], feed_dict=feed_dict)
        return actor_loss, critic_loss

def main():
    with tf.Session() as sess:
        env = gym.make("CartPole-v0")
        env._max_episode_steps = 1E3
        agent = ActorCritic(4, 2, 0.99)
        tf.global_variables_initializer().run()
        for i in range(250):
            state_obs = env.reset()
            done = False
            rollout = Rollout()
            while (not done):
                old_state_obs = state_obs
                action = np.random.choice(np.arange(2), p=agent.action_probs(old_state_obs[np.newaxis,:], sess)[0])
                state_obs, reward, done, _ = env.step(action)
                rollout.push(old_state_obs, action, reward, state_obs)
                if (i % 25 == 0):
                    env.render()
            loss = agent.update(rollout, sess)
            if (i % 5 == 0):
                print(f"Step {i}: Loss of (actor: {loss[0]*100:.2f}%, {loss[1]*100:.2f}%), reward of {sum(rollout.rewards)}")
    

if __name__ == "__main__":
    main()