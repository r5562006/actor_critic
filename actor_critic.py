import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 創建環境
env = gym.make('CartPole-v1')

# 建立Actor和Critic網絡
def build_actor(state_shape, action_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(24, input_shape=state_shape, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_shape, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy')
    return model

def build_critic(state_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(24, input_shape=state_shape, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mse')
    return model

# 設置參數
state_shape = env.observation_space.shape
action_shape = env.action_space.n
actor = build_actor(state_shape, action_shape)
critic = build_critic(state_shape)

gamma = 0.99
episodes = 1000

# 訓練Actor-Critic
def train_actor_critic(episodes):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_shape[0]])
        done = False
        rewards = []
        states = []
        actions = []
        while not done:
            action_probs = actor.predict(state)
            action = np.random.choice(action_shape, p=action_probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_shape[0]])
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                discounted_rewards = np.zeros_like(rewards)
                cumulative = 0
                for t in reversed(range(len(rewards))):
                    cumulative = cumulative * gamma + rewards[t]
                    discounted_rewards[t] = cumulative
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)
                for state, action, reward in zip(states, actions, discounted_rewards):
                    action_onehot = np.zeros(action_shape)
                    action_onehot[action] = 1
                    value = critic.predict(state)
                    advantage = reward - value
                    actor.fit(state, action_onehot.reshape([1, action_shape]), sample_weight=np.array([advantage]), verbose=0)
                    critic.fit(state, np.array([reward]), verbose=0)
                print(f"Episode: {episode+1}/{episodes}, Score: {sum(rewards)}")

train_actor_critic(episodes)