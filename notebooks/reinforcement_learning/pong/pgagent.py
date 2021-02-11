import tensorflow.keras as keras
import numpy as np


class PolicyGradientEngine(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.lr = 0.00001
        # States buffer
        self.states = []
        # Gradients buffer
        self.gradients = []
        # Rewards buffer
        self.rewards = []
        # Action probabilities returned by the policy
        self.action_probs = []
        # model factory function must be defined in subclasses
        self.model = self.build_model()

    def record(self, state, action, action_probs, reward):
        # one hot encode the action taken
        y = np.zeros([self.action_dim])
        y[action] = 1
        # Difference between action taken and action predicted by the model
        self.gradients.append(y.astype('float32') - action_probs)
        # Record the state
        self.states.append(state)
        # record the reward
        if reward < 0:
          reward = 0
        self.rewards.append(reward)

    def action(self, state):
        """ Determine action to take on a given state
        :param state: np.array representing the state of the environment
        :return: (action, actions_prob), where action is a one-hot-encoded
               representation of the action taken, and actions_prob is
               the probabiliyt distribution of actions determined by the
               model"""
        state = state.reshape([1, state.shape[0]])
        action_probs = self.model.predict(state, batch_size=1).flatten()
        self.action_probs.append(action_probs)
        action = np.random.choice(self.action_dim, p=action_probs)
        return action, action_probs

    def discounted_reward(self, rewards):
        disc_rewards = np.zeros_like(rewards)
        running_sum = 0
        for idx in reversed(range(0, rewards.size)):
          running_sum = running_sum * self.gamma + rewards[idx]
          disc_rewards[idx] = running_sum
        return disc_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discounted_reward(rewards)
        rewards = (rewards - np.mean(rewards) / (np.std(rewards) + 1e-7))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.action_probs + (self.lr * np.squeeze(np.vstack([gradients])))
        self.model.train_on_batch(X, Y)
        self.states = []
        self.action_probs = []
        self.gradients = []
        self.rewards = []

    def load(self, filepath):
      self.model.load_weights(filepath)

    def save(self, filepath):
      self.model.save_weights(filepath)
        

class DummyAgent(PolicyGradientEngine):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.state_dim,)))
        model.add(keras.layers.Dense(9, activation='relu'))
        model.add(keras.layers.Dense(10240, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))                
        model.add(keras.layers.Dense(self.action_dim, activation='softmax'))
        optimizer = keras.optimizers.SGD()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.summary()
        return model

