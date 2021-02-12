import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np


class PolicyGradientEngine(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.lr = 0.00000001
        
        # States buffer
        self.states = []
        #  buffer
        self.actions = []
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
        # Append to actions taken buffer
        self.actions.append(action)
        # Record the state
        self.states.append(state)
        # record the reward
        self.rewards.append(reward)

    def action(self, state):
        """ Determine action to take on a given state
        :param state: np.array representing the state of the environment
        :return: (action, actions_prob), where action is a one-hot-encoded
               representation of the action taken, and actions_prob is
               the probabiliyt distribution of actions determined by the
               model"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.model(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        return action.numpy()[0], action_probs.probs.numpy()

    def discounted_reward(self, rewards):
        disc_rewards = np.zeros_like(rewards)
        running_sum = 0
        for idx in reversed(range(0, rewards.size)):
            running_sum = running_sum * self.gamma + rewards[idx]
            disc_rewards[idx] = running_sum
        return disc_rewards

    def train(self):
        # Discounted normalized rewards
        rewards = np.vstack(self.rewards)
        rewards = self.discounted_reward(rewards)
        rewards = rewards - np.mean(rewards)
        rewards /= (np.std(rewards) + 1e-7)

        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        
        X = np.squeeze(np.vstack([self.states]))
        rw = np.array(rewards)

        # print(rw)
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (r, state) in enumerate(zip(rw, X)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.model(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                entropy_loss = keras.losses.categorical_crossentropy(probs, probs)
                # TODO: I am not sure about the sign of these term
                loss += (-r * tf.squeeze(log_prob) + entropy_loss)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        
        self.states = []
        self.action_probs = []
        self.actions = []
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
        model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'))
        model.add(keras.layers.Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
        model.add(keras.layers.Dense(4096, activation='relu', kernel_initializer='glorot_normal'))        
        model.add(keras.layers.Dense(self.action_dim, activation='softmax'))
        opt = keras.optimizers.Adam(lr=self.lr)
        model.compile(optimizer=opt)
        model.summary()
        return model    