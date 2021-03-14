import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np


class ActorCriticEngine(object):
    """Actor critic implementation inspied by Keras tutorials

    See: https://keras.io/examples/rl/actor_critic_cartpole/#implement-actor-critic-network
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.0001, lr2=0.001, entropy_c=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Reward discount factor
        self.gamma = gamma
        # Learning rate
        self.lr = lr
        self.lr2 = lr2
        # How much entropy to consider in the action loss
        self.entropy_c = entropy_c

        # States buffer
        self.states = []
        #  buffer
        self.actions = []
        # Rewards buffer
        self.rewards = []
        # model factory function must be defined in subclasses
        self.modela, self.modelc = self.build_model()

    def record(self, state, action, action_probs, reward):
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
        probs = self.modela(state)
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
        returns = np.array(rewards)

        with tf.GradientTape(persistent=True) as tape:
            actor_loss = []
            critic_loss = []
            for idx, (ret, state) in enumerate(zip(returns, X)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.modela(state)
                critic_value = self.modelc(state)
                
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                entropy_loss = keras.losses.categorical_crossentropy(probs, probs)

                actor_loss.append(-(ret - critic_value) * tf.squeeze(log_prob) + self.entropy_c * entropy_loss)
                critic_loss.append(keras.losses.huber(tf.expand_dims(critic_value, 0), tf.expand_dims(ret, 0)))
                loss_value = sum(actor_loss) + sum(critic_loss)
        gradient = tape.gradient(loss_value, self.modela.trainable_variables)
        self.modela.optimizer.apply_gradients(zip(gradient, self.modela.trainable_variables))
        gradient = tape.gradient(loss_value, self.modelc.trainable_variables)
        self.modelc.optimizer.apply_gradients(zip(gradient, self.modelc.trainable_variables))            

        self.states = []
        self.actions = []
        self.rewards = []

    def load(self, filepath):
      self.modela.load_weights("actor_" + filepath)
      self.modelc.load_weights("critit" + filepath)              

    def save(self, filepath):
      self.modela.save_weights("actor_" + filepath)
      self.modelc.save_weights("critit" + filepath)      


class DummyAgent(ActorCriticEngine):
    def __init__(self, state_dim, action_dim, gamma, lr, lr2, entropy_c,
                 layers_actor=[256, 1024, 2096, 4096, 256],
                 layers_critic=[256, 1024, 2096, 4096, 256]):        
        self.layers_actor = layers_actor
        self.layers_critic = layers_critic
        super().__init__(state_dim, action_dim, gamma, lr, lr2, entropy_c)

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.state_dim,)))

        for layer in self.layers_actor:
            model.add(keras.layers.Dense(layer, activation='relu', kernel_initializer='glorot_normal'))
            #model.add(keras.layers.Dropout(rate=0.6))

        model.add(keras.layers.Dense(self.action_dim, activation='softmax'))
        opt = keras.optimizers.Adam(lr=self.lr, clipnorm=0.5)
        model.compile(optimizer=opt)    
        model.summary()

        model_actor = model
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.state_dim,)))

        for layer in self.layers_critic:
            model.add(keras.layers.Dense(layer, activation='relu', kernel_initializer='glorot_normal'))
            #model.add(keras.layers.Dropout(rate=0.6))

        model.add(keras.layers.Dense(1))
        opt = keras.optimizers.Adam(lr=self.lr2, clipnorm=0.5)
        model.compile(optimizer=opt)    
        model.summary()        

        return (model_actor, model)
