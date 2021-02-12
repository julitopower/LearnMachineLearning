import ctypes
from ctypes import cdll, c_void_p, c_int, c_bool, POINTER
import random
import pgagent
import numpy as np


class Pong():
    """Game simulation for Reinforcement learning"""

    def __init__(self):
        # TODO: Remove hardcoded relative path to shared library
        self.lib = cdll.LoadLibrary('./build/libpong.so')
        # pong_new
        self.lib.pong_new.restype = c_void_p

        # pong_delete
        self.lib.pong_delete.argtypes = [c_void_p]

        # pong_step
        self.lib.pong_step.argtypes = [c_void_p, c_int]

        # pong_state
        self.lib.pong_state.argtypes = [c_void_p]
        self.lib.pong_state.restype = POINTER(ctypes.c_int * 8)

        # pong_reset
        self.lib.pong_reset.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_reward.restype = c_int
        self.lib.pong_reward.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_done.restype = c_bool
        self.lib.pong_done.argtypes = [c_void_p]

        # Initialize game
        self.game = self.lib.pong_new()
        self.lib.pong_reset(self.game)

    def step(self, action):
        self.lib.pong_step(self.game, action)

    def __del__(self):
        self.lib.pong_delete(self.game)

    def reset(self):
        self.lib.pong_reset(self.game)

    def state(self):
        st = list(self.lib.pong_state(self.game).contents)
        return np.array(st, dtype='double')

    def reward(self):
        r = self.lib.pong_reward(self.game)

        st = self.state()
        if r > 0:
            return 1
        elif r < 0:
            return -1
        return -abs(st[5] - st[1]) / ((1 + st[0]) * 1000)

    def done(self):
        return self.lib.pong_done(self.game)


def evaluate(iterations=10000):
    game = Pong()
    reward = 0
    for _ in range(0, iterations):
        game.reset()
        while game.reward() == 0:
            game.step(random.randint(0, 0))
        if game.reward() == 1:
            reward += 1
    return 100 * reward / iterations

if __name__ == "__main__":
    #print(evaluate())

    agent = pgagent.DummyAgent(8, 3)
    game = Pong()
    for i in range(0, 10000):
        print('Executing iteration')
        game.reset()
        idx = 0
        while not game.done():
            state = game.state()
            action, action_probs = agent.action(state)
            game.step(action)
            agent.record(state, action, action_probs, game.reward())
            if idx % 50 == 0:
                print(state, action, action_probs, game.reward())
            idx += 1
        if i % 5 == 0:
            agent.save("model.h5")
        print(game.state(), "iter: ", i, ", Reward: ", game.reward())
        agent.train()
