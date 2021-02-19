from ctypes import cdll, c_void_p, c_int, c_double, c_bool, POINTER
import random
import pgagent
import numpy as np
import sys


class Pong():
    """Game simulation for Reinforcement learning"""

    def __init__(self, world_w=2000, world_h=1000, proj_w=255, proj_h=255):
        # TODO: Remove hardcoded relative path to shared library
        self.lib = cdll.LoadLibrary('./build/libpong.so')

        # pong_new
        self.lib.pong_new.argtypes = [c_int, c_int, c_int, c_int]
        self.lib.pong_new.restype = c_void_p

        # pong_delete
        self.lib.pong_delete.argtypes = [c_void_p]

        # pong_step
        self.lib.pong_step.argtypes = [c_void_p, c_int]

        # pong_state
        self.lib.pong_state.argtypes = [c_void_p]
        self.lib.pong_state.restype = POINTER(c_double * 8)

        # pong_reset
        self.lib.pong_reset.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_reward.restype = c_int
        self.lib.pong_reward.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_done.restype = c_bool
        self.lib.pong_done.argtypes = [c_void_p]

        # Initialize game
        self.game = self.lib.pong_new(world_w, world_h, proj_w, proj_h)
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
            return 100
        elif r < 0:
            return -abs(st[5] - st[1])
        else:
            return -abs(st[5] - st[1]) / ( 1 + st[0])


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
    # print(evaluate())
    # Setup numpy print options
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    gamma = float(sys.argv[1])
    lr = float(sys.argv[2])
    entropy_c = float(sys.argv[3])
    score_history = []

    agent = pgagent.DummyAgent(8, 3, gamma, lr, entropy_c,
                               #layers=[32, 10240, 256, 20000, 2560])
                               #layers=[32, 64, 1024, 2560, 20000, 2560, 1024, 1024, 512])
                               layers=[64, 256, 512, 1024])
    game = Pong(400, 200, 400, 200)
    for i in range(0, 10000):
        game.reset()
        idx = 0
        while not game.done():
            state = game.state()
            action, action_probs = agent.action(state)
            game.step(action)
            agent.record(state, action, action_probs, game.reward())
            if idx % 50 == 0:
                pass
                # print(state, action, action_probs, game.reward())
            idx += 1
        if i % 5 == 0:
            agent.save("model.h5")
        score_history.append(game.reward())
        print(f"iter0 {i}, reward: {score_history[-1]:.2f}", " -- Avg over 100 episodes", np.mean(score_history[-100:]))
        agent.train()
