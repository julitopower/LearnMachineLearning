from ctypes import cdll, c_void_p, c_int, c_double, c_bool, POINTER
import random
import actorcriticagent2
import numpy as np
import argparse as ap

class Observation():
    def __init__(self):
        self.shape = [8]


class Action():
    def __init__(self):
        self.n = 3

class Pong():
    """Game simulation for Reinforcement learning"""

    def __init__(self, world_w=2000, world_h=1000, proj_w=255, proj_h=255):
        # TODO: Remove hardcoded relative path to shared library
        self.lib = cdll.LoadLibrary('./build/pong.so')

        # pong_new
        self.lib.pong_new.argtypes = [c_int, c_int, c_int, c_int]
        self.lib.pong_new.restype = c_void_p

        # pong_delete
        self.lib.pong_delete.argtypes = [c_void_p]

        # pong_render
        self.lib.pong_render.argtypes = [c_void_p]

        # pong_step
        self.lib.pong_step.argtypes = [c_void_p, c_int]

        # pong_state
        self.lib.pong_state.argtypes = [c_void_p]
        self.lib.pong_state.restype = POINTER(c_double * 8)

        # pong_reset
        self.lib.pong_reset.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_reward.restype = c_double
        self.lib.pong_reward.argtypes = [c_void_p]

        # pong_reward
        self.lib.pong_done.restype = c_bool
        self.lib.pong_done.argtypes = [c_void_p]

        # Initialize game
        self.game = self.lib.pong_new(world_w, world_h, proj_w, proj_h)
        self.lib.pong_reset(self.game)

        self.observation_space = Observation()
        self.action_space = Action()

        self.steps = 0;

    def step(self, action):
        if action != 2:
            self.steps += 1
        self.lib.pong_step(self.game, action)
        return (self.state(), self.reward(), self.done(), None)

    def __del__(self):
        self.lib.pong_delete(self.game)

    def reset(self):
        self.steps = 0
        self.lib.pong_reset(self.game)
        state = self.state()
        # State[bx, by, bvx, bvy, rx, ry, vrx, vry]
        # this is racket start y coordinate
        self.starty = state[5]
        return state

    def state(self):
        st = list(self.lib.pong_state(self.game).contents)
        return np.array(st, dtype='double')

    def reward(self):
        r = self.lib.pong_reward(self.game)

        # Measures ideal number of movements vr actual movements
        # We want to reward quickly going to the right final
        # location for the racket
        bally = self.state()[1]
        steps_penalty = (abs(self.starty - bally) -  self.steps)
        if r > 0:
            return 100 + steps_penalty
        elif r < 0:
            return r + steps_penalty
        else:
            return 0

    def done(self):
        return self.lib.pong_done(self.game)

    def render(self):
        self.lib.pong_render(self.game)


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

    parser = ap.ArgumentParser(description="Pong game RL")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr", help="Learning rate", type=float, required=True)
    parser.add_argument("--er", help="Entropy rate", type=float, required=True)
    parser.add_argument("-v", "--verbose", help="Visualize pong game", action="store_true", required=True)
    args = parser.parse_args()

    print(f"#######{args.gamma}")

    # print(evaluate())
    # Setup numpy print options
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    gamma = args.gamma
    lr = args.lr
    entropy_c = args.er
    render = args.v
    score_history = []

    agent = actorcriticagent2.DummyAgent(8, 3, gamma, lr, entropy_c,
                                        #layers=[32, 10240, 256, 20000, 2560])
                                        #layers=[32, 64, 1024, 2560, 20000, 2560, 1024, 1024, 512])
                                        layers_actor=[64, 256, 512, 1024],
                                        layers_critic=[64, 256, 64])
    # agent.load("model__test.h5")
    # agent.load('model___new_test.h5')
    # agent.load('model_good_march_3rd.h5')
    try:
        agent.load('ac_03072021.h5')
        print("######### Model loaded succesfully")
    except:
        pass
    game = Pong(800, 600, int(400/2), int(200/2))
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
            if render:
                game.render()
        if i % 10 == 0:
            agent.save("ac_03072021.h5")
        score_history.append(game.reward())
        print(f"iter0 {i}, reward: {score_history[-1]:.2f}", " -- Avg over 100 episodes", np.mean(score_history[-100:]))
        agent.train()
