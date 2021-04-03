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
        optimal_steps = abs(self.starty - bally)
        if self.steps > optimal_steps:
            steps_penalty = float(self.steps) / (optimal_steps + 1)
        else:
            steps_penalty = 1.0;
            
        if r > 0:
            return 100.0 / steps_penalty
        elif r < 0:
            return r - steps_penalty
        else:
            return 0

    def done(self):
        return self.lib.pong_done(self.game)

    def render(self):
        self.lib.pong_render(self.game)

