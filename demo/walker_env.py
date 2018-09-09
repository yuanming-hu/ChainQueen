import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class WalkerEnv(gym.Env):
  def __init__(self, max_act, max_obs, init_state):
    '''
    max_act = m-d array, where m is number of actuators
    max_obs is n-d array, where n is the state space of the robot.  Assumes 0 is the minimum observation
    init_state is the initial state of the entire robot
    '''
    self.action_space = spaces.Box(-max_act, max_act)
    self.observation_space = spaces.Box(np.array([0.0, 0.0], max_obs)
    
    self.seed()
    
    self.init_state = init_state
    self.state = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self.state = init_state
        #TODO: set robot state here
        return np.array(self.state)
        
    def step(self, action):
    
    
      #TODO: 1. sim forward
      #TODO: 2. update self.sate
      #TODO: 3. calculate reward as velocity toward the goal
      #TODO: 4. return if we're exactly at the goal and give a bonus to reward if we are
      
      
      return np.array(self.state), reward, done, {}
        
    def render(self):
      pass
      #TODO: run visualize here
