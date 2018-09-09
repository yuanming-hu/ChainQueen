import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import walker_sim as w_s

goal_ball = 0.001
class WalkerEnv(gym.Env):
  def __init__(self, max_act, max_obs, sess, goal_input):
    '''
    max_act = m-d array, where m is number of actuators
    max_obs is n-d array, where n is the state space of the robot.  Assumes 0 is the minimum observation
    init_state is the initial state of the entire robot
    '''
    self.action_space = spaces.Box(-10.0, 10.0)
    self.observation_space = spaces.Box(np.array([0.0, 0.0], max_obs)
    
    self.seed()
    
    
    self.state = None
    self.goal_input = goal_input
    
    self.init_stte, self.sim, self.loss, self.x, self.y = w_s.generate_sim(sess)
    sim.set_initial_state(initial_state=init_state)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self.state = init_state
        sime.set_initial_state(initial_state = self.init_state)
        return np.array(self.state)
        
    def step(self, action):
    
    
      #1. sim forward
      memo = sim.run(
          initial_state=self.state,
          num_steps=1,
          iteration_feed_dict={goal: self.goal_input, actuation: action},
          loss=self.loss)
          
      memo_x = sim.run(
          initial_state=self.state,
          num_steps=1,
          iteration_feed_dict={goal: self.goal_input},
          loss=self.x)
      
      memo_y = sim.run(
          initial_state=self.state,
          num_steps=1,
          iteration_feed_dict={goal: self.goal_input},
          loss=self.y)
          
          
      #2. update self.state
      
      
      self.state = np.array([memo_x.loss, memo_y.loss])     
      
      
      #3. calculate reward as velocity toward the goal
      reward = memo.loss
      
      #TODO: 4. return if we're exactly at the goal and give a bonus to reward if we are
      done = np.linalg.norm(self.state - self.goal_input) < goal_ball
      
      if done:
        reward += 1
      
      self.memo = memo
      return np.array(self.state), reward, done, {}
        
    def render(self):
      sim.visualize(self.memo, 1,show=True, export=exp = w_s.exp, interval=1)
