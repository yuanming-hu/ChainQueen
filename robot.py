from functools import partial
import cv2
import random
import os
from simulation import Simulation
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.layers as ly
from vector_math import *
import IPython

lr = 1e-3


class Robot:

  def __init__(self, frameskip, sess):

    self.sample_density = 20
    self.group_num_particles = self.sample_density**2
    # Robot A
    self.num_groups = 5
    self.group_offsets = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
    self.group_sizes = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    self.actuations = [0, 4]
    self.actuation_strength = 0.4
    self.num_particles = self.group_num_particles * self.num_groups
    
    t = time.time()

    self.batch_size = 1
    goal = tf.placeholder(tf.float32, [self.batch_size, 1, 2], name='goal')
    
    
    # NN weights
    self.W1 = tf.Variable(
        0.02 * tf.random_normal(shape=(len(self.actuations), 6 * len(self.group_sizes))),
        trainable=True)
    self.b1 = tf.Variable([[0.1] * len(self.actuations)], trainable=True)

    def controller(previous_state):
      controller_inputs = []
      for i in range(self.num_groups):
        mask = self.particle_mask(i * self.group_num_particles,
                             (i + 1) * self.group_num_particles)[:, :, None] * (
                                 1.0 / self.group_num_particles)
        pos = tf.reduce_sum(mask * previous_state.position, axis=1, keepdims=True)
        vel = tf.reduce_sum(mask * previous_state.velocity, axis=1, keepdims=True)
        controller_inputs.append(pos)
        controller_inputs.append(vel)
        controller_inputs.append(goal)
      controller_inputs = tf.concat(controller_inputs, axis=2)
      intermediate = tf.matmul(self.W1, controller_inputs[0, 0, :, None])
      actuation = tf.tanh(intermediate[:, 0] + self.b1) * self.actuation_strength
      actuation = actuation[0]
      debug = {'controller_inputs': controller_inputs, 'actuation': actuation}
      total_actuation = 0
      zeros = tf.zeros(shape=(1, self.num_particles))
      for i, group in enumerate(self.actuations):
        act = actuation[i][None, None]
        mask = self.particle_mask_from_group(group)
        act = act * mask
        # First PK stress here
        act = 4500 * make_matrix2d(zeros, zeros, zeros, act)
        # Convert to Kirchhoff stress
        total_actuation = total_actuation + matmatmul(
            act, transpose(previous_state['deformation_gradient']))
      return total_actuation, debug

    self.sim = Simulation(
        num_particles=self.num_particles,
        num_time_steps=frameskip,
        grid_res=(25, 25),
        controller=controller,
        batch_size=self.batch_size,
        sess=sess)
    print("Building time: {:.4f}s".format(time.time() - t))
    # os.system('cd outputs && rm *.png')

    t = time.time()

    self.final_state = self.sim.states[-1]['debug']['controller_inputs'][0, 0]
    self.final_position = [
        self.final_state[self.num_groups // 2 * 6], self.final_state[self.num_groups // 2 * 6 + 1]
    ]

    self.loss = (self.final_position[0] - self.sim.grid_res[0] * goal[0, 0, 0])**2 + (
        self.final_position[1] - self.sim.grid_res[1] * goal[0, 0, 1])**2

    self.initial_velocity = np.zeros(shape=[self.batch_size, self.num_particles, 2])
    self.results = [s.get_evaluated() for s in self.sim.states]

    self.initial_positions = [[]]

    for b in range(self.batch_size):
      for i, offset in enumerate(self.group_offsets):
        for x in range(self.sample_density):
          for y in range(self.sample_density):
            scale = 0.2
            u = ((x + 0.5) / self.sample_density * self.group_sizes[i][0] + offset[0]
                ) * scale + 0.2
            v = ((y + 0.5) / self.sample_density * self.group_sizes[i][1] + offset[1]
                ) * scale + 0.1
            self.initial_positions[b].append(
                [self.sim.grid_res[0] * u, self.sim.grid_res[1] * v])
    assert len(self.initial_positions[0]) == self.num_particles

    '''
    counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    opt = ly.optimize_self.loss(
        self.loss=self.loss,
        learning_rate=lr,
        optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9),
        variables=trainables,
        global_step=counter)
    '''

  def get_sim(self):
    return self.sim
    
  def init_pos(self):
    return np.concatenate(self.initial_positions).flatten()
  
  def init_vel(self):
    return np.concatenate(self.initial_velocity).flatten()
    
  def pos(self):
    return sim.states[-1].position.eval()
  
  def vel(self):
    return sim.states[-1].velocity.eval()

  def particle_mask(self, start, end):
    r = tf.range(0, self.num_particles)
    return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


  def particle_mask_from_group(self, g):
    return self.particle_mask(g * self.group_num_particles, (g + 1) * self.group_num_particles)




 
if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    robot = Robot(30, sess)  
  
