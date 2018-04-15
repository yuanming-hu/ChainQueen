import tensorflow as tf
import cv2
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 1
particle_count = 100
gravity = (0, -9.8)
dt = 1e-2
total_steps = 10


def polar_decomposition(m):
  assert False

# Initial State
class InitialState:
  def __init__(self):
    self.position = tf.placeholder(tf.float32, [batch_size, particle_count, 2], name='position')
    self.velocity = tf.placeholder(tf.float32, [batch_size, particle_count, 2], name='velocity')
    self.deformation_gradient = tf.placeholder(tf.float32, [batch_size, particle_count, 4], name='dg')

    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''

# Updated state
class UpdatedState:
  def __init__(self, previous_state):
    # Rotational velocity field
    self.velocity = previous_state.position * np.array((-1, 1))[None, None, :]
    # Advection
    self.position = previous_state.position + self.velocity * dt

class Simulation:
  def __init__(self):
    self.initial_state = InitialState()
    self.updated_states = []
    previous_state = self.initial_state

    for i in range(total_steps):
      new_state = UpdatedState(previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state



def main():
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  sess = tf.Session(config=sess_config)

  x = tf.placeholder(tf.float32, [batch_size, 1], name='x')
  y0 = tf.placeholder(tf.float32, [batch_size, 1], name='y0')

  sim = Simulation()



if __name__ == '__main__':
  main()