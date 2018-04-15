import tensorflow as tf
import cv2
import os
import numpy as np
import random
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
TODO:
dx
'''

batch_size = 1
particle_count = 100
gravity = (0, -9.8)
dt = 1e-1
total_steps = 10
res = 20


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
    self.velocity = (previous_state.position - res / 2) * np.array((1, -1))[None, None, ::-1]
    # Advection
    self.position = previous_state.position + self.velocity * dt

class Simulation:
  def __init__(self, sess):
    self.sess = sess
    self.initial_state = InitialState()
    self.updated_states = []
    previous_state = self.initial_state

    for i in range(total_steps):
      new_state = UpdatedState(previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state

  def run(self):
    positions = [self.initial_state.position]
    for i in range(total_steps):
      positions.append(self.updated_states[i].position)

    feed_dict = {
      self.initial_state.position: [[[random.random() * res / 2, random.random() * res / 2] for i in range(particle_count)]],
      self.initial_state.velocity: [[[0, 0] for i in range(particle_count)]]
    }

    eval_positions = self.sess.run(positions, feed_dict=feed_dict)

    for pos in eval_positions:
      # Visualize the first trajectory on the whole batch only
      print("Visualizing...", pos[0])

      self.visualize(pos[0])


  def visualize(self, pos):
    scale = 10

    # Pure-white background
    img = np.ones((scale * res, scale * res, 3), dtype=np.float)
    for p in pos:
      x, y = tuple(map(lambda x: math.ceil(x * scale), p))
      if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
        img[x, y] = (0, 0, 1)


    img = img.swapaxes(0, 1)[:, :, ::-1]
    cv2.imshow('img', img)
    cv2.waitKey(0)


def main(sess):
  sim = Simulation(sess)
  sim.run()


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)