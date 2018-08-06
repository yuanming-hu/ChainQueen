import tensorflow as tf
import tensorflow.contrib.layers as ly
from functools import partial
import cv2
import os
import numpy as np
import random
import math
import time

from vector_math import *

from states import *#InitialState, UpdatedState

class Simulation:

  def __init__(self, sess):
    self.sess = sess
    self.initial_velocity = tf.placeholder(shape=(2,), dtype=tf.float32)
    self.initial_state = InitialState(
        self, initial_velocity=self.initial_velocity)
    self.updated_states = []

    # Boundary condition
    if sticky:
      self.bc = np.zeros((1, res, res, 2))
      self.bc[:, 4:res - 4, 4:res - 4] = 1
    else:
      self.bc = np.zeros((1, res, res, 2, 2))
      self.bc[:, :, 4:, 1, 0] = 1
      self.bc[:, :, :res - 4, 1, 1] = 1
      self.bc[:, 4:, :, 1, 0] = 1
      self.bc[:, :res - 4, :, 1, 1] = 1

    previous_state = self.initial_state

    for i in range(total_steps):
      new_state = UpdatedState(self, previous_state)
      self.updated_states.append(new_state)
      previous_state = new_state

    self.states = [self.initial_state] + self.updated_states

  def run(self):
    results = [s.get_evaluated() for s in self.states]

    feed_dict = {
        self.initial_state.position: [[[
            random.uniform(0.3, 0.5) * res,
            random.uniform(0.2, 0.4) * res
        ] for i in range(particle_count)]],
        self.initial_velocity: [0, 0],
        self.initial_state.deformation_gradient:
            identity_matrix +
            np.zeros(shape=(batch_size, particle_count, 1, 1))
    }

    results = self.sess.run(results, feed_dict=feed_dict)

    while True:
      for i, r in enumerate(results):
        self.visualize(i, r)

  def optimize(self):
    # os.system('cd outputs && rm *.png')
    # Note: taking the first half only
    t = time.time()

    final_state = self.states[-1].controller_states[0, 0]

    final_position = [final_state[num_groups // 2 * 4], final_state[num_groups // 2 * 4 + 1]]

    goal_input = self.initial_state.goal
    loss = (final_position[0] - res * goal_input[0, 0, 0])**2 + (
        final_position[1] - res * goal_input[0, 0, 1])**2

    current_velocity = np.array([0, 0], dtype=np.float32)
    results = [s.get_evaluated() for s in self.states]

    # Initial particle samples
    particles = [[]]

    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]) * scale  + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]) * scale  + 0.1
          particles[0].append([res * u, res * v])
    assert len(particles[0]) == particle_count

    counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    opt = ly.optimize_loss(
        loss=loss,
        learning_rate=lr,
        optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9),
        variables=trainables,
        global_step=counter)

    self.sess.run(tf.global_variables_initializer())

    i = 0
    while True:
      goal = [0.50 + random.random() * 0.0, 0.4 + random.random() * 0.2]
      feed_dict = {
          self.initial_state.position:
              particles,
          self.initial_velocity:
              current_velocity,
          self.initial_state.deformation_gradient:
              identity_matrix +
              np.zeros(shape=(batch_size, particle_count, 1, 1)),
          goal_input: [[goal]]
      }
      pos, l, _, evaluated = self.sess.run(
          [final_position, loss, opt, results], feed_dict=feed_dict)
      print('  loss', l)
      try:
        for j, r in enumerate(evaluated):
          frame = i * (total_steps + 1) + j
          self.visualize(
              i=frame,
              r=r,
              output_fn='outputs/{:04d}.png'.format(frame),
              goal=goal)
      except Exception as e:
        print(e)
      print('time', time.time() - t)
      i += 1

    os.system('cd outputs && ti video')
    os.system('cp outputs/video.mp4 .')

  def visualize(self, i, r, goal, output_fn=None):
    pos = r['position'][0]
    mass = r['mass'][0]
    grid = r['grid'][0][:, :, 1:2]
    J = determinant(r['deformation_gradient'])[0]
    #5 print(grid.min(), grid.max())
    grid = grid / (1e-5 + np.abs(grid).max()) * 4 + 0.5
    grid = np.clip(grid, 0, 1)
    kernel_sum = np.sum(r['kernels'][0], axis=(1, 2))
    if 0 < i < 3:
      np.testing.assert_array_almost_equal(kernel_sum, 1, decimal=3)
      np.testing.assert_array_almost_equal(
          mass.sum(), particle_count, decimal=3)

    scale = 30

    # Pure-white background
    img = np.ones((scale * res, scale * res, 3), dtype=np.float)

    for i in range(len(pos)):
      p = pos[i]
      x, y = tuple(map(lambda x: math.ceil(x * scale), p))
      #if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
      #  img[x, y] = (0, 0, 1)
      cv2.circle(
          img,
          (y, x),
          radius=1,
          #color=(1, 1, float(J[i])),
          color=(0.2, 0.2, 0.2),
          thickness=-1)

    cv2.line(
        img, (int(res * scale * 0.101), 0),
        (int(res * scale * 0.101), res * scale),
        color=(0, 0, 0))
    cv2.circle(
        img, (int(res * scale * goal[1]), int(res * scale * goal[0])),
        radius=8,
        color=(0.0, 0.9, 0.0),
        thickness=-1)

    try:
      for i in range(len(actuations)):
        act = r['actuation'][i]
        if act < 0:
          color = (255, 0, 0)
        else:
          color = (0, 255, 0)
        x0 = 20 + 25 * i
        x1 = 40 + 25 * i
        y0 = 140
        y1 = int(act * 50 + 140)
        if y0 > y1:
          y0, y1 = y1, y0
        cv2.rectangle(img, (y0, x0), (y1, x1), color, thickness=-1)
    except Exception as e:
      raise e

    try:
      position = [
          r['controller_states'][0, 0][num_groups // 2 * 6], r['controller_states'][0, 0][num_groups // 2 * 6 + 1]
      ]
      cv2.circle(
          img, (int(scale * position[1]), int(scale * position[0])),
          radius=4,
          color=(0.9, 0.0, 0.0),
          thickness=-1)
    except:
      pass

    img = img.swapaxes(0, 1)[::-1, :, ::-1]
    mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    mass = cv2.resize(
        mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    grid = cv2.resize(
        grid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Particles', img)
    #cv2.imshow('Mass', mass / 10)
    #cv2.imshow('Velocity', grid)
    if output_fn is not None:
      cv2.imwrite(output_fn, img * 255)
    cv2.waitKey(1)
