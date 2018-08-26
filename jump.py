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

lr = 1e-3
sample_density = 20
group_num_particles = sample_density**2
import IPython

if True:
  # Robot A
  num_groups = 5
  group_offsets = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
  group_sizes = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
  actuations = [0, 4]
else:
  # Robot B
  num_groups = 7
  group_offsets = [(0, 0), (0.5, 0), (0, 1), (1, 1), (2, 1), (2, 0), (2.5, 0)]
  group_sizes = [(0.5, 1), (0.5, 1), (1, 1), (1, 1), (1, 1), (0.5, 1), (0.5, 1)]
  actuations = [0, 1, 5, 6]

actuation_strength = 0.4
num_particles = group_num_particles * num_groups


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# NN weights
W1 = tf.Variable(
    0.02 * tf.random_normal(shape=(len(actuations), 6 * len(group_sizes))),
    trainable=True)
b1 = tf.Variable([[0.1] * len(actuations)], trainable=True)


def main(sess):
  t = time.time()

  batch_size = 1
  goal = tf.placeholder(tf.float32, [batch_size, 1, 2], name='goal')

  # Define your controller here
  def controller(state):
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, :, None] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(mask * state.position, axis=1, keepdims=True)
      vel = tf.reduce_sum(mask * state.velocity, axis=1, keepdims=True)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append(goal)
    controller_inputs = tf.concat(controller_inputs, axis=2)
    intermediate = tf.matmul(W1, controller_inputs[0, 0, :, None])
    actuation = tf.tanh(intermediate[:, 0] + b1) * actuation_strength
    assert batch_size
    # TODO: here seems to work only for batch_size = 1
    actuation = actuation[0]
    debug = {'controller_inputs': controller_inputs, 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(1, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[i][None, None]
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = 40 * make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + matmatmul(
          act, transpose(state['deformation_gradient']))
    return total_actuation, debug

  sim = Simulation(
      dt=0.005,
      num_particles=num_particles,
      grid_res=(25, 25),
      controller=controller,
      batch_size=batch_size,
      sess=sess)
  print("Building time: {:.4f}s".format(time.time() - t))

  t = time.time()

  final_state = sim.initial_state['debug']['controller_inputs']
  s = num_groups // 2 * 6
  final_position = final_state[:, :, s:s+2]
  loss = tf.reduce_sum((final_position - goal) ** 2)

  initial_positions = [[]]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]) * scale + 0.1
          initial_positions[b].append([u, v])
  assert len(initial_positions[0]) == num_particles

  sess.run(tf.global_variables_initializer())
  
  initial_state = sim.get_initial_state(position=np.array(initial_positions), youngs_modulus=40)

  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  
  # Optimization loop
  while True:
    goal_input = np.array([[[0.50 + random.random() * 0.0, 0.6 + random.random() * 0.0]]], dtype=np.float32)
    memo = sim.run(initial_state=initial_state, num_steps=60,
                   iteration_feed_dict={goal: goal_input}, loss=loss)
    print('loss', memo.loss)
    dots = [(final_position[:, 0], (1, 0, 0), 3), (goal[:, 0], (0, 1, 0), 3)]
    grads = sim.gradients(loss, memo, variables=trainables)
    alpha = 1
    gradient_descent = [v.assign(v - alpha * g) for v, g in zip(trainables, grads)]
    sess.run(gradient_descent)
    sim.visualize(memo, dots=dots)
    print('time', time.time() - t)


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
