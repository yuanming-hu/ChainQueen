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
  def controller(previous_state):
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, :, None] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(mask * previous_state.position, axis=1, keepdims=True)
      vel = tf.reduce_sum(mask * previous_state.velocity, axis=1, keepdims=True)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append(goal)
    controller_inputs = tf.concat(controller_inputs, axis=2)
    intermediate = tf.matmul(W1, controller_inputs[0, 0, :, None])
    actuation = tf.tanh(intermediate[:, 0] + b1) * actuation_strength
    actuation = actuation[0]
    debug = {'controller_inputs': controller_inputs, 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(1, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[i][None, None]
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = 4500 * make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + matmatmul(
          act, transpose(previous_state['deformation_gradient']))
    return total_actuation, debug

  sim = Simulation(
      num_particles=num_particles,
      num_time_steps=30,
      grid_res=(25, 25),
      controller=controller,
      batch_size=batch_size,
      sess=sess)
  print("Building time: {:.4f}s".format(time.time() - t))
  # os.system('cd outputs && rm *.png')

  t = time.time()

  final_state = sim.states[-1]['debug']['controller_inputs'][0, 0]
  final_position = [
      final_state[num_groups // 2 * 6], final_state[num_groups // 2 * 6 + 1]
  ]

  loss = (final_position[0] - sim.grid_res[0] * goal[0, 0, 0])**2 + (
      final_position[1] - sim.grid_res[1] * goal[0, 0, 1])**2

  initial_velocity = np.zeros(shape=[batch_size, num_particles, 2])
  results = [s.get_evaluated() for s in sim.states]

  initial_positions = [[]]

  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[b].append(
              [sim.grid_res[0] * u, sim.grid_res[1] * v])
  assert len(initial_positions[0]) == num_particles

  counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  opt = ly.optimize_loss(
      loss=loss,
      learning_rate=lr,
      optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9),
      variables=trainables,
      global_step=counter)

  sess.run(tf.global_variables_initializer())

  # Optimization loop
  for i in range(1000000):
    goal_input = [0.50 + random.random() * 0.0, 0.6 + random.random() * 0.0]
    feed_dict = {
        sim.initial_state.to_tuple(): sim.get_initial_state(position=np.array(initial_positions)),
        goal: [[goal_input]]
    }
    pos, l, _, evaluated = sess.run(
        [final_position, loss, opt, results], feed_dict=feed_dict)
    print('  loss', l)

    for j, r in enumerate(evaluated):
      frame = i * (sim.num_time_steps + 1) + j
      img = sim.visualize(i=frame, r=r)
      scale = sim.scale
      cv2.circle(
          img, (int(sim.grid_res[0] * scale * goal_input[1]),
                int(sim.grid_res[1] * scale * goal_input[0])),
          radius=8,
          color=(0.0, 0.9, 0.0),
          thickness=-1)
      if j != 0:
        for k in range(len(actuations)):
          act = r['debug']['actuation'][k]
          if act < 0:
            color = (255, 0, 0)
          else:
            color = (0, 255, 0)
          x0 = 20 + 25 * k
          x1 = 40 + 25 * k
          y0 = 140
          y1 = int(act * 50 + 140)
          if y0 > y1:
            y0, y1 = y1, y0
          cv2.rectangle(img, (y0, x0), (y1, x1), color, thickness=-1)
        position = [
            r['debug']['controller_inputs'][0, 0][num_groups // 2 * 6],
            r['debug']['controller_inputs'][0, 0][num_groups // 2 * 6 + 1]
        ]
        cv2.circle(
            img, (int(scale * position[1]), int(scale * position[0])),
            radius=4,
            color=(0.9, 0.0, 0.0),
            thickness=-1)
      img = img.swapaxes(0, 1)[::-1, :, ::-1]

      output_fn = 'outputs/{:04d}.png'.format(frame)
      cv2.imshow('Particles', img)
      cv2.imwrite(output_fn, img * 255)
      cv2.waitKey(1)

    print('time', time.time() - t)

  print("Running time: {:.4f}s".format(time.time() - t))


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
