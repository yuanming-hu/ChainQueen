import sys
sys.path.append('..')

import random
import os
import numpy as np
from simulation import Simulation, get_bounding_box_bc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.layers as ly
from vector_math import *
import export 
import IPython

lr = 0.1
gamma = 0.0

sample_density = 15
group_num_particles = sample_density**3
goal_pos = np.array([1.4, 0.4, 0.5])
goal_range = np.array([0.0, 0.0, 0.0])
batch_size = 1
actuation_strength = 1.3

config = 'B'

exp = export.Export('walker3d')

# Robot B
num_groups = 7
group_offsets = [(0, 0, 0), (0.5, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0), (2.5, 0, 0)]
group_sizes = [(0.5, 1, 2), (0.5, 1, 2), (1, 1, 2), (1, 1, 2), (1, 1, 2), (0.5, 1, 2), (0.5, 1, 2)]
actuations = [0, 1, 5, 6]
fixed_groups = []
head = 3
gravity = (0, -2, 0)

num_particles = group_num_particles * num_groups


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# NN weights
W1 = tf.Variable(
    0.02 * tf.random_normal(shape=(len(actuations), 9 * len(group_sizes))),
    trainable=True)
b1 = tf.Variable([0.0] * len(actuations), trainable=True)


def main(sess):
  t = time.time()

  goal = tf.placeholder(dtype=tf.float32, shape=[batch_size, 3], name='goal')

  # Define your controller here
  def controller(state):
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, None, :] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(mask * state.position, axis=2, keepdims=False)
      vel = tf.reduce_sum(mask * state.velocity, axis=2, keepdims=False)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append((goal - goal_pos) / np.maximum(goal_range, 1e-5))
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, 9 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, 9 * num_groups, 1)
    # Batch, 6 * num_groups, 1
    intermediate = tf.matmul(W1[None, :, :] +
                             tf.zeros(shape=[batch_size, 1, 1]), controller_inputs)
    # Batch, #actuations, 1
    assert intermediate.shape == (batch_size, len(actuations), 1)
    assert intermediate.shape[2] == 1
    intermediate = intermediate[:, :, 0]
    # Batch, #actuations
    actuation = tf.tanh(intermediate + b1[None, :]) * actuation_strength
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      act = make_matrix3d(zeros, zeros, zeros, zeros, act, zeros, zeros, zeros, zeros)
      total_actuation = total_actuation + act
    return total_actuation, debug
  
  res = (60, 30, 30)
  bc = get_bounding_box_bc(res)
  
  sim = Simulation(
      dt=0.005,
      num_particles=num_particles,
      grid_res=res,
      dx=1.0 / res[1],
      gravity=gravity,
      controller=controller,
      batch_size=batch_size,
      bc=bc,
      sess=sess,
      scale=20)
  print("Building time: {:.4f}s".format(time.time() - t))

  final_state = sim.initial_state['debug']['controller_inputs']
  s = head * 9
  
  final_position = final_state[:, s:s+3]
  final_velocity = final_state[:, s + 3: s + 6]
  loss1 = tf.reduce_mean(tf.reduce_sum((final_position - goal) ** 2, axis = 1))
  loss2 = tf.reduce_mean(tf.reduce_sum(final_velocity ** 2, axis = 1)) 

  loss = loss1 + gamma * loss2

  initial_positions = [[] for _ in range(batch_size)]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          for z in range(sample_density):
            scale = 0.2
            u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
                ) * scale + 0.2
            v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
                ) * scale + 0.1
            w = ((z + 0.5) / sample_density * group_sizes[i][2] + offset[2]
                 ) * scale + 0.1
            initial_positions[b].append([u, v, w])
  assert len(initial_positions[0]) == num_particles
  initial_positions = np.array(initial_positions).swapaxes(1, 2)

  sess.run(tf.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=10)

  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state=initial_state)
  
  sym = sim.gradients_sym(loss, variables=trainables)

  gx, gy, gz = goal_range
  pos_x, pos_y, pos_z = goal_pos
  goal_train = [np.array(
    [[pos_x + (random.random() - 0.5) * gx,
      pos_y + (random.random() - 0.5) * gy,
      pos_z + (random.random() - 0.5) * gz
      ] for _ in range(batch_size)],
    dtype=np.float32) for __ in range(1)]

  vis_id = list(range(batch_size))
  random.shuffle(vis_id)

  # Optimization loop
  for i in range(100000):
    t = time.time()
    print('Epoch {:5d}, learning rate {}'.format(i, lr))

    loss_cal = 0.
    print('train...')
    for it, goal_input in enumerate(goal_train):
      tt = time.time()
      memo = sim.run(
          initial_state=initial_state,
          num_steps=800,
          iteration_feed_dict={goal: goal_input},
          loss=loss)
      print('forward', time.time() - tt)
      tt = time.time()
      grad = sim.eval_gradients(sym=sym, memo=memo)
      print('backward', time.time() - tt)

      for i, g in enumerate(grad):
        print(i, np.mean(np.abs(g)))
      grad = [np.clip(g, -1, 1) for g in grad]


      gradient_descent = [
          v.assign(v - lr * g) for v, g in zip(trainables, grad)
      ]
      sess.run(gradient_descent)
      print('Iter {:5d} time {:.3f} loss {}'.format(
          it, time.time() - t, memo.loss))
      loss_cal = loss_cal + memo.loss
      sim.visualize(memo, batch=random.randrange(batch_size), export=exp,
                    show=True, interval=16)
    #exp.export()
    print('train loss {}'.format(loss_cal / len(goal_train)))
    
if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
