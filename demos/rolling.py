import sys
sys.path.append('..')

import random
import time
from simulation import Simulation, get_bounding_box_bc
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np
from IPython import embed

def main(sess):
  batch_size = 1
  gravity = (0, -1)
  # gravity = (0, 0)
  N = 9
  num_particles = N * N
  steps = 1000
  dt = 1e-2
  goal_range = 0.15
  res = (30, 30)
  bc = get_bounding_box_bc(res)

  lr = 1e-2
  
  goal = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  def F_controller(state):
    F = state.position - state.center_of_mass()[:, :, None]
    F = tf.stack([F[:, 1], -F[:, 0]], axis = 1)
    T = tf.cast(state.step_count < 2, dtype = tf.float32)
    return F * 10

  sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      bc=bc,
      gravity=gravity,
      m_p=1,
      V_p=1,
      E = 10,
      nu = 0.3,
      sess=sess,
      use_visualize = True,
      F_controller = F_controller)
  position = np.zeros(shape=(batch_size, num_particles, 2))

  # velocity_ph = tf.constant([0.2, 0.3])
  velocity_ph = tf.constant([0, 0], dtype = tf.float32)
  velocity = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, num_particles], dtype=tf.float32)
  for b in range(batch_size):
    for i in range(N):
      for j in range(N):
        position[b, i * N + j] = ((i * 0.5 + 3) / 30,
                                  (j * 0.5 + 3) / 30)
  position = np.array(position).swapaxes(1, 2)

  sess.run(tf.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=position, velocity=velocity)

  final_position = sim.initial_state.center_of_mass()
  loss = tf.reduce_sum((final_position - goal) ** 2)
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = trainables)

  goal_input = np.array([[0.7, 0.3]], dtype=np.float32)
  

  memo = sim.run(
      initial_state = initial_state, 
      num_steps = steps,
      iteration_feed_dict = {goal: goal_input},
      loss = loss)

  sim.visualize(memo, show = True, interval = 2)
    
if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
