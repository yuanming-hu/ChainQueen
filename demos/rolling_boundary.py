import sys
sys.path.append('..')

import random
import time
from simulation import Simulation, get_new_bc
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np
from IPython import embed
import export


difficulty = 0.6

tf.set_random_seed(1234)
np.random.seed(0)
# NN
W1 = tf.Variable(
    0.2 * tf.random_normal(shape=(8, 16)),
    trainable=True)
b1 = tf.Variable(np.random.random(16) * 0.2, trainable=True, dtype = tf.float32)

W2 = tf.Variable(
    0.2 * tf.random_normal(shape=(16, 1)),
    trainable=True)
b2 = tf.Variable(np.random.random(1) * 0.2, trainable=True, dtype = tf.float32)

'''
b = tf.Variable(np.random.random(2000) - 0.5, trainable = True, dtype = tf.float32)
thb = tf.tanh(b)
'''

def main(sess):
  batch_size = 1
  gravity = (0, -1.)
  # gravity = (0, 0)
  N = 10
  dR = 0.1
  R = (N - 1) * dR
  dC = 1.6
  num_particles = int(((N - 1) * dC + 1) ** 2)
  steps = 1000
  dt = 1e-2
  goal_range = 0.15
  res = (90, 60)
  dx = 1. / res[0]
  max_speed = 0.3
  exp = export.Export('rolling_acc_{}'.format(max_speed))

  lef = 2.36
  rig = 9.
  
  temp_f = lambda x: 2 * x - 1 if x > 0.5 else -2 * x + 1
  temp_f_ = lambda x: 2 if x > 0.5 else -2
  boundary = lambda x: ((temp_f(x / res[0]) - 0.5) * difficulty + 0.5) * res[1]
  boundary_ = lambda x: temp_f_(x / res[0]) / res[0] * res[1] * difficulty

  boundary = lambda x: (np.sin((rig - lef) * x / res[0] + lef) * difficulty + 1) * res[1] / 2
  boundary_ = lambda x: (rig - lef) / 2 * difficulty * res[1] * np.cos((rig - lef) * x / res[0] + lef) / res[0]
  tf_boundary = lambda x: (tf.sin((rig - lef) * x + lef) * difficulty + 1) * res[1] / 2
  tf_boundary_ = lambda x: (rig - lef) / 2 * difficulty * res[1] * tf.cos((rig - lef) * x + lef) / res[0]
  
  bc = get_new_bc(res, boundary = boundary, boundary_ = boundary_)

  lr = 1
  
  goal = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  def state_tensor(state):
    cm = state.center_of_mass()
    mv = tf.reduce_mean(state.velocity, axis = 2)
    time1 = tf.sin(tf.cast(state.step_count, dtype = tf.float32) / 100)
    time2 = tf.sin(tf.cast(state.step_count, dtype = tf.float32) / 31)
    time3 = tf.sin(tf.cast(state.step_count, dtype = tf.float32) / 57)
    time4 = tf.sin(tf.cast(state.step_count, dtype = tf.float32) / 7)
    lis = [mv, cm, time1, time2, time3, time4]
    tensor = tf.concat([tf.reshape(t, shape = [batch_size, -1])
                        for t in lis],
                       axis = 1)
    return tensor

  def F_controller(state):
    # F = state.position - state.center_of_mass()[:, :, None]
    # F = tf.stack([F[:, 1], -F[:, 0]], axis = 1)
    # F = tf.constant([1, 2 / res[0] * res[1]])[None, :, None]

    # accelerate
    cm = state.center_of_mass()
    k = tf_boundary_(cm[:, 0])
    L = (k ** 2 + 1) ** 0.5
    dx = tf.reciprocal(L)
    dy = k / L
    F = tf.stack([dx, dy], axis = 1) * max_speed

    # inputs
    inputs = state_tensor(state)
    
    # network
    outputs1 = tf.tanh(tf.matmul(inputs, W1) + b1[None, :])
    outputs2 = tf.tanh(tf.matmul(outputs1, W2) + b2[None, :])
    direction = outputs2
    '''
    direction = thb[state.step_count]
    '''

    '''
    # best solution
    vx = tf.reduce_mean(state.velocity, axis = 2)[:, 0]
    direction = tf.cast(vx > 0, dtype = tf.float32) * 2 - 1
    '''
    

    return (F * direction)[:, :, None]

  sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      bc=bc,
      gravity=gravity,
      m_p=0.5,
      V_p=0.5,
      E = 1,
      nu = 0.3,
      dx = dx,
      sess=sess,
      use_visualize = True,
      F_controller = F_controller,
      part_size = 10)
  position = np.zeros(shape=(batch_size, num_particles, 2))

  # velocity_ph = tf.constant([0.2, 0.3])
  velocity_ph = tf.constant([0, 0], dtype = tf.float32)
  velocity = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, num_particles], dtype=tf.float32)
  random.seed(123)
  for b in range(batch_size):
    dx, dy = 15.81, 7.2
    cnt = 0
    las = 0
    for i in range(N):
      l = int((dC * i + 1) ** 2)
      l, las = l - las, l
      print(l)
      dth = 2 * np.pi / l
      dr = R / (N - 1) * i
      theta = np.pi * 2 * np.random.random()
      for j in range(l):
        theta += dth
        x, y = np.cos(theta) * dr, np.sin(theta) * dr
        position[b, cnt] = ((dx + x) / 45, (dy + y) / 45)
        cnt += 1

  position = np.array(position).swapaxes(1, 2)


  initial_state = sim.get_initial_state(
      position=position, velocity=velocity)
  state_sum = sim.stepwise_sym(state_tensor)

  final_position = sim.initial_state.center_of_mass()
  final_velocity = tf.reduce_mean(sim.initial_state.velocity, axis = 2)
  final_F = tf.reduce_mean(sim.updated_state.F, axis = 2)
  loss = tf.reduce_sum((final_position - goal) ** 2)
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)
  sim.add_vector_visualization(pos=final_position, vector=final_velocity, color=(0, 0, 1), scale=50)
  sim.add_vector_visualization(pos=final_position, vector=final_F, color=(1, 0, 1), scale=500)

  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = trainables)

  goal_input = np.array([[37.39 / 45, 25. / 45]], dtype=np.float32)
  grad_ph = [
      tf.placeholder(shape = v.shape, dtype = tf.float32) for v in trainables
  ]
  velocity = [
      tf.Variable(np.zeros(shape = v.shape, dtype = np.float32),
      trainable=True) for v in trainables
  ]
  
  gradient_descent = [
      v.assign(v - lr * g) for v, g in zip(trainables, grad_ph)
  ]
  
  
  momentum_beta = 0.9
  momentum = [
      v.assign(v * momentum_beta + g * (1 - momentum_beta))
      for v, g in zip(velocity, grad_ph)
  ] + [
      v.assign(v - lr * g) for v, g in zip(trainables, velocity)
  ] 
  
  sess.run(tf.global_variables_initializer())

  flog = open('rolling_{}.log'.format(max_speed), 'w')
  for i in range(100000):
    t = time.time()
    print('Epoch {:5d}, learning rate {}'.format(i, lr))
  
    tt = time.time()
    memo = sim.run(
        initial_state = initial_state, 
        num_steps = steps,
        iteration_feed_dict = {goal: goal_input},
        loss = loss,
        stepwise_loss = state_sum)
    print(memo.stepwise_loss)
    break
    print('forward', time.time() - tt)

    if False:# i % 10 == 0:
      tt = time.time()
      sim.visualize(memo, show = False, interval = 1, export = exp)
      print('visualize', time.time() - tt)
      if memo.loss < 1e-3:
        break

    tt = time.time()
    grad = sim.eval_gradients(sym=sym, memo=memo)
    print('eval_gradients', time.time() - tt)

    tt = time.time()
    grad_feed_dict = {}
    tot = 0.
    for gp, g in zip(grad_ph, grad):
      grad_feed_dict[gp] = g * (np.random.random(g.shape) < 0.8).astype('float32')
      tot += (g ** 2).sum()
      # if i % 10 == 0:
      #   grad_feed_dict[gp] += (np.random.random(g.shape) - 0.5) * 0.01
    print('gradient norm', tot ** 0.5)
    sess.run(momentum, feed_dict = grad_feed_dict)
    print('gradient_descent', time.time() - tt)
    # log1 = tf.Print(W1, [W1], 'W1:')
    # log2 = tf.Print(b1, [b1], 'b1:')
    # sess.run(log1)
    # sess.run(log2)

    # if i % 10 == 0:
    if tot ** 0.5 > 100:# i % 10 == 0:
      embed()
      tt = time.time()
      sim.visualize(memo, show = True, interval = 1, export = exp)
      sim.visualize(memo, show = False, interval = 1, export = exp)
      print('visualize', time.time() - tt)
      if memo.loss < 1e-3:
        break

    print('Iter {:5d} time {:.3f} loss {}'.format(
        i, time.time() - t, memo.loss))
    print(i, memo.loss, file = flog)
    flog.flush()

    
if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
