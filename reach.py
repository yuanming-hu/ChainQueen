import random
import os
from simulation import Simulation, get_bounding_box_bc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.layers as ly
from vector_math import *
import IPython
import copy

lr = 1.0

sample_density = 40
group_num_particles = sample_density**2
goal_range = 0.0
batch_size = 1
actuation_strength = 8

nn_control = True
use_bfgs = Flase
wolfe_search = False
num_acts = 200

config = 'B'
if config == 'A':
  num_steps = 150
else:
  num_steps = 200

if config == 'A':
  # Robot A
  num_groups = 5
  group_offsets = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
  group_sizes = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
  actuations = [0, 4]
  head = 2
  gravity = (0, -2)
elif config == 'B':
  # Finger
  num_links = 1
  group_sizes = []
  group_offsets = []
  actuations = []
  group_size = [(0.5, 2 / num_links), (0.5, 2 / num_links), (1, 1 / num_links)]
  for i in range(num_links):
    group_offsets += [(1, group_size[0][1] *i + 0), (1.5, group_size[1][1] *i + 0), (1, group_size[2][1] *i + 2)]
    group_sizes += copy.deepcopy(group_size)
    actuations += [0  + 3*i, 1 + 3*i]
  num_groups = len(group_sizes)
  
  head = 2
  gravity = (0, 0)
elif config == 'C':
  # Robot B
  num_groups = 7
  group_offsets = [(0, 0), (0.5, 0), (0, 1), (1, 1), (2, 1), (2, 0), (2.5, 0)]
  group_sizes = [(0.5, 1), (0.5, 1), (1, 1), (1, 1), (1, 1), (0.5, 1), (0.5, 1)]
  actuations = [0, 1, 5, 6]
  fixed_groups = []
  head = 3
  gravity = (0, -2)
else:
  print('Unknown config {}'.format(config))

num_particles = group_num_particles * num_groups
num_actuators = len(actuations)


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# NN weights
if nn_control:
  W1 = tf.Variable(
      0.02 * tf.random_normal(shape=(len(actuations), 6 * len(group_sizes))),
      trainable=True)
  b1 = tf.Variable([0.0] * len(actuations), trainable=True)
else:
  actuation_seq = tf.Variable(0.1 * tf.random_normal(shape=(1, num_acts, num_actuators), dtype=np.float32), trainable=True)

def step_callback(dec_vec):
  pass

def main(sess):
  t = time.time()

  goal = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  # Define your controller here
  def controller(state):    
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, :, None] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(mask * state.position, axis=1, keepdims=False)
      vel = tf.reduce_sum(mask * state.velocity, axis=1, keepdims=False)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append(goal)
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, 6 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, 6 * num_groups, 1)
    # Batch, 6 * num_groups, 1
    if nn_control:
      intermediate = tf.matmul(W1[None, :, :] +
                               tf.zeros(shape=[batch_size, 1, 1]), controller_inputs)
      # Batch, #actuations, 1
      assert intermediate.shape == (batch_size, len(actuations), 1)
      assert intermediate.shape[2] == 1
      intermediate = intermediate[:, :, 0]
      # Batch, #actuations
      actuation = tf.tanh(intermediate + b1[None, :]) * actuation_strength
    else:
      #IPython.embed()
      actuation = tf.expand_dims(actuation_seq[0, state.step_count // (num_steps // num_acts), :], 0)
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + matmatmul(
        act, transpose(state['deformation_gradient']))
    return total_actuation, debug
  
  res = (30, 30)
  bc = get_bounding_box_bc(res)
  
  if config == 'B':
    bc[0][:, :, :5] = -1 # Sticky
    bc[1][:, :, :5] = 0 # Sticky

  sim = Simulation(
      dt=0.005,
      num_particles=num_particles,
      grid_res=res,
      gravity=gravity,
      controller=controller,
      batch_size=batch_size,
      bc=bc,
      sess=sess)
  print("Building time: {:.4f}s".format(time.time() - t))

  final_state = sim.initial_state['debug']['controller_inputs']
  s = head * 6
  
  final_position = final_state[:, s:s+2]
  final_velocity = final_state[:, s + 2: s + 4]
  gamma = 0.0
  loss1 = tf.reduce_sum((final_position - goal) ** 2)
  loss2 = tf.reduce_sum(final_velocity ** 2)

  loss = loss1 + gamma * loss2

  initial_positions = [[] for _ in range(batch_size)]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[b].append([u, v])
  assert len(initial_positions[0]) == num_particles

  youngs_modulus =tf.Variable(10.0 * tf.ones(shape = [1, num_particles, 1], dtype = tf.float32), trainable=True)
  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=tf.identity(youngs_modulus))
      
  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  if use_bfgs:
    B = [tf.Variable(tf.eye(tf.size(trainable)), trainable=False) for trainable in trainables]
  
  sess.run(tf.global_variables_initializer())
  
  sim.set_initial_state(initial_state=initial_state)
  
  sym = sim.gradients_sym(loss, variables=trainables)
  sim.add_point_visualization(pos=goal, color=(0, 1, 0), radius=3)
  sim.add_vector_visualization(pos=final_position, vector=final_velocity, color=(0, 0, 1), scale=50)
 
  sim.add_point_visualization(pos=final_position, color=(1, 0, 0), radius=3)

  if config == 'A':
    goal_input = np.array(
      [[0.5 + (random.random() - 0.5) * goal_range * 2,
        0.6 + (random.random() - 0.5) * goal_range] for _ in range(batch_size)],
      dtype=np.float32)
  elif config == 'B':
    goal_input = np.array(
    [[0.75 + (random.random() - 0.5) * goal_range * 2,
      0.5 + (random.random() - 0.5) * goal_range] for _ in range(batch_size)],
    dtype=np.float32)
  # Optimization loop
  #IPython.embed()
  #In progress code
  '''
  memo = sim.run(
        initial_state=initial_state,
        num_steps=num_steps,
        iteration_feed_dict={goal: goal_input},
        loss=loss)
  IPython.embed()
  
  
  def loss_callback():
    memo = sim.run(
        initial_state=initial_state,
        num_steps=num_steps,
        iteration_feed_dict={goal: goal_input},
        loss=loss)
    
    return loss
  '''
  
  c1 = 1e-4
  c2 = 0.9
  
  def eval_sim():
    memo = sim.run(
        initial_state=initial_state,
        num_steps=num_steps,
        iteration_feed_dict={goal: goal_input},
        loss=loss)
    grad = sim.eval_gradients(sym=sym, memo=memo)
    return memo.loss, grad, memo
  
  def flatten_trainables():
    return tf.concat([tf.squeeze(ly.flatten(trainable)) for trainable in trainables], 0)
    
  def flatten_vectors(vectors):
    return tf.concat([tf.squeeze(ly.flatten(vector)) for vector in vectors], 0)
    
  def assignment_run(xs):
    sess.run([trainable.assign(x) for x, trainable in zip(xs, trainables)])
  
  def f_and_grad_step(step_size, x, delta_x):
    old_x = [x_i.eval() for x_i in x]
    assignment_run([x_i + step_size * delta_x_i for x_i, delta_x_i in zip(x, delta_x)]) #take step
    loss, grad, _ = eval_sim()
    assignment_run(old_x) #revert
    return loss, grad
    
  def wolfe_1(delta_x, new_f, current_f, current_grad, step_size):    
    valid = new_f <= current_f + c1 * step_size * tf.tensordot(flatten_vectors(current_grad), flatten_vectors(delta_x), 1)   
    return valid.eval()
    
  def wolfe_2(delta_x, new_grad, current_grad, step_size):   
    valid = np.abs(tf.tensordot(flatten_vectors(new_grad), flatten_vectors(delta_x), 1).eval()) <= -c2 * tf.tensordot(flatten_vectors(current_grad), flatten_vectors(delta_x), 1).eval()
    return valid
    
  
  def zoom(a_min, a_max, search_dirs, current_f, current_grad):
    while True:
      a_mid = (a_min + a_max) / 2.0
      print('a_min: ', a_min, 'a_max: ', a_max, 'a_mid: ', a_mid)
      step_loss_min, step_grad_min = f_and_grad_step(a_min, trainables, search_dirs)
      step_loss, step_grad = f_and_grad_step(a_mid, trainables, search_dirs)      
      valid_1 = wolfe_1(search_dirs, step_loss, current_f, current_grad, a_mid)
      valid_2 = wolfe_2(search_dirs, step_grad, current_grad, a_mid)
      if not valid_1 or step_loss >= step_loss_min:
        a_max = a_mid
      else:
        if valid_2:
          return a_mid
        if tf.tensordot(flatten_vectors(step_grad), flatten_vectors(search_dirs), 1) * (a_max - a_min) >= 0:
          a_max = a_min
        a_min = a_mid
        
  loss_val, grad, memo = eval_sim() #TODO: this is to get dimensions, find a better way to do this without simming
  old_g_flat = [None] * len(grad)
  old_v_flat = [None] * len(grad)
  for i in range(1000000):
    t = time.time()
    
    loss_val, grad, memo = eval_sim()
    
    #BFGS update:
    #IPython.embed()
    
    if use_bfgs:
      bfgs = [None] * len(grad)
      B_update = [None] * len(grad)
      search_dirs = [None] * len(grad)    
      #TODO: for now, assuming there is only one trainable and one grad for ease
      for v, g, idx in zip(trainables, grad, range(len(grad))):
        g_flat = ly.flatten(g) 
        v_flat = ly.flatten(v)
        if B[idx] == None:
            B[idx] = tf.eye(tf.size(v_flat))
        if i > 0:          
          y_flat = tf.squeeze(g_flat - old_g_flat[idx])
          s_flat = tf.squeeze(v_flat - old_v_flat[idx])
          B_s_flat = tf.tensordot(B[idx], s_flat, 1)
          term_1 = -tf.tensordot(B_s_flat, tf.transpose(B_s_flat), 0) / tf.tensordot(s_flat, B_s_flat, 1)
          term_2 = tf.tensordot(y_flat, y_flat, 0) / tf.tensordot(y_flat, s_flat, 1)
          B_update[idx] = B[idx].assign(B[idx] + term_1 + term_2)    
          sess.run([B_update[idx]])
        
        

        if tf.abs(tf.matrix_determinant(B[idx])).eval() < 1e-6:
          sess.run( [ B[idx].assign(tf.eye(tf.size(v_flat))) ] )
          search_dir = -tf.transpose(g_flat)
        else:
          #search_dir = -tf.matrix_solve_ls(B[idx],tf.transpose(g_flat), l2_regularizer=0.0, fast=True) #adding regularizer for stability
          search_dir = -tf.matmul(tf.linalg.inv(B[idx]), tf.transpose(g_flat))   #TODO: inverse bad,speed htis up
        search_dir_reshape = tf.reshape(search_dir, g.shape)
        search_dirs[idx] = search_dir_reshape
        old_g_flat[idx] = g_flat
        old_v_flat[idx] = v_flat.eval()
          #TODO: B upate
      
      #Now it's linesearch time
      if wolfe_search:
        a_max = 1.0
        a_1 = a_max / 2.0
        a_0 = 0.0
        
        iterate = 1
        while True:
          step_loss, step_grad = f_and_grad_step(a_1, trainables, search_dirs)
          print(a_1)
          valid_1 = wolfe_1(search_dirs, step_loss, loss_val, grad, a_1)
          valid_2 = wolfe_2(search_dirs, step_grad, grad, a_1)
          print('wolfe 1: ', valid_1, 'wolfe 2: ', valid_2)
          if (not valid_1) or (iterate > 1 and step_loss > loss_val):
            print('cond1')
            a = zoom(a_0, a_1, search_dirs, loss_val, grad)
          if valid_2:
            print('cond2')
            a = a_1
            break
          if tf.tensordot(flatten_vectors(step_grad), flatten_vectors(search_dirs), 1).eval() >= 0:
            print('cond3')
            a = zoom(a_1, a_0, search_dirs, current_f, current_grad)
            break
          print('no cond')
          temp = a_1
          a_1 = (a_1 + a_max) / 2.0
          a_0 = temp        
          iterate+=1
          if iterate > 5:
            #close enough
            a = a_1
            break
      else:
        a = lr
      for v, idx in zip(trainables, range(len(grad))):
        print('final a ', a)
        bfgs[idx] = v.assign(v + search_dirs[idx] * a)
      sess.run(bfgs)
      print('stepped!!')
    else:
      gradient_descent = [
        v.assign(v - lr * g) for v, g in zip(trainables, grad)
      ]
      sess.run(gradient_descent)
      
    print('iter {:5d} time {:.3f} loss {:.4f}'.format(
        i, time.time() - t, memo.loss))
    if i % 1 == 0:
      sim.visualize(memo)
    
  #in progress code
  '''
  optimizer = tf.contrib.opt.ScipyOptimizerInterface(
    loss, method='BFGS')
  optimizer.minimize(sess, loss_callback=loss_callback)
  '''



if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
