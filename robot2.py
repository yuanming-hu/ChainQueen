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

import pygmo as pg
import pygmo_plugins_nonfree as ppnf


class RobotDefinition:
  def __init__(self, group_sizes, actuated_groups, group_offsets, res, bc):
    self.bc = bc
    self.group_sizes = group_sizes
    self.actuated_groups = actuated_groups
    self.group_offsets = group_offsets
    self.res = res
    
class OLController:
  def __init__(self, robot, act_res):
    self.robot = robot
    self.act_res = act_res
    self.actuation_seq = tf.Variable(1.0 * tf.random_normal(shape=(1, (robot.sim_timesteps // self.act_res), robot.num_actuators), dtype=np.float32), trainable=True)
    
    
  def get_outputs(self, state):
    #TODO: Refactor this
    controller_inputs = []
    for i in range(num_groups):
      mask = self.robot.particle_mask(i * self.robot.group_num_particles,
                           (i + 1) * self.robot.group_num_particles)[:, :, None] * (
                               1.0 / self.robot.group_num_particles)
      pos = tf.reduce_sum(mask * state.position, axis=1, keepdims=False)
      vel = tf.reduce_sum(mask * state.velocity, axis=1, keepdims=False)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append(self.robot.goal_position)
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (1, 6 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (1, 6 * num_groups, 1)
  
  
  
    actuation = tf.expand_dims(self.actuation_seq[0, state.step_count // self.act_res, :], 0)   
    total_actuation = 0
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = self.robot.particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      zeros = tf.zeros(shape=(1, self.robot.num_particles))
      act = make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + matmatmul(
        act, transpose(state['deformation_gradient']))
        
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation}
    return total_actuation, debug
    
    
    

  #Should be thought of as a struct for now


class Robot:


  def __init__(self, sess, robot_definition, goal_input_pos, goal_input_vel, sim_timesteps=200, dt=0.005, gravity=(0, 0), actuation_bounds=(-1.0, 1.0), ym_bounds = (9.0, 11.0), neural_net=False, sample_density=20,
               loss_weights_input = np.array([0.0, 0.0]), constraint_weights_input = np.array([1.0, 1.0])  ):
               
               
               
    self.group_sizes = robot_definition.group_sizes 
    self.actuated_groups = robot_definition.actuated_groups
    self.group_offsets = robot_definition.group_offsets
    self.bc = robot_definition.bc
    self.sess = sess
    self.goal_input_pos = goal_input_pos
    self.goal_input_vel = goal_input_vel
               
    self.sim_timesteps=sim_timesteps        
    self.group_num_particles = sample_density**2
    self.num_particles = self.group_num_particles * len(self.group_sizes) #self.group_sizes needs to be set, as does self.actuated_groups and self.group_offsets, goal_input
    self.num_actuators = len(self.actuated_groups)
    
    #For now, just the target of a single end-effector
    self.goal_position = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='goal_position')
    self.goal_velocity = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='goal_velocity')
    
    self.controller = self.controller(neural_net=neural_net)
    
    
    
    

    self.sim = Simulation(
      dt=dt,
      num_particles=self.num_particles,
      grid_res=robot_definition.res,
      gravity=gravity,
      controller=self.controller.get_outputs,
      batch_size=1,
      bc=self.bc,
      sess=self.sess)
    
    
    
    
    initial_positions = [[] for _ in range(1)]
    
    for i, offset in enumerate(self.group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[0].append([u, v])
    assert len(initial_positions[0]) == self.num_particles

    self.youngs_modulus = tf.Variable(np.mean(ym_bounds) * tf.ones(shape = [1, self.num_particles, 1], dtype = tf.float32), trainable=True)
    self.initial_state = self.sim.get_initial_state(
        position=np.array(initial_positions), youngs_modulus=tf.identity(self.youngs_modulus))
        
    self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    self.sess.run(tf.global_variables_initializer())
  
    self.sim.set_initial_state(initial_state=self.initial_state)
    
    
    loss = self.construct_loss(loss_weights_input)
    self.sym = self.sim.gradients_sym(loss, variables=self.trainables)
    #self.sim.add_point_visualization(pos=self.goal_input_pos, color=(0, 1, 0), radius=3)
    
    #final_position, final_velocity = self.get_final_state()
    #self.sim.add_vector_visualization(pos=final_position, vector=final_velocity, color=(0, 0, 1), scale=50)   
    #self.sim.add_point_visualization(pos=final_position, color=(1, 0, 0), radius=3)
    
    IPython.embed()
    
    #main idea here: __init__ gets called after everything is set
    #TODO: need to set sess, and bc, goal, group_sizes, group_offset, actuated_groups

  def controller(self, neural_net = False, act_res=10):
    if neural_net:
      #hardcode two layers
      controller = NNController(self, neurons=18) #for now, a single layer
    else:
      controller = OLController(self, act_res=act_res)      
    return controller
      
  def construct_loss(self, loss_weights):
    #loss_weights needs to be an np.array
    final_position, final_velocity = self.get_final_state()
    loss_position = tf.reduce_sum((final_position - self.goal_position) ** 2)
    loss_velocity = tf.reduce_sum((final_velocity - self.goal_velocity) ** 2)
    
    return tf.tensordot(loss_weights, tf.stack([loss_position, loss_velocity], 0), axes=1)
      
  def get_final_state(self):
    head = 2 #TODO: unhardcode
    final_state = self.sim.initial_state['debug']['controller_inputs']
    s = head * 6    
    final_position = final_state[:, s:s+2]
    final_velocity = final_state[:, s + 2: s + 4]
    return final_position, final_velocity    
    
  def particle_mask(self, start, end):
    r = tf.range(0, self.num_particles)
    return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


  def particle_mask_from_group(self, g):
    return self.particle_mask(g * self.group_num_particles, (g + 1) * self.group_num_particles)
  
  
  
  def eval_sim(self, loss_tensor):
    memo = self.sim.run(
        initial_state=self.initial_state,
        num_steps=self.sim_timesteps,
        iteration_feed_dict={self.goal_position: self.goal_input_pos, self.goal_velocity : self.goal_input_vel},
        loss=loss_tensor)
    grad = self.sim.eval_gradients(sym=self.sym, memo=memo)
    return memo.loss, grad, memo
    
  def assignment_helper(self, x):
    assignments = []
    idx = 0
    x = x.astype(np.float32)
    for v in self.trainables:
      #first, get count:
      var_cnt = tf.size(v).eval()
      assignments += [v.assign(tf.reshape(x[idx:idx+var_cnt],v.shape))]
      idx += var_cnt
    self.sess.run(assignments)
  
  def motion_planner(self):
    raise NotImplementedError
    
    
if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
  
  #create robot definition:
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
  
  res = (30, 30)
  bc = get_bounding_box_bc(res)
  
  bc[0][:, :, :5] = -1 # Sticky
  bc[1][:, :, :5] = 0 # Sticky
  
  robot_definition = RobotDefinition(group_sizes, actuations, group_offsets, res, bc)

  goal_range = 0.0 #This can change if we want stochasticity
  goal_input = np.array(
    [[0.75 + (random.random() - 0.5) * goal_range * 2,
      0.5 + (random.random() - 0.5) * goal_range]],
    dtype=np.float32)

  #TODO: move goal_input
  with tf.Session(config=sess_config) as sess:
    robot = Robot(sess, robot_definition, goal_input, np.zeros((1, 2), dtype=np.float32), sim_timesteps=200, dt=0.005, gravity=(0, 0), actuation_bounds=(-8.0, 8.0), ym_bounds = (9.0, 11.0), neural_net=False, sample_density=20,
               loss_weights_input = np.array([0.0, 0.0], dtype=np.float32), constraint_weights_input = np.array([1.0, 1.0], dtype=np.float32)  )
    
  
