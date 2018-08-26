import tensorflow as tf
from vector_math import *
import numpy as np
from time_integration import InitialSimulationState, UpdatedSimulationState
from memo import Memo


class Simulation:

  def __init__(self,
               sess,
               grid_res,
               num_particles,
               controller=None,
               gravity=(0, -9.8),
               dt=0.01,
               dx=None,
               batch_size=1):
    self.sess = sess
    self.num_particles = num_particles
    self.scale = 30
    self.grid_res = grid_res
    self.dim = len(self.grid_res)
    if dx is None:
      dx = 1.0 / grid_res[0]
    self.batch_size = batch_size
    
    self.bc_parameter = np.zeros(shape=(1, ) + self.grid_res + (1,), dtype=np.float32)
    self.bc_parameter += 0.5 # Coefficient of friction
    self.bc_normal = np.zeros(shape=(1, ) + self.grid_res + (self.dim,), dtype=np.float32)
    boundary_thickness = 3
    self.bc_normal[:, :boundary_thickness] = (1, 0)
    self.bc_normal[:, self.grid_res[0] - boundary_thickness - 1:] = (-1, 0)
    self.bc_normal[:, :, :boundary_thickness] = (0, 1)
    self.bc_normal[:, :, self.grid_res[1] - boundary_thickness - 1:] = (0, -1)

    self.initial_state = InitialSimulationState(self, controller)
    self.grad_state = InitialSimulationState(self, controller)
    self.updated_states = []
    self.gravity = gravity
    self.dt = dt
    self.dx = dx
    self.inv_dx = 1.0 / dx
    self.updated_state = UpdatedSimulationState(self, self.initial_state)
    self.controller = controller
    
  def visualize(self, memo, interval=1):
    import math
    import cv2
    import numpy as np
    
    scale = self.scale

    # Pure-white background
    background = np.ones(
      (self.grid_res[0], self.grid_res[1], 3), dtype=np.float)
    
    for i in range(self.grid_res[0]):
      for j in range(self.grid_res[1]):
        normal = self.bc_normal[0][i][j]
        if np.linalg.norm(normal) != 0:
          background[i][j] *= 0.7
          
    background = cv2.resize(background, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    for i, s in enumerate(memo.steps):
      if i % interval != 0:
        continue
      pos = s[0][0] * self.inv_dx + 0.5
    
      scale = self.scale
    
      img = background.copy()
    
      for p in pos:
        x, y = tuple(map(lambda t: math.ceil(t * scale), p))
        cv2.circle(img, (y, x), radius=1, color=(0.2, 0.2, 0.2), thickness=-1)
    
      img = img.swapaxes(0, 1)[::-1, :, ::-1]
      cv2.imshow('Differentiable MPM Simulator', img)
      cv2.waitKey(1)

  def initial_state_place_holder(self):
    return self.initial_state.to_tuple()

  def run(self, num_steps, initial_state, initial_feed_dict={}, iteration_feed_dict={}, loss=None):
    memo = Memo()
    memo.initial_feed_dict = initial_feed_dict
    memo.iteration_feed_dict = iteration_feed_dict
    memo.initial_state = initial_state
    
    initial_evaluated = []
    for t in initial_state:
      if isinstance(t, tf.Tensor):
        initial_evaluated.append(self.sess.run(t, initial_feed_dict))
      else:
        initial_evaluated.append(t)
        
    memo.steps = [initial_evaluated]
    for i in range(num_steps):
      feed_dict = {
        self.initial_state.to_tuple(): memo.steps[-1]
      }
      feed_dict.update(iteration_feed_dict)

      memo.steps.append(
          self.sess.run(
              self.updated_state.to_tuple(),
              feed_dict=feed_dict))
    if loss is not None:
      feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
      feed_dict.update(iteration_feed_dict)
      memo.loss = self.sess.run(loss, feed_dict=feed_dict)
    return memo
  
  @staticmethod
  def replace_none_with_zero(grads, data):
    ret = []
    for g, t in zip(grads, data):
      if g is None:
        ret.append(tf.zeros_like(t))
      else:
        ret.append(g)
    return tuple(ret)
    

  def gradients(self, loss, memo, variables):
    # loss = loss(initial_state)
    variables = tuple(variables)
    
    last_grad_sym = tf.gradients(ys=loss, xs=self.initial_state.to_tuple())

    last_grad_sym_valid = self.replace_none_with_zero(last_grad_sym, memo.steps[-1])
    
    feed_dict = {
      self.initial_state.to_tuple(): memo.steps[-1]
    }
    feed_dict.update(memo.iteration_feed_dict)
    last_grad_valid = self.sess.run(last_grad_sym_valid, feed_dict=feed_dict)
    
    for v in variables:
      assert tf.convert_to_tensor(v).dtype == tf.float32, v
    grad = [np.zeros(shape=v.shape, dtype=np.float32) for v in variables]

    # partial S / partial var
    step_grad_variables = tf.gradients(
        ys=self.updated_state.to_tuple(),
        xs=variables,
        grad_ys=self.grad_state.to_tuple())
    
    step_grad_variables = self.replace_none_with_zero(step_grad_variables, variables)

    # partial S / partial S'
    step_grad_states = tf.gradients(
        ys=self.updated_state.to_tuple(),
        xs=self.initial_state.to_tuple(),
        grad_ys=self.grad_state.to_tuple())
    
    step_grad_states = self.replace_none_with_zero(step_grad_states, self.initial_state.to_tuple())
    
    for i in reversed(range(1, len(memo.steps))):
      if any(v is not None for v in step_grad_variables):
        feed_dict = {
          self.initial_state.to_tuple(): memo.steps[i - 1],
          self.updated_state.to_tuple(): memo.steps[i],
          self.grad_state.to_tuple(): last_grad_valid
        }
        feed_dict.update(memo.iteration_feed_dict)
        grad_acc = self.sess.run(step_grad_variables, feed_dict=feed_dict)
        for g, a in zip(grad, grad_acc):
          g += a
      if i != 0:
        feed_dict={
          self.initial_state.to_tuple(): memo.steps[i - 1],
          self.updated_state.to_tuple(): memo.steps[i],
          self.grad_state.to_tuple(): last_grad_valid
        }
        feed_dict.update(memo.iteration_feed_dict)
        last_grad_valid = self.sess.run(step_grad_states, feed_dict=feed_dict)
    
    parameterized_initial_state = tuple([v for v in memo.initial_state if isinstance(v, tf.Tensor)])
    parameterized_initial_state_indices = [i for i, v in enumerate(memo.initial_state)
                                           if isinstance(v, tf.Tensor)]
    
    def pick(l):
      return tuple(l[i] for i in parameterized_initial_state_indices)
    
    initial_grad_sym = tf.gradients(
      ys=parameterized_initial_state,
      xs=variables,
      grad_ys=pick(self.grad_state.to_tuple())
    )
    
    initial_grad_sym_valid = self.replace_none_with_zero(initial_grad_sym, variables)
    
    if any(v is not None for v in initial_grad_sym_valid):
      feed_dict = {}
      feed_dict[parameterized_initial_state] = pick(memo.steps[0])
      feed_dict[pick(self.grad_state.to_tuple())] = pick(last_grad_valid)
      grad_acc = self.sess.run(initial_grad_sym_valid, feed_dict=feed_dict)
      for g, a in zip(grad, grad_acc):
        g += a

    return grad

  def get_initial_state(self,
                        position,
                        velocity=None,
                        particle_mass=None,
                        particle_volume=None,
                        youngs_modulus=None,
                        poissons_ratio=None):
    if velocity is not None:
      initial_velocity = velocity
    else:
      initial_velocity = np.zeros(
          shape=[self.batch_size, self.num_particles, 2])
    deformation_gradient = identity_matrix +\
                           np.zeros(shape=(self.batch_size, self.num_particles, 1, 1)),
    affine = identity_matrix * 0 + \
                           np.zeros(shape=(self.batch_size, self.num_particles, 1, 1)),
    batch_size = self.batch_size
    num_particles = len(position[0])

    if particle_mass is None:
      particle_mass = np.ones(shape=(batch_size, num_particles, 1))
    if particle_volume is None:
      particle_volume = np.ones(shape=(batch_size, num_particles, 1))
    if youngs_modulus is None:
      youngs_modulus = np.ones(shape=(batch_size, num_particles, 1)) * 10
    if type(youngs_modulus) in [int, float]:
      youngs_modulus = np.ones(shape=(batch_size, num_particles, 1)) * youngs_modulus
    
    if poissons_ratio is None:
      poissons_ratio = np.ones(shape=(batch_size, num_particles, 1)) * 0.3

    return (position, initial_velocity, deformation_gradient, affine,
            particle_mass, particle_volume, youngs_modulus, poissons_ratio)
