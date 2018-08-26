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
               num_time_steps=None,
               controller=None,
               gravity=(0, -9.8),
               dt=0.01,
               dx=1,
               batch_size=1):
    self.sess = sess
    self.num_time_steps = num_time_steps
    self.num_particles = num_particles
    self.scale = 30
    self.grid_res = grid_res
    self.batch_size = batch_size
    self.initial_state = InitialSimulationState(self)
    self.grad_state = InitialSimulationState(self)
    self.updated_states = []
    self.gravity = gravity
    self.dt = dt
    self.dx = dx
    self.inv_dx = 1.0 / dx
    self.updated_state = UpdatedSimulationState(self, self.initial_state)

    # Boundary condition
    previous_state = self.initial_state

    # Controller is a function that takes states and generates action
    if controller is not None:
      assert num_time_steps is not None
      for i in range(num_time_steps):
        new_state = UpdatedSimulationState(self, previous_state, controller)
        self.updated_states.append(new_state)
        previous_state = new_state

      self.states = [self.initial_state] + self.updated_states

  def visualize_particles(self, pos):
    import math
    import cv2
    import numpy as np
    pos = pos * self.inv_dx

    scale = self.scale

    img = np.ones(
        (scale * self.grid_res[0], scale * self.grid_res[1], 3), dtype=np.float)

    for p in pos:
      x, y = tuple(map(lambda t: math.ceil(t * scale), p))
      cv2.circle(img, (y, x), radius=1, color=(0.2, 0.2, 0.2), thickness=-1)

    img = img.swapaxes(0, 1)[::-1, :, ::-1]

    cv2.imshow('Differentiable MPM Simulator', img)
    cv2.waitKey(1)

  def visualize(self, i, r):
    import math
    import cv2
    import numpy as np
    pos = r['position'][0]
    # mass = r['mass'][0]
    # grid = r['grid'][0][:, :, 1:2]
    # J = determinant(r['deformation_gradient'])[0]
    #5 print(grid.min(), grid.max())
    # grid = grid / (1e-5 + np.abs(grid).max()) * 4 + 0.5
    # grid = np.clip(grid, 0, 1)
    # kernel_sum = np.sum(r['kernels'][0], axis=(1, 2))
    # if 0 < i < 3:
    #   np.testing.assert_array_almost_equal(kernel_sum, 1, decimal=3)
    #   np.testing.assert_array_almost_equal(
    #       mass.sum(), num_particles, decimal=3)

    scale = self.scale

    # Pure-white background
    img = np.ones(
        (scale * self.grid_res[0], scale * self.grid_res[1], 3), dtype=np.float)

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
        img, (int(self.grid_res[0] * scale * 0.101), 0),
        (int(self.grid_res[0] * scale * 0.101), self.grid_res[1] * scale),
        color=(0, 0, 0))

    #mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    # mass = cv2.resize(
    #     mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # grid = cv2.resize(
    #     grid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img

  def initial_state_place_holder(self):
    return self.initial_state.to_tuple()

  def run(self, initial_state, num_steps, initial_feed_dict={}, iteration_feed_dict={}):
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

      memo.steps.append(
          self.sess.run(
              self.updated_state.to_tuple(),
              feed_dict=feed_dict))
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
    
    last_grad_valid = self.sess.run(last_grad_sym_valid, feed_dict={
        self.initial_state.to_tuple(): memo.steps[-1]})
    
    for v in variables:
      assert v.dtype == tf.float32
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
        grad_acc = self.sess.run(step_grad_variables,
            feed_dict={
                self.updated_state.to_tuple(): memo.steps[i],
                self.grad_state.to_tuple(): last_grad_valid
            })
        for g, a in zip(grad, grad_acc):
          g += a
      if i != 0:
        last_grad_valid = self.sess.run(step_grad_states,
            feed_dict={
                self.initial_state.to_tuple(): memo.steps[i - 1],
                self.updated_state.to_tuple(): memo.steps[i],
                self.grad_state.to_tuple(): last_grad_valid
            })
    
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
    if poissons_ratio is None:
      poissons_ratio = np.ones(shape=(batch_size, num_particles, 1)) * 0.3

    return (position, initial_velocity, deformation_gradient, affine,
            particle_mass, particle_volume, youngs_modulus, poissons_ratio)
