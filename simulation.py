import tensorflow as tf
from vector_math import *
import numpy as np
from time_integration import InitialSimulationState, UpdatedSimulationState
from memo import Memo
import IPython

try:
  import taichi as tc
  from taichi import Task
  import ctypes
except:
  print("Warning: cannot import taichi or CUDA solver.")

def get_bounding_box_bc(res, boundary_thickness=3):
  if len(res) == 2:
    bc_parameter = np.zeros(
      shape=(1,) + res + (1,), dtype=np.float32)
    bc_parameter += 0.5  # Coefficient of friction
    bc_normal = np.zeros(shape=(1,) + res + (len(res),), dtype=np.float32)
    bc_normal[:, :boundary_thickness] = (1, 0)
    bc_normal[:, res[0] - boundary_thickness - 1:] = (-1, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1)
    bc_normal[:, :, :boundary_thickness] = (0, 1)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1)
  else:
    assert len(res) == 3
    bc_parameter = np.zeros(
      shape=(1,) + res + (1,), dtype=np.float32)
    bc_parameter += 0.5  # Coefficient of friction
    bc_normal = np.zeros(shape=(1,) + res + (len(res),), dtype=np.float32)
    bc_normal[:, :boundary_thickness] = (1, 0, 0)
    bc_normal[:, res[0] - boundary_thickness - 1:] = (-1, 0, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1, 0)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1, 0)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1, 0)
    bc_normal[:, :, :, :boundary_thickness] = (0, 0, 1)
    bc_normal[:, :, :, res[2] - boundary_thickness - 1:] = (0, 0, -1)
  return bc_parameter, bc_normal

class Simulation:

  def __init__(self,
               sess,
               grid_res,
               num_particles,
               controller=None,
               gravity=(0, -9.8),
               dt=0.01,
               damping=0.0,
               dx=None,
               bc=None,
               E=10,
               nu=0.3,
               m_p=1,
               V_p=1,
               batch_size=1,
               scale=None):
    self.dim = len(grid_res)
    self.InitialSimulationState = InitialSimulationState
    self.UpdatedSimulationState = UpdatedSimulationState
    if self.dim == 2:
      self.identity_matrix = identity_matrix
    else:
      self.identity_matrix = identity_matrix_3d

    assert batch_size == 1, "Only batch_size = 1 is supported."

    self.sess = sess
    self.num_particles = num_particles
    if scale is None:
      self.scale = 900 // grid_res[0]
    else:
      self.scale = scale
    self.grid_res = grid_res
    self.dim = len(self.grid_res)
    if dx is None:
      dx = 1.0 / grid_res[0]
    self.batch_size = batch_size
    self.damping = damping

    if bc is None and self.dim == 2:
      bc = get_bounding_box_bc(grid_res)
      
    if bc is not None:
      self.bc_parameter, self.bc_normal = bc
    self.initial_state = self.InitialSimulationState(self, controller)
    self.grad_state = self.InitialSimulationState(self, controller)
    self.updated_states = []
    self.gravity = gravity
    self.dx = dx
    self.dt = dt
    self.E = E
    self.nu = nu
    self.m_p = m_p
    self.V_p = V_p
    self.inv_dx = 1.0 / dx
    self.updated_state = self.UpdatedSimulationState(self, previous_state=self.initial_state, controller=controller)
    self.controller = controller
    self.parameterized_initial_state = None
    self.point_visualization = []
    self.vector_visualization = []
    self.frame_counter = 0

  def visualize_2d(self, memo, interval=1, batch=0, export=None, show=False):
    import math
    import cv2
    import numpy as np

    scale = self.scale

    b = batch
    # Pure-white background
    background = np.ones(
      (self.grid_res[0], self.grid_res[1], 3), dtype=np.float)

    for i in range(self.grid_res[0]):
      for j in range(self.grid_res[1]):
        if self.bc_parameter[0][i][j] == -1:
          background[i][j][0] = 0
        normal = self.bc_normal[0][i][j]
        if np.linalg.norm(normal) != 0:
          background[i][j] *= 0.7
    background = cv2.resize(
      background, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    for i, (s, points, vectors) in enumerate(zip(memo.steps, memo.point_visualization, memo.vector_visualization)):
      if i % interval != 0:
        continue


      pos = s[0][b] * self.inv_dx + 0.5
      pos = np.transpose(pos)
      youngs = np.ndarray.flatten(s[6][b])

      scale = self.scale

      img = background.copy()


      
      for young, p in zip(youngs, pos):
        x, y = tuple(map(lambda t: math.ceil(t * scale), p))
        intensity = (young) / 50.0
        cv2.circle(img, (y, x), radius=1, color=(0.0, intensity, 1.0), thickness=-1)

      for dot in points:
        coord, color, radius = dot
        coord = (coord * self.inv_dx + 0.5) * scale
        cv2.circle(img, (coord[b][1], coord[b][0]), color=color, radius=radius, thickness=-1)

      for line in vectors:
        pos, vec, color, gamma = line
        pos = (pos * self.inv_dx + 0.5) * scale
        vec = vec * gamma + pos
        cv2.line(img, (pos[b][1], pos[b][0]), (vec[b][1], vec[b][0]), color = color, thickness = 1)

      img = img.swapaxes(0, 1)[::-1, :, ::-1]
      if export is None or show:
        cv2.imshow('Differentiable MPM Simulator', img)
        cv2.waitKey(1)
      if export is not None:
        export(img)

    if export is not None:
      export.wait()

  def visualize_3d(self, memo, interval=1, batch=0, export=None, show=False):
    if export:
      frame_count_delta = self.frame_counter
    else:
      frame_count_delta = 0
    print("Warning: skipping the 0th frame..")
    for i, (s, points, vectors) in enumerate(zip(memo.steps, memo.point_visualization, memo.vector_visualization)):
      if i % interval != 0 or i == 0:
        continue
      pos = s[0][batch].copy()
      #print(np.mean(pos, axis=(0)))
      task = Task('write_partio_c')
      ptr = pos.ctypes.data_as(ctypes.c_void_p).value
      task.run(str(self.num_particles),
               str(ptr), '{:04d}.bgeo'.format(i // interval + frame_count_delta))
      self.frame_counter += 1

  def visualize(self, memo, interval=1, batch=0, export=None, show=False):
    if self.dim == 2:
      self.visualize_2d(memo, interval, batch, export, show)
    else:
      self.visualize_3d(memo, interval, batch, export, show)


  def initial_state_place_holder(self):
    return self.initial_state.to_tuple()

  def evaluate_points(self, state, extra={}):
    if self.point_visualization is None:
      return []
    pos_tensors = [p[0] for p in self.point_visualization]
    feed_dict = {self.initial_state.to_tuple(): state}
    feed_dict.update(extra)

    pos = self.sess.run(pos_tensors, feed_dict=feed_dict)
    return [(p,) + tuple(list(r)[1:])
            for p, r in zip(pos, self.point_visualization)]

  def evaluate_vectors(self, state, extra = {}):
    if self.vector_visualization is None:
      return []
    pos_tensors = [v[0] for v in self.vector_visualization]
    vec_tensors = [v[1] for v in self.vector_visualization]
    feed_dict = {self.initial_state.to_tuple(): state}
    feed_dict.update(extra)
    pos = self.sess.run(pos_tensors, feed_dict=feed_dict)
    vec = self.sess.run(vec_tensors, feed_dict=feed_dict)
    return [(p,v) + tuple(list(r)[2:]) for p, v, r in zip(pos, vec, self.vector_visualization)]

  def run(self,
          num_steps,
          initial_state=None,
          initial_feed_dict={},
          iteration_feed_dict={},
          loss=None):
    memo = Memo()
    memo.initial_feed_dict = initial_feed_dict
    memo.iteration_feed_dict = iteration_feed_dict
    if initial_state is None:
      initial_state = self.initial_state
    memo.initial_state = initial_state

    initial_evaluated = []
    for t in initial_state:
      if isinstance(t, tf.Tensor):
        initial_evaluated.append(self.sess.run(t, initial_feed_dict))
      else:
        initial_evaluated.append(t)

    memo.steps = [initial_evaluated]
    memo.point_visualization.append(self.evaluate_points(memo.steps[0], iteration_feed_dict))
    memo.vector_visualization.append(self.evaluate_vectors(memo.steps[0], iteration_feed_dict))

    for i in range(num_steps):
      feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
      feed_dict.update(iteration_feed_dict)

      memo.steps.append(
          self.sess.run(
              self.updated_state.to_tuple(),
              feed_dict=feed_dict))
      memo.point_visualization.append(self.evaluate_points(memo.steps[-1], iteration_feed_dict))
      memo.vector_visualization.append(self.evaluate_vectors(memo.steps[-1], iteration_feed_dict))
      
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

  def set_initial_state(self, initial_state):
    self.parameterized_initial_state = initial_state

  def gradients_sym(self, loss, variables):
    # loss = loss(initial_state)
    variables = tuple(variables)

    last_grad_sym = tf.gradients(ys=loss, xs=self.initial_state.to_tuple())

    last_grad_sym_valid = self.replace_none_with_zero(
        last_grad_sym, self.initial_state.to_tuple())

    for v in variables:
      assert tf.convert_to_tensor(v).dtype == tf_precision, v

    # partial S / partial var
    step_grad_variables = tf.gradients(
        ys=self.updated_state.to_tuple(),
        xs=variables,
        grad_ys=self.grad_state.to_tuple())

    step_grad_variables = self.replace_none_with_zero(step_grad_variables,
                                                      variables)

    # partial S / partial S'
    step_grad_states = tf.gradients(
        ys=self.updated_state.to_tuple(),
        xs=self.initial_state.to_tuple(),
        grad_ys=self.grad_state.to_tuple())

    step_grad_states = self.replace_none_with_zero(
        step_grad_states, self.initial_state.to_tuple())

    parameterized_initial_state = tuple([
        v for v in self.parameterized_initial_state if isinstance(v, tf.Tensor)
    ])
    parameterized_initial_state_indices = [
        i for i, v in enumerate(self.parameterized_initial_state)
        if isinstance(v, tf.Tensor)
    ]

    def pick(l):
      return tuple(l[i] for i in parameterized_initial_state_indices)

    initial_grad_sym = tf.gradients(
        ys=parameterized_initial_state,
        xs=variables,
        grad_ys=pick(self.grad_state.to_tuple()))

    initial_grad_sym_valid = self.replace_none_with_zero(
        initial_grad_sym, variables)

    sym = {}
    sym['last_grad_sym_valid'] = last_grad_sym_valid
    sym['initial_grad_sym_valid'] = initial_grad_sym_valid
    sym['step_grad_variables'] = step_grad_variables
    sym['step_grad_states'] = step_grad_states
    sym['parameterized_initial_state'] = parameterized_initial_state
    sym['pick'] = pick
    sym['variables'] = variables

    return sym

  def eval_gradients(self, sym, memo):
    last_grad_sym_valid = sym['last_grad_sym_valid']
    initial_grad_sym_valid = sym['initial_grad_sym_valid']
    step_grad_variables = sym['step_grad_variables']
    step_grad_states = sym['step_grad_states']
    parameterized_initial_state = sym['parameterized_initial_state']
    pick = sym['pick']
    variables = sym['variables']

    grad = [np.zeros(shape=v.shape, dtype=np.float32) for v in variables]
    feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
    feed_dict.update(memo.iteration_feed_dict)
    last_grad_valid = self.sess.run(last_grad_sym_valid, feed_dict=feed_dict)
    for i in reversed(range(1, len(memo.steps))):
      #print('backward step ', i)
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
        feed_dict = {
            self.initial_state.to_tuple(): memo.steps[i - 1],
            self.updated_state.to_tuple(): memo.steps[i],
            self.grad_state.to_tuple(): last_grad_valid
        }
        feed_dict.update(memo.iteration_feed_dict)
        last_grad_valid = self.sess.run(step_grad_states, feed_dict=feed_dict)

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
                        poissons_ratio=None,
                        deformation_gradient=None):
    acceleration = np.zeros(
      shape=[self.batch_size, self.dim, self.num_particles])
    if velocity is not None:
      initial_velocity = velocity
    else:
      initial_velocity = np.zeros(
          shape=[self.batch_size, self.dim, self.num_particles])
    if deformation_gradient is None:
      deformation_gradient = self.identity_matrix +\
                             np.zeros(shape=(self.batch_size, 1, 1, self.num_particles)),
    affine = self.identity_matrix * 0 + \
                           np.zeros(shape=(self.batch_size, 1, 1, self.num_particles)),
    batch_size = self.batch_size
    num_particles = self.num_particles

    if particle_mass is None:
      particle_mass = np.ones(shape=(batch_size, 1, num_particles)) * self.m_p
      self.m_p = 1
    if particle_volume is None:
      particle_volume = np.ones(shape=(batch_size, 1, num_particles)) * self.V_p
      self.V_p = 1
    if youngs_modulus is None:
      youngs_modulus = np.ones(shape=(batch_size, 1, num_particles)) * self.E
    elif type(youngs_modulus) in [int, float]:
      self.E = youngs_modulus
      youngs_modulus = np.ones(shape=(batch_size, 1, num_particles)) * youngs_modulus
    else:
      self.E = youngs_modulus[0][0][0]
      print(self.E)

    if poissons_ratio is None:
      poissons_ratio = np.ones(shape=(batch_size, 1, num_particles)) * 0.3

    return (position, initial_velocity, deformation_gradient, affine,
            particle_mass, particle_volume, youngs_modulus, poissons_ratio, 0, acceleration)

  def add_point_visualization(self, pos, color=(1, 0, 0), radius=3):
    self.point_visualization.append((pos, color, radius))
  
  def add_vector_visualization(self, pos, vector, color=(1, 0, 0), scale=10):
    self.vector_visualization.append((pos, vector, color, scale))
