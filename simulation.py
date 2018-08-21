import tensorflow as tf
from vector_math import *
import numpy as np
from time_integration import InitialSimulationState, UpdatedSimulationState

class Simulation:

  def __init__(self,
               grid_res,
               num_particles,
               num_time_steps=None,
               controller=None,
               gravity=(0, -9.8),
               dt=0.01,
               dx=0.1,
               batch_size=1,
               E=4500):
    self.E = E
    self.num_time_steps = num_time_steps
    self.num_particles = num_particles
    self.scale = 30
    self.grid_res = grid_res

    assert batch_size == 1
    self.batch_size = batch_size
    self.initial_state = InitialSimulationState(self)
    self.updated_states = []
    self.gravity = gravity
    self.dt = dt
    self.dx = dx

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
    img = np.ones((scale * self.grid_res[0], scale * self.grid_res[1], 3), dtype=np.float)

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
    return (self.initial_state.velocity, self.initial_state.position,
            self.initial_state.deformation_gradient)

  def get_initial_state(self, position):
    initial_velocity = np.zeros(shape=[1, self.num_particles, 2])
    deformation_gradient = identity_matrix +\
                           np.zeros(shape=(self.batch_size, self.num_particles, 1, 1)),
    return (initial_velocity, position, deformation_gradient)
