import tensorflow as tf
from time_integration import InitialState, UpdatedState


class Simulation:

  def __init__(self,
               sess,
               res,
               num_particles,
               num_steps,
               controller,
               gravity=(0, -9.8),
               dt=0.01,
               batch_size=1,
               E=4500):
    self.E = E
    self.num_steps = num_steps
    self.num_particles = num_particles
    self.scale = 30
    self.res = res
    self.sess = sess
    self.initial_velocity = tf.placeholder(shape=(2,), dtype=tf.float32)
    assert batch_size == 1
    self.batch_size = batch_size
    self.initial_state = InitialState(
        self, initial_velocity=self.initial_velocity)
    self.updated_states = []
    self.gravity = gravity
    self.dt = dt

    # Boundary condition
    previous_state = self.initial_state

    # Controller is a function that takes states and generates action
    for i in range(num_steps):
      new_state = UpdatedState(self, previous_state, controller)
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
    img = np.ones((scale * self.res[0], scale * self.res[1], 3), dtype=np.float)

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
        img, (int(self.res[0] * scale * 0.101), 0),
        (int(self.res[0] * scale * 0.101), self.res[1] * scale),
        color=(0, 0, 0))

    #mass = mass.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = grid.swapaxes(0, 1)[::-1, :, ::-1]
    #grid = np.concatenate([grid, grid[:, :, 0:1] * 0], axis=2)
    # mass = cv2.resize(
    #     mass, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # grid = cv2.resize(
    #     grid, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img
