import numpy as np
import tensorflow as tf
from vector_math import *

sticky = False
linear = False
dim = 2


class SimulationState:

  def __init__(self, sim):
    self.sim = sim
    self.affine = tf.zeros(shape=(self.sim.batch_size, sim.num_particles, 2, 2))
    self.position = None
    self.particle_mass = None
    self.particle_volume = None
    self.youngs_modulus = None
    self.poissons_ratio = None
    self.velocity = None
    self.deformation_gradient = None
    self.controller_states = None
    self.grid_mass = None
    self.grid_velocity = None
    self.kernels = None
    self.debug = None

  def get_state_names(self):
    return [
        'position', 'velocity', 'deformation_gradient', 'affine',
        'particle_mass', 'particle_volume', 'youngs_modulus', 'poissons_ratio'
    ]

  def get_evaluated(self):
    # # batch, particle, dimension
    # assert len(self.position.shape) == 3
    # assert len(self.position.shape) == 3
    # # batch, particle, matrix dimension1, matrix dimension2
    # assert len(self.deformation_gradient.shape) == 4
    # # batch, x, y, dimension
    # assert len(self.grid_mass.shape) == 4
    # assert len(self.grid_velocity.shape) == 4

    ret = {
        'affine': self.affine,
        'position': self.position,
        'velocity': self.velocity,
        'deformation_gradient': self.deformation_gradient,
        'controller_states': self.controller_states,
        'grid_mass': self.grid_mass,
        'grid_velocity': self.grid_velocity,
        'kernels': self.kernels,
        'particle_mass': self.particle_mass,
        'particle_volume': self.particle_volume,
        'youngs_modulus': self.youngs_modulus,
        'poissons_ratio': self.poissons_ratio,
        'debug': self.debug
    }
    ret_filtered = {}
    for k, v in ret.items():
      if v is not None:
        ret_filtered[k] = v
    return ret_filtered

  def to_tuples(self):
    evaluated = self.get_evaluated()
    return tuple([evaluated[k] for k in self.get_state_names()])

  def __getitem__(self, item):
    return self.get_evaluated()[item]

  @staticmethod
  def compute_kernels(positions):
    grid_node_coord = [[(i, j) for j in range(3)] for i in range(3)]
    grid_node_coord = np.array(grid_node_coord)[None, None, :, :]
    frac = (positions - tf.floor(positions - 0.5))[:, :, None, None, :]

    x = tf.abs(frac - grid_node_coord)
    #print('x', x.shape)

    mask = tf.cast(x < 0.5, tf.float32)
    y = mask * (0.75 - x * x) + (1 - mask) * (0.5 * (1.5 - x)**2)
    #print('y', y.shape)
    y = tf.reduce_prod(y, axis=4, keepdims=True)
    #print('y', y.shape)
    return y


class InitialSimulationState(SimulationState):

  def __init__(self, sim):
    super().__init__(sim)
    self.t = 0
    num_particles = sim.num_particles
    self.position = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, dim], name='position')

    self.velocity = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, dim], name='velocity')
    self.deformation_gradient = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, dim, dim], name='dg')
    self.particle_mass = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, 1],
        name='particle_mass')
    self.particle_volume = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, 1],
        name='particle_volume')
    self.youngs_modulus = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, 1],
        name='youngs_modulus')
    self.poissons_ratio = tf.placeholder(
        tf.float32, [self.sim.batch_size, num_particles, 1],
        name='poissons_ratio')
    self.grid_mass = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               1))
    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               dim))
    self.kernels = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               3, 3))


class UpdatedSimulationState(SimulationState):

  def __init__(self, sim, previous_state, controller=None):
    super().__init__(sim)
    self.particle_mass = previous_state.particle_mass
    self.particle_volume = previous_state.particle_volume
    self.youngs_modulus = previous_state.youngs_modulus
    self.poissons_ratio = previous_state.poissons_ratio

    self.t = previous_state.t + self.sim.dt
    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               dim))

    # Rasterize mass and velocity
    base_indices = tf.cast(
        tf.floor(previous_state.position * sim.inv_dx - 0.5), tf.int32)
    batch_size = self.sim.batch_size
    assert batch_size == 1
    # print('base indices', base_indices.shape)
    num_particles = sim.num_particles
    # TODO:
    # Add the batch size indices
    base_indices = tf.concat(
        [
            tf.zeros(shape=(batch_size, num_particles, 1), dtype=tf.int32),
            base_indices
        ],
        axis=2)
    # print('base indices', base_indices.shape)
    self.grid_mass = tf.zeros(
        shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], 1))

    # Compute stress tensor (Kirchhoff stress instead of First Piola-Kirchhoff stress)
    self.deformation_gradient = previous_state.deformation_gradient

    nu = 0.3
    # Lame parameters
    mu = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    lam = self.youngs_modulus * self.poissons_ratio / ((
        1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
    mu = mu[:, :, :, None]
    lam = lam[:, :, :, None]
    if linear:
      self.stress_tensor1 = mu * (
          transpose(self.deformation_gradient) + self.deformation_gradient -
          2 * identity_matrix)
      self.stress_tensor2 = lam * identity_matrix * (
          trace(self.deformation_gradient)[:, :, None, None] - dim)
    else:
      # Corotated elasticity
      # P(F) = dPhi/dF(F) = 2 mu (F-R) + lambda (J-1)JF^-T

      r, s = polar_decomposition(self.deformation_gradient)
      j = determinant(self.deformation_gradient)[:, :, None, None]

      # Note: stress_tensor here is right-multiplied by F^T
      self.stress_tensor1 = 2 * mu * matmatmul(
          self.deformation_gradient - r, transpose(self.deformation_gradient))

      if controller:
        act, self.debug = controller(previous_state)
        self.stress_tensor1 += act

      self.stress_tensor2 = lam * (j - 1) * j * identity_matrix

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid_velocity = tf.zeros(
        shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], dim))

    self.kernels = self.compute_kernels(previous_state.position * sim.inv_dx)
    assert self.kernels.shape == (batch_size, num_particles, 3, 3, 1)

    self.velocity = previous_state.velocity

    # Quadratic B-spline kernel
    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        #print((base_indices + delta_indices).shape)
        self.grid_mass = self.grid_mass + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], 1),
            indices=base_indices + delta_indices,
            updates=self.particle_mass * self.kernels[:, :, i, j])

        delta_node_position = np.array([i, j])[None, None, :]
        # xi - xp
        offset = (tf.floor(previous_state.position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf.float32) -
                  previous_state.position * sim.inv_dx) * sim.dx

        grid_velocity_contributions = self.particle_mass * self.kernels[:, :, i, j] * (
            self.velocity + matvecmul(previous_state.affine, offset))
        grid_force_contributions = self.particle_volume * self.kernels[:, :, i, j] * (
            matvecmul(self.stress_tensor, offset) *
            (-4 * self.sim.dt * self.sim.inv_dx * self.sim.inv_dx))
        self.grid_velocity = self.grid_velocity + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], dim),
            indices=base_indices + delta_indices,
            updates=grid_velocity_contributions + grid_force_contributions)
    assert self.grid_mass.shape == (batch_size, self.sim.grid_res[0],
                                    self.sim.grid_res[1], 1), 'shape={}'.format(
                                        self.grid_mass.shape)

    self.grid_velocity += self.grid_mass * np.array(
        self.sim.gravity)[None, None, None, :] * self.sim.dt
    self.grid_velocity = self.grid_velocity / tf.maximum(1e-30, self.grid_mass)

    # Boundary conditions
    if sticky:
      self.grid_velocity = self.grid_velocity * self.sim.bc
    else:
      # TODO: use sim.bc
      mask = np.zeros((1, self.sim.grid_res[0], self.sim.grid_res[1], 2))
      mask_x = np.zeros((1, self.sim.grid_res[0], self.sim.grid_res[1], 2))
      mask_y = np.zeros((1, self.sim.grid_res[0], self.sim.grid_res[1], 2))

      # bottom
      mask[:, :, :3, :] = 1
      mask_x[:, :, :3, 0] = 1
      mask_y[:, :, :3, 1] = 1

      friction = 0.5

      # X component
      projected_bottom = tf.sign(self.grid_velocity) * \
                         tf.maximum(tf.abs(self.grid_velocity) + friction * tf.minimum(0.0, self.grid_velocity[:, :, :, 1, None]), 0.0)
      self.grid_velocity = self.grid_velocity * (1 - mask) + (
          mask_x * projected_bottom) + mask_y * tf.maximum(
              self.grid_velocity, 0.0)

      mask = np.zeros((1, self.sim.grid_res[0], self.sim.grid_res[1], 2))
      mask[:, 3:self.sim.grid_res[0] - 3, :self.sim.grid_res[1] - 3] = 1
      self.grid_velocity = self.grid_velocity * mask

    # Resample velocity and local affine velocity field
    self.velocity *= 0
    for i in range(3):
      for j in range(3):
        assert batch_size == 1
        delta_indices = np.array([0, i, j])[None, None, :]
        self.velocity = self.velocity + tf.gather_nd(
            params=self.grid_velocity,
            indices=base_indices + delta_indices) * self.kernels[:, :, i, j]

        delta_node_position = np.array([i, j])[None, None, :]

        # xi - xp
        offset = (tf.floor(previous_state.position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf.float32) -
                  previous_state.position * sim.inv_dx) * sim.dx
        assert offset.shape == previous_state.position.shape
        weighted_node_velocity = tf.gather_nd(
            params=self.grid_velocity,
            indices=base_indices + delta_indices) * self.kernels[:, :, i, j]
        self.affine += outer_product(weighted_node_velocity, offset)

    self.affine *= 4 * sim.inv_dx * sim.inv_dx
    dg_change = identity_matrix + self.sim.dt * self.affine
    #print(dg_change.shape)
    #print(previous_state.deformation_gradient)
    self.deformation_gradient = matmatmul(dg_change,
                                          previous_state.deformation_gradient)

    # Advection
    self.position = previous_state.position + self.velocity * self.sim.dt
