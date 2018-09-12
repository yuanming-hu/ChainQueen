import numpy as np
import tensorflow as tf
from vector_math import *

use_cuda = False

if use_cuda:
   import mpm3d

kernel_size = 3

class SimulationState:

  def __init__(self, sim):
    self.sim = sim
    self.dim = sim.dim
    dim = self.dim
    self.grid_shape = (self.sim.batch_size,) + self.sim.grid_res + (1,)
    self.affine = tf.zeros(shape=(self.sim.batch_size, dim, dim, sim.num_particles), dtype=tf_precision)
    self.acceleration = tf.zeros(shape=(self.sim.batch_size, dim, sim.num_particles), dtype=tf_precision)
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
    self.step_count = None
    self.kernels = None
    self.debug = None

  def center_of_mass(self, left = None, right = None):
    return tf.reduce_sum(self.position[:, :, left:right] * self.particle_mass[:, :, left:right], axis=2) *\
           (1 / tf.reduce_sum(self.particle_mass[:, :, left:right], axis=2))

  def get_state_names(self):
    return [
        'position', 'velocity', 'deformation_gradient', 'affine',
        'particle_mass', 'particle_volume', 'youngs_modulus', 'poissons_ratio', 'step_count', 'acceleration'
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
        'acceleration': self.acceleration,
        'controller_states': self.controller_states,
        'grid_mass': self.grid_mass,
        'grid_velocity': self.grid_velocity,
        'kernels': self.kernels,
        'particle_mass': self.particle_mass,
        'particle_volume': self.particle_volume,
        'youngs_modulus': self.youngs_modulus,
        'poissons_ratio': self.poissons_ratio,
        'step_count': self.step_count,
        'debug': self.debug,
    }
    ret_filtered = {}
    for k, v in ret.items():
      if v is not None:
        ret_filtered[k] = v
    return ret_filtered

  def to_tuple(self):
    evaluated = self.get_evaluated()
    return tuple([evaluated[k] for k in self.get_state_names()])

  def __getitem__(self, item):
    return self.get_evaluated()[item]

  def compute_kernels(self, positions):
    # (x, y, dim)
    grid_node_coord = [[(i, j) for j in range(3)] for i in range(3)]
    grid_node_coord = np.array(grid_node_coord)[None, :, :, :, None]
    frac = (positions - tf.floor(positions - 0.5))[:, None, None, :, :]
    assert frac.shape[3] == self.dim
    #print('frac', frac.shape)
    #print('grid_node_coord', grid_node_coord.shape)

    # (batch, x, y, dim, p) - (?, x, y, dim, ?)
    x = tf.abs(frac - grid_node_coord)
    #print('x', x.shape)

    mask = tf.cast(x < 0.5, tf_precision)
    y = mask * (0.75 - x * x) + (1 - mask) * (0.5 * (1.5 - x)**2)
    #print('y', y.shape)
    y = tf.reduce_prod(y, axis=3, keepdims=True)
    return y


class InitialSimulationState(SimulationState):

  def __init__(self, sim, controller=None):
    super().__init__(sim)
    dim = self.dim
    self.t = 0
    num_particles = sim.num_particles
    self.position = tf.placeholder(
        tf_precision, [self.sim.batch_size, dim, num_particles], name='position')

    self.velocity = tf.placeholder(
        tf_precision, [self.sim.batch_size, dim, num_particles], name='velocity')
    self.deformation_gradient = tf.placeholder(
        tf_precision, [self.sim.batch_size, dim, dim, num_particles], name='dg')
    self.particle_mass = tf.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='particle_mass')
    self.particle_volume = tf.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='particle_volume')
    self.youngs_modulus = tf.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='youngs_modulus')
    self.poissons_ratio = tf.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='poissons_ratio')
    self.grid_mass = tf.zeros(shape=self.grid_shape, dtype=tf_precision)
    self.grid_velocity = tf.zeros(shape=self.grid_shape, dtype=tf_precision)
    if self.dim == 2:
      self.kernels = tf.zeros(shape=(self.sim.batch_size,
                                     kernel_size, kernel_size, 1, num_particles), dtype=tf_precision)
    self.step_count = tf.zeros(shape=(), dtype=np.int32)

    self.controller = controller
    if controller is not None:
      self.actuation, self.debug = controller(self)
      self.actuation = matmatmul(self.deformation_gradient, matmatmul(self.actuation, transpose(self.deformation_gradient)))


class UpdatedSimulationState(SimulationState):
  def cuda(self, sim, previous_state, controller):
    self.particle_mass = tf.identity(previous_state.particle_mass)
    self.particle_volume = tf.identity(previous_state.particle_volume)
    self.youngs_modulus = tf.identity(previous_state.youngs_modulus)
    self.poissons_ratio = tf.identity(previous_state.poissons_ratio)
    self.step_count = previous_state.step_count + 1

    if controller:
      self.actuation, self.debug = controller(self)
    else:
      self.actuation = np.zeros(shape=(self.sim.batch_size, self.dim, self.dim, self.sim.num_particles))
    print(self.actuation[:, 1, 1, :])


    self.t = previous_state.t + self.sim.dt

    self.position, self.velocity, self.deformation_gradient, self.affine, _, _ = \
      mpm3d.mpm(previous_state.position, previous_state.velocity,
                previous_state.deformation_gradient, previous_state.affine, dx=sim.dx,
                dt=sim.dt, gravity=sim.gravity, resolution=sim.grid_res, E=sim.E, nu=sim.nu,
                V_p=sim.V_p, m_p=sim.m_p, actuation=self.actuation)


  def __init__(self, sim, previous_state, controller=None):
    super().__init__(sim)
    dim = self.dim
    if dim == 3 or use_cuda:
      print("Running with cuda")
      self.cuda(sim, previous_state, controller=controller)
      return


    # 2D time integration
    self.particle_mass = tf.identity(previous_state.particle_mass)
    self.particle_volume = tf.identity(previous_state.particle_volume)
    self.youngs_modulus = tf.identity(previous_state.youngs_modulus)
    self.poissons_ratio = tf.identity(previous_state.poissons_ratio)
    self.step_count = previous_state.step_count + 1

    self.t = previous_state.t + self.sim.dt
    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               dim))
    
    position = previous_state.position
    
    minimum_positions = np.zeros(shape=previous_state.position.shape, dtype=np_precision)
    minimum_positions[:, :, :] = self.sim.dx * 2
    maximum_positions = np.zeros(shape=previous_state.position.shape, dtype=np_precision)
    for i in range(dim):
      maximum_positions[:, i, :] = (self.sim.grid_res[i] - 2) * self.sim.dx
    # Safe guard
    position = tf.clip_by_value(position, minimum_positions, maximum_positions)

    # Rasterize mass and velocity
    base_indices = tf.cast(
        tf.floor(position * sim.inv_dx - 0.5), tf.int32)
    base_indices = tf.transpose(base_indices, perm=[0, 2, 1])
    batch_size = self.sim.batch_size
    num_particles = sim.num_particles

    # Add the batch size indices
    base_indices = tf.concat(
        [
            tf.zeros(shape=(batch_size, num_particles, 1), dtype=tf.int32),
            base_indices,
        ],
        axis=2)

    # print('base indices', base_indices.shape)
    self.grid_mass = tf.zeros(shape=(batch_size, self.sim.grid_res[0],
                                     self.sim.grid_res[1], 1))

    # Compute stress tensor (Kirchhoff stress instead of First Piola-Kirchhoff stress)
    self.deformation_gradient = previous_state.deformation_gradient

    # Lame parameters
    mu = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    lam = self.youngs_modulus * self.poissons_ratio / ((
        1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
    # (b, 1, p) -> (b, 1, 1, p)
    mu = mu[:, :, None, :]
    lam = lam[:, :, None, :]
    # Corotated elasticity
    # P(F) = dPhi/dF(F) = 2 mu (F-R) + lambda (J-1)JF^-T

    r, s = polar_decomposition(self.deformation_gradient)
    j = determinant(self.deformation_gradient)[:, None, None, :]

    # Note: stress_tensor here is right-multiplied by F^T
    self.stress_tensor1 = 2 * mu * matmatmul(
        self.deformation_gradient - r, transpose(self.deformation_gradient))

    if previous_state.controller:
      self.stress_tensor1 += previous_state.actuation

    self.stress_tensor2 = lam * (j - 1) * j * identity_matrix

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid_velocity = tf.zeros(shape=(batch_size, self.sim.grid_res[0],
                                         self.sim.grid_res[1], dim))

    self.kernels = self.compute_kernels(position * sim.inv_dx)
    assert self.kernels.shape == (batch_size, kernel_size, kernel_size, 1, num_particles)

    self.velocity = previous_state.velocity

    # Quadratic B-spline kernel
    for i in range(kernel_size):
      for j in range(kernel_size):
        delta_indices = np.zeros(
            shape=(self.sim.batch_size, 1, dim + 1), dtype=np.int32)

        for b in range(batch_size):
          delta_indices[b, 0, :] = [b, i, j]
        self.grid_mass = self.grid_mass + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], 1),
            indices=base_indices + delta_indices,
            updates=tf.transpose((self.particle_mass * self.kernels[:, i, j, :, :]), perm=[0, 2, 1]))

        # (b, dim, p)
        delta_node_position = np.array([i, j])[None, :, None]
        # xi - xp
        offset = (tf.floor(position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf_precision) -
                  position * sim.inv_dx) * sim.dx

        grid_velocity_contributions = self.particle_mass * self.kernels[:, i, j, :] * (
            self.velocity + matvecmul(previous_state.affine, offset))
        grid_force_contributions = self.particle_volume * self.kernels[:, i, j, :] * (
            matvecmul(self.stress_tensor, offset) *
            (-4 * self.sim.dt * self.sim.inv_dx * self.sim.inv_dx))
        self.grid_velocity = self.grid_velocity + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], dim),
            indices=base_indices + delta_indices,
            updates=tf.transpose(grid_velocity_contributions + grid_force_contributions, perm=[0, 2, 1]))
    assert self.grid_mass.shape == (batch_size, self.sim.grid_res[0],
                                    self.sim.grid_res[1],
                                    1), 'shape={}'.format(self.grid_mass.shape)

    self.grid_velocity += self.grid_mass * np.array(
        self.sim.gravity)[None, None, None, :] * self.sim.dt
    self.grid_velocity = self.grid_velocity / tf.maximum(1e-30, self.grid_mass)

    sticky_mask = tf.cast(self.sim.bc_parameter == -1, tf_precision)
    self.grid_velocity *= (1 - sticky_mask)
    
    mask = tf.cast(
        tf.reduce_sum(self.sim.bc_normal**2, axis=3, keepdims=True) != 0,
        tf_precision)
    normal_component_length = tf.reduce_sum(
        self.grid_velocity * self.sim.bc_normal, axis=3, keepdims=True)
    perpendicular_component = self.grid_velocity - self.sim.bc_normal * normal_component_length
    perpendicular_component_length = tf.sqrt(
        tf.reduce_sum(perpendicular_component**2, axis=3, keepdims=True) + 1e-7)
    normalized_perpendicular_component = perpendicular_component / tf.maximum(
        perpendicular_component_length, 1e-7)
    perpendicular_component_length = tf.sign(perpendicular_component_length) * \
                                     tf.maximum(tf.abs(perpendicular_component_length) +
                                                tf.minimum(normal_component_length, 0) * self.sim.bc_parameter, 0)
    projected_velocity = sim.bc_normal * tf.maximum(
        normal_component_length,
        0) + perpendicular_component_length * normalized_perpendicular_component
    self.grid_velocity = self.grid_velocity * (
        1 - mask) + mask * projected_velocity

    # Resample velocity and local affine velocity field
    self.velocity *= 0
    for i in range(kernel_size):
      for j in range(kernel_size):
        delta_indices = np.zeros(
          shape=(self.sim.batch_size, 1, dim + 1), dtype=np.int32)
        for b in range(batch_size):
          delta_indices[b, 0, :] = [b, i, j]


        #print('indices', (base_indices + delta_indices).shape)
        grid_v = tf.transpose(tf.gather_nd(
            params=self.grid_velocity,
            indices=base_indices + delta_indices), perm=[0, 2, 1])
        self.velocity = self.velocity + grid_v * self.kernels[:, i, j, :]

        delta_node_position = np.array([i, j])[None, :, None]

        # xi - xp
        offset = (tf.floor(position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf_precision) -
                  position * sim.inv_dx) * sim.dx
        assert offset.shape == position.shape
        weighted_node_velocity = grid_v * self.kernels[:, i, j, :]
        # weighted_node_velocity = tf.transpose(weighted_node_velocity, perm=[0, 2, 1])
        self.affine += outer_product(weighted_node_velocity, offset)

    if sim.damping != 0:
      self.velocity *= np.exp(-sim.damping * sim.dt)

    self.affine *= 4 * sim.inv_dx * sim.inv_dx
    dg_change = identity_matrix + self.sim.dt * self.affine
    #print(dg_change.shape)
    #print(previous_state.deformation_gradient)
    self.deformation_gradient = matmatmul(dg_change,
                                          previous_state.deformation_gradient)

    # Advection
    self.position = position + self.velocity * self.sim.dt
    assert self.position.shape == previous_state.position.shape
    assert self.velocity.shape == previous_state.velocity.shape
    self.acceleration = (self.velocity - previous_state.velocity) * (1 / self.sim.dt)
