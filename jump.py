from functools import partial
import tensorflow as tf
import cv2
import os
from simulation import *
import simulation
import time
from vector_math import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(sess):
  t = time.time()
  sim = Simulation(sess=sess, num_particles=simulation.num_particles, num_steps=3, res=(25, 25))
  print("Building time: {:.4f}s".format(time.time() - t))
  t = time.time()
  # os.system('cd outputs && rm *.png')

  t = time.time()

  final_state = sim.states[-1].controller_states[0, 0]

  final_position = [final_state[num_groups // 2 * 4], final_state[num_groups // 2 * 4 + 1]]

  goal_input = sim.initial_state.goal
  loss = (final_position[0] - sim.res[0] * goal_input[0, 0, 0])**2 + (
                                                                        final_position[1] - sim.res[1] * goal_input[0, 0, 1])**2

  current_velocity = np.array([0, 0], dtype=np.float32)
  results = [s.get_evaluated() for s in sim.states]

  # Initial particle samples
  particles = [[]]

  for i, offset in enumerate(group_offsets):
    for x in range(sample_density):
      for y in range(sample_density):
        scale = 0.2
        u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]) * scale  + 0.2
        v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]) * scale  + 0.1
        particles[0].append([sim.res[0] * u, sim.res[1] * v])
  assert len(particles[0]) == num_particles

  counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  opt = ly.optimize_loss(
    loss=loss,
    learning_rate=lr,
    optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9),
    variables=trainables,
    global_step=counter)

  sim.sess.run(tf.global_variables_initializer())

  for i in range(1000000):
    goal = [0.50 + random.random() * 0.0, 0.4 + random.random() * 0.2]
    feed_dict = {
      sim.initial_state.position:
        particles,
      sim.initial_velocity:
        current_velocity,
      sim.initial_state.deformation_gradient:
        identity_matrix +
        np.zeros(shape=(sim.batch_size, num_particles, 1, 1)),
      goal_input: [[goal]]
    }
    pos, l, _, evaluated = sim.sess.run(
      [final_position, loss, opt, results], feed_dict=feed_dict)
    print('  loss', l)

    for j, r in enumerate(evaluated):
      frame = i * (sim.num_steps + 1) + j
      img = sim.visualize(i=frame, r=r)
      scale = sim.scale
      cv2.circle(
        img, (int(sim.res[0] * scale * goal[1]), int(sim.res[1] * scale * goal[0])),
        radius=8,
        color=(0.0, 0.9, 0.0),
        thickness=-1)
      img = img.swapaxes(0, 1)[::-1, :, ::-1]
      output_fn='outputs/{:04d}.png'.format(frame)
      cv2.imshow('Particles', img)
      cv2.imwrite(output_fn, img * 255)
      cv2.waitKey(1)

    print('time', time.time() - t)

  print("Running time: {:.4f}s".format(time.time() - t))


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
