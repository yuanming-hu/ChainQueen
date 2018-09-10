import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import os

file_dir = os.path.dirname(os.path.realpath(__file__))
MPM_module = tf.load_op_library(os.path.join(file_dir, '../../build/libtaichi_tf_differentiable_mpm.so'))

mpm = MPM_module.mpm
p2g = MPM_module.p2g

@ops.RegisterGradient("Mpm")
def _mpm_grad_cc(op, *grads):
  return MPM_module.mpm_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3],
    op.outputs[0], op.outputs[1], op.outputs[2], op.outputs[3], op.outputs[4], op.outputs[5],
    grads[0], grads[1], grads[2], grads[3], grads[4], grads[5])
