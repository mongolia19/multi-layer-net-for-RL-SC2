from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from pysc2.lib import features


def build_net(minimap, screen, info, msize, ssize, num_action, ntype):
  if ntype == 'atari':
    return build_atari(minimap, screen, info, msize, ssize, num_action)
  elif ntype == 'fcn':
    return build_fcn(minimap, screen, info, msize, ssize, num_action)
  else:
    raise 'FLAGS.net must be atari or fcn'


def build_atari(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc, mconv1], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value


def max_unpool_2x2(x, shape):
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1]*2, shape[2]*2]))
    return inference


def build_linear_model(trajectories, linear_units):
    """
    Function to build a subgraph
    """
    import math
    with tf.name_scope('linear_att'):
        weights = tf.get_variable(name='linear_weights', initializer=
            tf.truncated_normal((linear_units, linear_units), stddev=1.0 / math.sqrt(float(150))))
        biases = tf.get_variable(name='linear_biases', initializer=tf.zeros((linear_units)))
        linear_trans = tf.matmul(trajectories, weights) + biases
    return linear_trans


def build_fcn(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  # player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
  mconv1 = layers.fully_connected(layers.flatten(minimap),
                         num_outputs=16,
                         scope='mconv1', activation_fn=tf.nn.relu)
  # mconv2 = layers.conv2d(mconv1,
  #                        num_outputs=32,
  #                        kernel_size=3,
  #                        stride=1,
  #                        scope='mconv2')
  mconv2 = mconv1
  # sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
  #                        num_outputs=16,
  #                        kernel_size=5,
  #                        stride=1,
  #                        scope='sconv1')
  # pooled = tf.layers.max_pooling2d(inputs=sconv1, pool_size=[2, 2], strides=2)
  # unpooled = max_unpool_2x2(pooled, sconv1.shape)
  # pooled_1 = tf.layers.max_pooling2d(inputs=unpooled, pool_size=[2, 2], strides=2)
  # sconv2 = layers.conv2d(sconv1,
  #                        num_outputs=32,
  #                        kernel_size=3,
  #                        stride=1,
  #                        scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions
  # feat_conv = tf.concat([mconv2, sconv2], axis=3)
  # feat_conv = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2)], axis=1)
  # feat_conv = pooled_1
  # spatial_action = layers.conv2d(feat_conv,
  #                                num_outputs=1,
  #                                kernel_size=1,
  #                                stride=1,
  #                                activation_fn=None,
  #                                scope='spatial_action')
  spatial_action = layers.fully_connected(mconv1,ssize*ssize,activation_fn=tf.nn.relu, scope='spatial_layer')
  attention_weighted_vec = spatial_action
  # feat_conv_attention = tf.layers.flatten(spatial_action)
  # att_size = feat_conv_attention.shape[-1].value
  # attention_k = layers.fully_connected(feat_conv_attention,num_outputs=att_size, activation_fn=None, scope='attention_k')
  # with tf.variable_scope('traj_embedding',reuse=tf.AUTO_REUSE) as scope:
  #   attention_k = build_linear_model(feat_conv_attention, att_size)
  #   scope.reuse_variables()
  #   attention_q = build_linear_model(feat_conv_attention, att_size)
  #   attention_v = build_linear_model(feat_conv_attention, att_size)
  #   k_q_matrix = tf.matmul(attention_k, attention_q,transpose_b=True)
  #   attention_weighted_vec = tf.matmul(k_q_matrix, attention_v)
  attention_weighted_vec = tf.layers.flatten(attention_weighted_vec)
  # create spatial action mask
  spatial_action_mask = np.zeros((ssize,ssize))
  for i in range(ssize):
      for j in range(ssize):
          if i == ssize//2:
              spatial_action_mask[i][j] = 1
          if j == ssize//2:
              spatial_action_mask[i][j] = 1
          if i == ssize//3:
              spatial_action_mask[i][j] = 1
          if j == ssize//3:
              spatial_action_mask[i][j] = 1
          if i == 2*ssize//3:
              spatial_action_mask[i][j] = 1
          if j == 2 * ssize // 3:
              spatial_action_mask[i][j] = 1
  spatial_action_mask = np.ndarray.flatten(spatial_action_mask)
  # mask spatial actions
  attention_weighted_vec = tf.multiply(attention_weighted_vec, spatial_action_mask)
  spatial_action_out = tf.nn.softmax(layers.flatten(attention_weighted_vec))

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv1), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=128,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  # num_attack_action = 3
  local_fc = layers.flatten(mconv1)
  att_ct_layer = layers.fully_connected(local_fc, num_outputs=4, activation_fn=tf.nn.relu, scope="attack_control_layer")
  attack_action = layers.fully_connected(att_ct_layer, num_outputs=num_action, activation_fn = None, scope='action_attack_layer')
  att_mask = np.zeros((num_action), np.float)
  att_mask[0:8] = 1
  att_mask[12:19] = 1
  attack_action_mask = tf.multiply(attack_action, att_mask)


  # num_motion_action = 10
  global_fc = layers.flatten(mconv2)
  mot_ct_layer = layers.fully_connected(global_fc, num_outputs=4, activation_fn=tf.nn.relu, scope="motion_control_layer")
  motion_action = layers.fully_connected(mot_ct_layer, num_outputs=num_action, activation_fn = None,
                                         scope='action_motion_layer')
  mot_mask = np.zeros((num_action), np.float)
  mot_mask[:11] = 1
  mot_mask[330:350] = 1
  motion_action_mask = tf.multiply(motion_action, mot_mask)

  policy_router = layers.fully_connected(feat_fc,
                                              num_outputs=2,
                                              activation_fn=tf.nn.softmax,
                                              scope='policy_router')
  a = tf.reshape(tf.convert_to_tensor([1, 0]), [1,2])
  b = tf.convert_to_tensor([0, 1])
  left = tf.boolean_mask(policy_router, np.array([True, False]), axis=1)
  
  right = tf.boolean_mask(policy_router, np.array([False, True]), axis=1)
  left = tf.reduce_sum(left)
  right = tf.reduce_sum(right)
  # merged_attack_motion = layers.flatten(attack_action_mask)
  masked_attack_vector = tf.multiply(attack_action_mask, left)
  masked_motion_vector = tf.multiply(motion_action_mask, right)

  # merged_attack_motion = layers.flatten(motion_action_mask)
  merged_attack_motion = tf.add(masked_motion_vector, masked_attack_vector)
  # c = tf.equal(b, tf.cast(policy_shape, tf.int32))
  def conditon_tf(condition_tensor_1, condition_tensor_2 ,tensor_a, tensor_b):
      tile_tensor_a = tf.py_func(_condition_tf, [condition_tensor_1, condition_tensor_2, tensor_a, tensor_b], tf.float32)
      return tile_tensor_a

  def _condition_tf(a, b, c, d):
      if a == b:
          return c
      else:
          return d

  def f1():return layers.flatten(attack_action_mask)
  def f2():return layers.flatten(motion_action_mask)

  # merged_attack_motion = tf.cond(c, f1, f2)
  # merged_attack_motion = conditon_tf(policy_router, b, layers.flatten(attack_action_mask), layers.flatten(motion_action_mask))
  # if policy_router[0]>0:
  #     merged_attack_motion = layers.flatten(attack_action_mask)
  # else:
  #     merged_attack_motion = layers.flatten(motion_action_mask)
  non_spatial_action = layers.fully_connected(merged_attack_motion,
                                                  num_outputs=num_action,
                                                  activation_fn=tf.nn.softmax,
                                                  scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])
  return spatial_action_out, non_spatial_action, value
