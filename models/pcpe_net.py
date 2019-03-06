import tensorflow as tf
import numpy as np
import math
import sys
import os
from tensorflow.python.framework import function

# test serial structure to predict translation or rotation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def get_trans_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    feature_dim = point_cloud.get_shape()[2].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1, feature_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay) # check the maximum along point dimension

    max_indices = tf.argmax(net, axis=1)
    net = tf_util.max_pool2d(net, [num_point, 1],
                            padding='VALID', scope='maxpool')

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net, _, _ = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net, _, _ = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    net, out_weight, out_biases = tf_util.fully_connected(net, 3, activation_fn=None, scope='output')

    return net, max_indices


def get_rot_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    feature_dim = point_cloud.get_shape()[2].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1, feature_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1_rot', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2_rot', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3_rot', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4_rot', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5_rot', bn_decay=bn_decay) # check the maximum along point dimension

    max_indices = tf.argmax(net, axis=1)
    net = tf_util.max_pool2d(net, [num_point, 1],
                            padding='VALID', scope='maxpool_rot')

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net, _, _ = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1_rot', bn_decay=bn_decay)
    net, _, _ = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2_rot', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    net, out_weight, out_biases = tf_util.fully_connected(net, 3, activation_fn=None, scope='output_rot')

    return net, max_indices