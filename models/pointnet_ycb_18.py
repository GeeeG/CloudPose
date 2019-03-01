import tensorflow as tf
import numpy as np
import math
import sys
import os
from tensorflow.python.framework import function

# water flow model

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
                         scope='conv1_trans', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2_trans', bn_decay=bn_decay)

    point_feat = net

    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3_trans', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4_trans', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5_trans', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    global_feat_1 = tf_util.max_pool2d(point_feat, [num_point, 1], padding='VALID', scope='maxpool_trans')
    global_feat_2 = tf_util.max_pool2d(net, [num_point, 1],
                            padding='VALID', scope='maxpool_trans')

    net = tf.concat([global_feat_1, global_feat_2], 3)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net, _, _ = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                        scope='fc1_trans', bn_decay=bn_decay)
    net, _, _ = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                        scope='fc2_trans', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')

    net, _, _ = tf_util.fully_connected(net, 3, activation_fn=None, scope='output_trans')

    return net, max_indices


def get_rot_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    k = 5
    adj_matrix = tf_util.pairwise_xyz_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool_rot')

    net = global_feat

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net, _, _ = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)

    net, _, _ = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)


    net, out_weight, out_biases = tf_util.fully_connected(net, 3, activation_fn=None, scope='output')

    return net, global_feat, end_points, out_weight, out_biases, max_indices

