import argparse
import math
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import open3d
import transforms3d
import matplotlib.pyplot as plt
import random
import scipy.io


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'losses'))
sys.path.append(os.path.join(BASE_DIR, 'ycb_video_data'))
import trans_distance
import angular_distance_taylor
from datetime import datetime

class_names = ["00_master_chef_can", "01_cracker_box", "02_sugar_box", "03_tomato_soup_can", "04_mustard_bottle", "05_tuna_fish_can", "06_pudding_box",
               "07_gelatin_box", "08_potted_meat_can", "09_banana", "10_pitcher_base", "11_bleach_cleanser", "12_bowl", "13_mug",
               "14_power_drill", "15_wood_block", "16_scissors", "17_large_marker", "18_large_clamp", "19_extra_large_clamp", "20_foam_brick"]
NUM_CLASS = 21

# Global settings, change according to your setup
data_dir = '/data_c/CloudPose_git/ycb_video_data_tfRecords/FPS1024/'
object_model_dir = "/data_c/CloudPose_git/object_model_tfrecord/obj_models.tfrecords"

target_cls = np.arange(21)

train_filenames = []
for cls in target_cls:
    for i in range(2):
        train_filename = data_dir + "train_files_FPS1024_" + str(cls) + "_" + str(i) + ".tfrecords"
        train_filenames.append(train_filename)


def decode(serialized_example, total_num_point):
  features = tf.parse_example(
      [serialized_example],
      features={
          'xyz': tf.FixedLenFeature([total_num_point, 3], tf.float32),
          'rgb': tf.FixedLenFeature([total_num_point, 3], tf.float32),
          'translation': tf.FixedLenFeature([3], tf.float32),
          'quaternion': tf.FixedLenFeature([4], tf.float32),
          'num_valid_points_in_segment': tf.FixedLenFeature([], tf.int64),
          'seq_id': tf.FixedLenFeature([], tf.int64),
          'frame_id': tf.FixedLenFeature([], tf.int64),
          'class_id': tf.FixedLenFeature([], tf.int64)
      })
  return features


def get_tfrecord_data(dataset, batch_size, total_num_point):
    dataset = dataset.map(lambda x: decode(x, total_num_point))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def reshape_element(element, batch_size, total_num_point):
    element['xyz'] = tf.reshape(element['xyz'], [batch_size, total_num_point, 3])
    element['rgb'] = tf.reshape(element['rgb'], [batch_size, total_num_point, 3])
    element['translation'] = tf.reshape(element['translation'], [batch_size, 3])
    element['quaternion'] = tf.reshape(element['quaternion'], [batch_size, 4])
    element['class_id'] = tf.reshape(element['class_id'], [batch_size])
    element['num_valid_points_in_segment'] = tf.reshape(element['num_valid_points_in_segment'], [batch_size])

    return element


# the object models can be used for visualization and inspectation during training
def read_and_decode_obj_model(filename):
    models = []
    labels = []

    features = {'label': tf.FixedLenFeature([], tf.int64),
                'model': tf.FixedLenFeature([2048, 6], tf.float32)}

    for examples in tf.python_io.tf_record_iterator(filename):
        example = tf.parse_single_example(examples, features=features)
        models.append(example['model'])
        labels.append(example['label'])

    return models, labels


def quat2axag(quat):
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    len2 = x * x + y * y + z * z
    theta = 2 * tf.acos(tf.maximum(tf.minimum(w, 1), -1))
    ax = tf.stack([x, y, z], axis=1)
    ax = ax / tf.expand_dims(tf.math.sqrt(len2),1)
    ag = theta
    axag = ax * tf.expand_dims(ag, 1)
    return axag

# ==============================================================================


def log_string(out_str, dir):
    dir.write(out_str + '\n')
    dir.flush()
    print(out_str)


# define the graph
def setup_graph(general_opts, train_opts, hyperparameters):
    tf.reset_default_graph()
    now = datetime.now()

    BATCH_SIZE = hyperparameters['batch_size']
    NUM_POINT = general_opts['num_point']
    TOTAL_NUM_POINT = general_opts['total_num_point']
    MAX_EPOCH = train_opts['max_epoch']
    BASE_LEARNING_RATE = hyperparameters['learning_rate']
    GPU_INDEX = general_opts['gpu']
    OPTIMIZER = train_opts['optimizer']
    MODEL = importlib.import_module(general_opts['model'])  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, 'models', general_opts['model'] + '.py')
    CURRENT_FILE = os.path.realpath(__file__)

    LOG_DIR = general_opts['log_dir'] + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(general_opts) + '\n')
    LOG_FOUT.write(str(train_opts) + '\n')
    LOG_FOUT.write(str(hyperparameters) + '\n')

    tf.set_random_seed(123456789)

    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
    os.system('cp %s %s' % (CURRENT_FILE, LOG_DIR))  # bkp of train procedure

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(40)
    BN_DECAY_CLIP = 0.99

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):

            with tf.name_scope('prepare_data'):
                tr_dataset = tf.data.TFRecordDataset(train_filenames).shuffle(1000000)
                tr_dataset = get_tfrecord_data(tr_dataset, batch_size=BATCH_SIZE, total_num_point=TOTAL_NUM_POINT)
                tr_iterator = tr_dataset.make_initializable_iterator()

                iter_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(iter_handle, tr_dataset.output_types,
                                                               tr_dataset.output_shapes)
                next_element = iterator.get_next()
                next_element = reshape_element(next_element, batch_size=BATCH_SIZE, total_num_point=TOTAL_NUM_POINT)

                obj_model, _ = read_and_decode_obj_model(object_model_dir)
                obj_model_tf = tf.convert_to_tensor(obj_model)

        with tf.device('/gpu:' + str(GPU_INDEX)):

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0.)

            bn_momentum = tf.train.exponential_decay(
                BN_INIT_DECAY,
                batch * BATCH_SIZE,
                BN_DECAY_DECAY_STEP,
                BN_DECAY_DECAY_RATE,
                staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            tf.summary.scalar('bn_decay', bn_decay)

            # the object models can be used for visualization and inspectation during training
            obj_batch = tf.gather(obj_model_tf, next_element['class_id'], axis=0)
            obj_batch = obj_batch[:, 0:1024, :]

            next_element_xyz = next_element['xyz'][:, 0:NUM_POINT, :]

            cls_gt_onehot = tf.one_hot(indices=next_element['class_id'], depth=len(target_cls))
            cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
            cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

            current_batch_axag = quat2axag(next_element['quaternion'])
            current_batch_axag = tf.dtypes.cast(current_batch_axag, dtype=tf.float64)

            xyz_graph_input = next_element_xyz
            trans_gt_graph_input = next_element['translation']
            rot_gt_graph_input = tf.cast(current_batch_axag, tf.float64)

            with tf.name_scope('6d_pose'):
                element_mean = tf.reduce_mean(xyz_graph_input, axis=1)

                xyz_normalized = xyz_graph_input - tf.expand_dims(element_mean, 1)

                trans_pred_res, _ = MODEL.get_trans_model(tf.concat([xyz_normalized, cls_gt_onehot_tile], axis=2),
                                               is_training_pl, bn_decay=bn_decay)
                trans_pred = trans_pred_res + element_mean
                rot_pred, _ = MODEL.get_rot_model(tf.concat([xyz_graph_input, cls_gt_onehot_tile], axis=2),
                                               is_training_pl, bn_decay=bn_decay)

            with tf.name_scope('translation'):

                trans_loss, trans_loss_perSample = trans_distance.get_translation_error(trans_pred,
                                                                                        trans_gt_graph_input)
                mean_dist_loss, mean_dist_loss_perSample = trans_distance.get_translation_error(element_mean,
                                                                                                trans_gt_graph_input)
                tf.summary.scalar('trans_loss', trans_loss)
                tf.summary.scalar('mean_dist_loss', mean_dist_loss)
                tf.summary.scalar('trans_loss_min', tf.reduce_min(trans_loss_perSample))
                tf.summary.scalar('trans_loss_max', tf.reduce_max(trans_loss_perSample))

            with tf.name_scope('rotation'):

                rot_pred = tf.cast(rot_pred, tf.float64)

                axag_loss, axag_loss_perSample = angular_distance_taylor.get_rotation_error(rot_pred,
                                                                                            rot_gt_graph_input)
                axag_loss = tf.cast(axag_loss, tf.float32)
                tf.summary.scalar('axag_loss', axag_loss)
                tf.summary.scalar('axag_loss_min', tf.reduce_min(axag_loss_perSample))
                tf.summary.scalar('axag_loss_max', tf.reduce_max(axag_loss_perSample))

            learning_rate = BASE_LEARNING_RATE

            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'gd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate*10)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            total_loss = 10 * trans_loss + axag_loss
            tf.summary.scalar('total_loss', total_loss)

            # reference: http://matpalm.com/blog/viz_gradient_norms/
            gradients = optimizer.compute_gradients(loss=total_loss)
            train_op = optimizer.apply_gradients(gradients, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=None)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})

        ops = {'is_training_pl': is_training_pl,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'trans_loss': trans_loss,
               'trans_loss_perSample': trans_loss_perSample,
               'mean_dist_loss_perSample': mean_dist_loss_perSample,
               'axag_loss': axag_loss,
               'axag_loss_perSample': axag_loss_perSample,
               'class_id': next_element['class_id'],
               'handle': iter_handle}

        count_perClass = np.zeros([NUM_CLASS], dtype=np.int32)
        axag_loss_perClass = [[] for _ in range(NUM_CLASS)]
        trans_loss_perClass = [[] for _ in range(NUM_CLASS)]
        mean_dist_loss_perClass = [[] for _ in range(NUM_CLASS)]

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch), dir=LOG_FOUT)
            sys.stdout.flush()

            training_handle = sess.run(tr_iterator.string_handle())

            sess.run(tr_iterator.initializer)

            train_graph(sess, ops, train_writer, training_handle, epoch,
                        count_perClass, axag_loss_perClass, trans_loss_perClass, mean_dist_loss_perClass,
                        logFOut=LOG_FOUT, batch_size=BATCH_SIZE, logdir=LOG_DIR, saver=saver)


def train_graph(sess, ops, train_writer, training_handle, epoch, count_perClass,
                axag_loss_perClass, trans_loss_perClass, mean_dist_loss_perClass, logFOut, batch_size, logdir, saver):
    """ ops: dict mapping from string to tf ops """
    print "==================train======================"
    is_training = True

    # measure duration of each subprocess
    start_time = datetime.now()
    summary1 = tf.Summary()

    batch_idx = 0

    while True:
        try:
            total_seen = 0

            # evaluate ever 100 batch during training
            if batch_idx != 0 and batch_idx % 2000 == 0:
                model_dir = "model.ckpt"
                save_path = saver.save(sess, os.path.join(logdir, model_dir))
                print "Model saved in file: %s" % save_path

            feed_dict = {ops['is_training_pl']: is_training, ops['handle']: training_handle}

            summary, step, _, class_id, trans_loss, trans_loss_perSample, \
            mean_dist_loss_perSample, axag_loss, axag_loss_perSample = sess.run([ops['merged'],
                                                                                 ops['step'],
                                                                                 ops['train_op'],
                                                                                 ops['class_id'],
                                                                                 ops['trans_loss'],
                                                                                 ops['trans_loss_perSample'],
                                                                                 ops['mean_dist_loss_perSample'],
                                                                                 ops['axag_loss'],
                                                                                 ops['axag_loss_perSample'],
                                                                                 ],
                                                                                feed_dict=feed_dict)

            # for each sample in current batch
            for x, y, z, c in zip(axag_loss_perSample, trans_loss_perSample, mean_dist_loss_perSample, class_id):
                axag_loss_perClass[c].append(x)
                trans_loss_perClass[c].append(y)
                mean_dist_loss_perClass[c].append(z)

            print("epoch %d batch %d trans_loss %f axag_loss %f" \
                  % (epoch, batch_idx, trans_loss, axag_loss))

            # write to tensorboard
            if batch_idx != 0 and batch_idx % 500 == 0:
                for i in target_cls:
                    count_perClass[i] = count_perClass[i] + len(axag_loss_perClass[i])
                    avg_axag_loss = np.average(axag_loss_perClass[i])
                    avg_trans_loss = np.average(trans_loss_perClass[i])
                    avg_mean_dist_loss = np.average(mean_dist_loss_perClass[i])
                    summary1.value.add(tag="axag_loss_per_class_train/"+class_names[i], simple_value=avg_axag_loss)
                    summary1.value.add(tag="trans_loss_per_class_train/"+class_names[i], simple_value=avg_trans_loss)
                    summary1.value.add(tag="mean_dist_loss_per_class_train/"+class_names[i], simple_value=avg_mean_dist_loss)
                    summary1.value.add(tag="sample_count_per_class_train/"+class_names[i], simple_value=count_perClass[i])
                    axag_loss_perClass[i] = []
                    trans_loss_perClass[i] = []
                    mean_dist_loss_perClass[i] = []

                train_writer.add_summary(summary, step)
                train_writer.add_summary(summary1, step)

            total_seen += batch_size
            batch_idx = batch_idx + 1

        except tf.errors.OutOfRangeError:
            print("End of data!")
            model_dir = "model.ckpt"
            save_path = saver.save(sess, os.path.join(logdir, model_dir))
            print("Model saved in file: %s" % save_path)
            break

    time_elapsed = datetime.now() - start_time
    out_str = 'Current epoch Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed)
    logFOut.write(out_str + '\n')
    logFOut.flush()
    print(out_str)


def get_training_argparser():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group('general')
    general.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    general.add_argument('--model', default='pcpe_net',
                         help='Model name: name of network model [default: pcpe_net]')
    general.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    general.add_argument('--num_point', type=int, default=256, help='Point Number [256/512/1024] [default: 256]')
    general.add_argument('--total_num_point', type=int, default=1024, help='Dataset Point Number [256/512/1024] [default: 1024]')

    train_opts = parser.add_argument_group('training_options')
    train_opts.add_argument('--max_epoch', type=int, default=90, help='Epoch to run [default: 90]')
    train_opts.add_argument('--optimizer', default='adam', help='adam or gd [default: adam]')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--batch_size', type=int, default=128,
                                 help='Batch Size during training [default: 128]')
    hyperparameters.add_argument('--learning_rate', type=float, default=0.0008,
                                 help='Initial learning rate [default: 0.0008]')
    hyperparameters.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    hyperparameters.add_argument('--decay_step', type=int, default=30000,
                                 help='Decay step for lr decay [default: 30000]')
    hyperparameters.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser


def parse_arg_groups(parser):
    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    return arg_groups


if __name__ == "__main__":
    parser = get_training_argparser()
    arg_groups = parse_arg_groups(parser)
    general_opts, train_opts, hyperparameters = arg_groups['general'], arg_groups['training_options'], arg_groups[
        'hyperparameters']
    setup_graph(general_opts=general_opts,
                train_opts=train_opts,
                hyperparameters=hyperparameters)
