import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_process_tools
import importlib
import os
import sys
import argparse
import open3d
import transforms3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

DATA_PATH = 'data'
object_model_dir = "object_model_tfrecord/obj_models.tfrecords"
color_file = os.path.join(DATA_PATH, 'color.png')
depth_file = os.path.join(DATA_PATH, 'depth.png')
label_file = os.path.join(DATA_PATH, 'label.png')

total_target_cls = np.arange(21) # total number of classes
target_cls_choosen = 9 # target class, 9 for banana

# For data pre-processing
num_points_per_sample_FPS = 1024 # number of points after Furthest Point Sampling
threshold_distance_per_class = 0.2 * np.ones((len(total_target_cls),), dtype=np.float32) # box filtering threshold
class_vector = np.zeros(21) # class one hot vector
class_vector[target_cls_choosen] = 1  # banana
class_vector[14] = 1 # 14 for power drill

b_visual = True


def create_dataset(num_points_per_sample_FPS, threshold_distance_per_class):
    ds = tf.data.Dataset.from_tensors(
        {"color_file": tf.convert_to_tensor(color_file),
         "depth_file": tf.convert_to_tensor(depth_file),
         "label_file": tf.convert_to_tensor(label_file),
         'fx': tf.convert_to_tensor(1066.8, dtype=tf.float32),
         'fy': tf.convert_to_tensor(1067.5, dtype=tf.float32),
         'cx': tf.convert_to_tensor(313.0, dtype=tf.float32),
         'cy': tf.convert_to_tensor(241.3, dtype=tf.float32),
         'factor_depth': tf.convert_to_tensor(10000, dtype=tf.float32),
         'class_one_hot': tf.convert_to_tensor(class_vector, dtype=tf.int16)
         })
    ds = ds.flat_map(data_process_tools.read_data)
    ds = ds.flat_map(data_process_tools.split_samples)
    ds = ds.map(lambda x: data_process_tools.segment_filter(x, threshold_distance_per_class))
    ds = ds.filter(lambda x: tf.equal(x["class_id"], total_target_cls[target_cls_choosen]))  # only take target cls segment
    ds = ds.map(lambda x: data_process_tools.segment_sample_FPS(x, num_points_per_sample_FPS, threshold_distance_per_class))
    return ds


def reshape_element(element, batch_size, num_point):
    element['xyz'] = tf.reshape(element['xyz'], [batch_size, num_point, 3])
    element['rgb'] = tf.reshape(element['rgb'], [batch_size, num_point, 3])
    element['hsv'] = tf.reshape(element['hsv'], [batch_size, num_point, 3])
    element['class_id'] = tf.reshape(element['class_id'], [batch_size])
    element['num_valid_points_in_segment'] = tf.reshape(element['num_valid_points_in_segment'], [batch_size])

    return element


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


def setup_graph(general_opts, hyperparameters):
    tf.reset_default_graph()
    tf.set_random_seed(123456789)

    BATCH_SIZE = hyperparameters['batch_size']
    NUM_POINT = general_opts['num_point']
    GPU_INDEX = general_opts['gpu']
    MODEL = importlib.import_module(general_opts['model'])
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(40)
    BN_DECAY_CLIP = 0.99

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            dataset = create_dataset(num_points_per_sample_FPS, threshold_distance_per_class)
            dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(1)
            ds_iterator = dataset.make_initializable_iterator()
            iter_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
            iterator = tf.data.Iterator.from_string_handle(iter_handle, dataset.output_types, dataset.output_shapes)
            next_element = iterator.get_next()
            next_element = reshape_element(next_element, batch_size=BATCH_SIZE, num_point=num_points_per_sample_FPS)
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

            obj_batch = tf.gather(obj_model_tf, next_element['class_id'], axis=0)
            obj_batch = obj_batch[ 0:1024, :]

            next_element_xyz = next_element['xyz'][:, 0:NUM_POINT, :]
            next_element_rgb = next_element['rgb'][:, 0:NUM_POINT, :]

            element_mean = tf.reduce_mean(next_element['xyz'], axis=1)

            xyz_normalized = next_element_xyz - tf.expand_dims(element_mean, 1)

            cls_gt_onehot = tf.one_hot(indices=next_element['class_id'], depth=len(total_target_cls))
            cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
            cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

            with tf.name_scope('translation'):
                trans_pred_res, max_indices_trans = \
                    MODEL.get_trans_model(tf.concat([xyz_normalized, cls_gt_onehot_tile], axis=2),
                                          is_training_pl, bn_decay=bn_decay)

                trans_pred = trans_pred_res + element_mean

            xyz_remove_trans = next_element_xyz - tf.expand_dims(trans_pred, axis=1)

            with tf.name_scope('rotation'):
                rot_pred, global_feat_rot, _, out_weight, out_biases, max_indices = \
                    MODEL.get_rot_model(tf.concat([xyz_remove_trans, cls_gt_onehot_tile], axis=2),
                                        is_training_pl, bn_decay=bn_decay)
                rot_pred = tf.cast(rot_pred, tf.float64)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=None)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: False})

        # Restore variables from disk.
        trained_model = general_opts['trained_model']
        saver.restore(sess, trained_model)
        print "Model restored."

        ops = {'is_training_pl': is_training_pl,
               'step': batch,
               'xyz': next_element_xyz,
               'rgb': next_element_rgb,
               'obj_batch': obj_batch,
               'trans_pred': trans_pred,
               'rot_pred': rot_pred,
               'handle': iter_handle
               }

        test_handle = sess.run(ds_iterator.string_handle())
        sess.run(ds_iterator.initializer)
        eval_graph(sess, ops, test_handle)


def eval_graph(sess, ops, test_handle):
    is_training = False
    while True:
        try:
            feed_dict = {ops['is_training_pl']: is_training,
                         ops['handle']: test_handle}

            # trans_pred and rot_pred are estimation results
            trans_pred, rot_pred, xyz, rgb, obj_batch = sess.run([ops['trans_pred'],
                                                                  ops['rot_pred'],
                                                                  ops['xyz'],
                                                                  ops['rgb'],
                                                                  ops['obj_batch']],
                                                                 feed_dict=feed_dict)
            print "translation prediction ", trans_pred
            print "rotation prediction ", rot_pred

            # Visualize pose alignment
            if b_visual:
                batch_sample_idx = 0
                current_rot = rot_pred[batch_sample_idx]
                current_ag = np.linalg.norm(current_rot, ord=2)
                current_ax = current_rot / current_ag
                rotmat = transforms3d.axangles.axangle2mat(current_ax, current_ag)
                xyz_remove_rot = np.dot(xyz[batch_sample_idx,:,:], rotmat)
                xyz_remove_trans = xyz_remove_rot - np.dot(rotmat.T, trans_pred[batch_sample_idx,:])

                segment_ptCloud = open3d.PointCloud()
                segment_ptCloud.points = open3d.Vector3dVector(xyz_remove_trans)
                segment_ptCloud.colors = open3d.Vector3dVector(rgb[batch_sample_idx,:,:])

                model_pCloud = open3d.PointCloud()
                model_pCloud.points = open3d.Vector3dVector(obj_batch[batch_sample_idx, :, 0:3])
                model_pCloud.colors = open3d.Vector3dVector(obj_batch[batch_sample_idx, :, 3:6])
                model_pCloud.paint_uniform_color([0.1, 0.9, 0.1])

                model_frame = open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0])
                open3d.draw_geometries([model_pCloud, segment_ptCloud, model_frame])

        except tf.errors.OutOfRangeError:
            print('End of data!')
            break


def get_training_argparser():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group('general')
    general.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    general.add_argument('--model', default='pointnet_ycb_18',
                         help='Model name: pointnet_ycb_ [default: pointnet_ycb_18]')
    general.add_argument('--num_point', type=int, default=256, help='Point Number [256/512/1024/2048] [default: 256]')
    general.add_argument('--trained_model', help='Path to trained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--batch_size', type=int, default=128, help='Batch Size [default: 128]')

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
    general_opts, hyperparameters = arg_groups['general'], arg_groups['hyperparameters']
    setup_graph(general_opts=general_opts,
                hyperparameters=hyperparameters)
