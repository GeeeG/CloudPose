import os
import random
import imageio
import numpy as np
from scipy.io import loadmat
import transforms3d
import tensorflow as tf
import sys

# Global settings
num_classes = 21
num_sequences = 92
YCB_PATH = "/data_c/YCB_Video_Dataset/" # change this to your path to YCB Video Dataset
OUT_PATH = "../sort_by_seq/"

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_sequence(seq_dir, seq_id, output_file):
	files = os.listdir(seq_dir)
	num_frames = len(files) / 5

	frame_id_list = list(range(1,num_frames+1))
	random.shuffle(frame_id_list)
	with tf.python_io.TFRecordWriter(output_file) as writer:
		for frame_id in frame_id_list:

			image_name = os.path.join(seq_dir, "{:06d}-color.png".format(frame_id))
			image = np.asarray(imageio.imread(image_name))
			image_shape = np.array(image.shape, np.int64)

			depth_name = os.path.join(seq_dir, "{:06d}-depth.png".format(frame_id))
			depth = np.asarray(imageio.imread(depth_name))
			depth_shape = np.array(depth.shape, np.int64)

			label_name = os.path.join(seq_dir, "{:06d}-label.png".format(frame_id))
			label = np.asarray(imageio.imread(label_name))
			label_shape = np.array(label.shape, np.int64)

			meta_name = os.path.join(seq_dir, "{:06d}-meta.mat".format(frame_id))
			meta = loadmat(meta_name)

			fx = np.array(meta["intrinsic_matrix"][0,0])
			fy = np.array(meta["intrinsic_matrix"][1,1])
			cx = np.array(meta["intrinsic_matrix"][0,2])
			cy = np.array(meta["intrinsic_matrix"][1,2])
			factor_depth = np.array(meta["factor_depth"])

			class_ids = np.array(meta["cls_indexes"]).flatten().astype(np.int64) - 1 # convert to zero-based indexing

			class_one_hot = np.zeros((num_classes,), dtype=np.int64)
			quaternions = np.zeros((num_classes,4), dtype=np.float32)
			translations = np.zeros((num_classes,3), dtype=np.float32)

			poses = np.array(meta["poses"])
			poses = np.transpose(poses, [2, 0, 1]) # swap class dimension to be first
			for class_id, pose in zip(class_ids, poses):
				rotation_matrix = pose[:3,:3]
				quaternions[class_id,:] = transforms3d.quaternions.mat2quat(rotation_matrix)
				translations[class_id,:] = pose[:,3]
				class_one_hot[class_id] = 1

			data_dict = {
			  'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
			  'depth': _bytes_feature(tf.compat.as_bytes(depth.tostring())),
			  'label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
			  'quaternions': _float_feature(quaternions.reshape(-1)),
			  'translations': _float_feature(translations.reshape(-1)),
			  'class_one_hot': _int64_feature(class_one_hot.reshape(-1)),
			  'seq_id': _int64_feature([seq_id]),
			  'frame_id': _int64_feature([frame_id]),
			  'fx': _float_feature([fx]),
			  'fy': _float_feature([fy]),
			  'cx': _float_feature([cx]),
			  'cy': _float_feature([cy]),
			  'factor_depth': _float_feature([factor_depth]),
			  'image_shape': _int64_feature(image_shape),
			  'depth_shape': _int64_feature(depth_shape),
			  'label_shape': _int64_feature(label_shape)
			}
			data = tf.train.Features(feature=data_dict)
			example = tf.train.Example(features=data)

			writer.write(example.SerializeToString())


for seq_id in range(num_sequences):
	seq_dir = os.path.join(YCB_PATH, "data/{:04d}".format(seq_id))
	out_file = os.path.join(OUT_PATH, "{:04d}.tfrecord".format(seq_id))

	print('Writing sequence {:04d} to file {:s}'.format(seq_id,out_file))
	write_sequence(seq_dir, seq_id, out_file)

seq_dir = os.path.join(YCB_PATH, "data_syn")
out_file = os.path.join(OUT_PATH, "synthetic.tfrecord")
print('Writing sequence data_syn to file {:s}'.format(out_file))
write_sequence(seq_dir, -1, out_file)

