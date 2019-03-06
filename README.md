# PCPE(Point Cloud Pose Estimation)
This is an tensorflow implementation of estimating object 6D pose from a point cloud segment

# Files
* **data**: a color, a depth and a label image for testing.
* **models**: a python file defining model structure.
* **object_model_tfrecord**: full object models for visualization purpose.
* **trained_network**: a trained network.
* **utils**: utility files for defining model structure.
* *data_process_tools.py*: tools for data pre-processing.
* *object_6d_pose.py*: main file for object 6d pose estimation.

# Requirements
* Tensorflow-GPU (tested with 1.12.0)
* [transforms3d](https://matthew-brett.github.io/transforms3d/)
* [open3d](http://www.open3d.org/docs/getting_started.html) for visualization

# Usage
After activate tensorflow
```
python object_6d_pose.py --trained_model trained_network/20190222-130143/model.ckpt --batch_size 1
```
Translation prediction is in unit meter.
Rotation prediction is in axis-angle format.