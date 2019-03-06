# PCPE(Point Cloud Pose Estimation)
This is an tensorflow implementation of estimating object 6D pose from a point cloud segment

# Files
1. **data**: a color, a depth and a label image for testing.
2. **models**: a python file defining model structure.
3. **object_model_tfrecord**: full object models for visualization purpose.
4. **trained_network**: a trained network.
5. **utils**: utility files for defining model structure.
6. *data_process_tools.py*: tools for data pre-processing.
7. *object_6d_pose.py*: main file for object 6d pose estimation.

# Requirements
* Tensorflow-GPU (tested with 1.12.0)
* [transforms3d](https://matthew-brett.github.io/transforms3d/)
* [open3d](http://www.open3d.org/docs/getting_started.html) for visualization

# Test a trained network
1. Available classes in the test scenario are: 
 * 1: 003_cracker_box
 * 3: 005_tomato_soup_can
 * 4: 006_mustard_bottle
 * 9: 011_banana
 * 14: 035_power_drill
2. After activate tensorflow
```
python object_6d_pose.py --trained_model trained_network/20190222-130143/model.ckpt --batch_size 1 --target_class 9
```
Translation prediction is in unit meter.
Rotation prediction is in axis-angle format.

# Train your own network