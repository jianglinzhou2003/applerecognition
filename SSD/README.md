[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
# A port of [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd) to [Keras](https://keras.io) framework.
For more details, please refer to [arXiv paper](http://arxiv.org/abs/1512.02325).
For forward pass for 300x300 model, please, follow `SSD.ipynb` for examples. For training procedure for 300x300 model, please, follow `SSD_training.ipynb` for examples. Moreover, in `testing_utils` folder there is a useful script to test `SSD` on video or on camera input.

The previous text was written by the original author, now you may edit `SSD_training_main_new.py` for training and `SSD_main_new.py` for predicting. Moreover, check \PASCAL_VOC and `XML_to_pkl.py` and `get_data_from_XML.py` is added their by us to help you with data format conversion.

Weights are ported from the original models and are available [here](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA). You need `weights_SSD300.hdf5`, `weights_300x300_old.hdf5` is for the old version of architecture with 3x3 convolution for `pool6`.

The original code was tested with `Keras` v1.2.2, `Tensorflow` v1.0.0, `OpenCV` v3.1.0-dev

For the new requirements, see `Russell_Requirements.txt`. It is recommend to use conda.

We added `CheckImage.py` to help you check if any image is bad if the training process broke down.
