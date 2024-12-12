#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import cv2
import keras
import matplotlib
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


# In[2]:


# voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
#                'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
#                'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
#                'Sheep', 'Sofa', 'Train', 'Tvmonitor']

voc_classes = ['apple']
NUM_CLASSES = len(voc_classes) + 1


# In[3]:


input_shape = (300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
# model.load_weights('weights_SSD300.hdf5', by_name=True)
model.load_weights('checkpoints/weights.09-3.48.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)


# In[4]:

inputs = []
images = []
mypaths = []
for root, dirs, files in os.walk("E:\\Attachment\\Attachment\\Attachment 1"):
    for file in files:
        file_path = os.path.join(root, file)
        # 在此处添加对文件的处理逻辑
        if (file_path[-3:]=='jpg'):
            print(file_path)
            img_path = file_path
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            images.append(imread(img_path))
            inputs.append(img.copy())
            mypaths.append(img_path)
inputs = preprocess_input(np.array(inputs))


# In[5]:


preds = model.predict(inputs, batch_size=1, verbose=1)



# In[6]:


results = bbox_util.detection_out(preds)


# In[7]:


# get_ipython().run_cell_magic('time', '', 'a = model.predict(inputs, batch_size=1)\nb = bbox_util.detection_out(preds)\n')


# In[8]:


for i, img in enumerate(images):
    thispath = mypaths[i]
    # Parse the outputs.
    # with open(thispath, 'w') as file:
    #     file.write(str(results[i].tolist()))
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.8]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{} {:0.2f}'.format("Apple",score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=5))
        currentAxis.text(xmin, ymin, display_txt, fontsize = 18, bbox={'facecolor':color, 'alpha':0.5})
    plt.savefig('E:\\SSD_pltfig\\{}.png'.format(i))
    plt.show()