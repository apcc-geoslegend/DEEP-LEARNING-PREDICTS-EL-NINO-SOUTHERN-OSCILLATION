# coding: utf-8

import tensorflow as tf
import scipy.io as sio
import keras
import numpy as np
import pandas as pd
from utils import *
from model import *
from resnet_model import *
from keras import backend as K
import os
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend.tensorflow_backend as KTF
import  rmse as rmse
from keras.optimizers import Adam
import sys
from metrics import *


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
config.allow_soft_placement = True
config.log_device_placement = True
sess = tf.Session(config=config)

# 设置session
KTF.set_session(sess)

path_result = 'Result/prediction'
path_model = 'Result/model'
#model_name = 'ResNet50_Multi_5d_2016'
#shift = int(sys.argv[2])
lr = 3e-5
epochs = 50  # number of epoch at training (cont) stage
batch_size = 32
key = 0
p1 = 3
p2 = 4
timestep = 1;

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

root_path = os.getcwd().replace("\\","/")
X_path = root_path + '/dataset/test_for_paper/SST_OISST_time_19820101-20161231.mat'
Y_path = root_path + '/dataset/test_for_paper/Nino3.4_OISST_19820101-20161231.mat'