# coding: utf-8
import time
import tensorflow as tf
import scipy.io as sio
import keras
import numpy as np
import pandas as pd
from utils import *
from model import *
#from resnet_model import *
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
batch_size = 256
key = 0
p1 = 0
p2 = 1
timestep = 1;

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

root_path = os.getcwd().replace("\\","/")
#X_path = root_path + '/dataset/test_for_paper/SST_OISST_time_19820101-20161231.mat'
#X_path = root_path + '/dataset/test_for_paper/SST_OISST_region_19820101-20161231.mat'
X_path = root_path + '/dataset/test_for_paper/SST_OISST_space_19820101-20161231.mat'
Y_path = root_path + '/dataset/test_for_paper/Nino3.4_OISST_19820101-20161231.mat'

X = load_data(X_path,'sst_daily_2deg')
Y = load_data(Y_path, 'nino34_daily')
varname = ["SST_daily_2deg","SST_pentad","SST_godas","SST_5d"]
#varname = ["SST_30p","SST_10p","SST_daily_05deg"]

for i in range(p1,p2):
    model_name = 'ResNet50_' + varname[i] + '_1deg_noind_model0'
    print(model_name)

    X, Y, scalar = preprocess_data(X, Y, split_ratio=0.8)
    X = np.asarray(X)
    print(X.shape)
    day = [180]
    for shift in day:
        lead = int(shift/timestep)
        train_X, extra_train_X, train_Y, test_X, extra_test_X, test_Y = create_st_dataset(X, Y, predict_day=lead, split_ratio=0.8)

        print("train X shape: {}".format(train_X.shape))
        print("extra train X shape: {}".format(extra_train_X.shape))
        print("train Y shape: {}".format(train_Y.shape))

        for it in range(1,11):
            time_start=time.time()
            if key == 0:
               model = ResNet(train_X[0].shape, None)
            else:
               model = ResNet(train_X[0].shape, extra_train_X[0].shape)

            adam = Adam(lr)
            model.compile(optimizer=adam, loss='mse', metrics=[rmse.rmse])

            model.summary()

            hyperparams_name = 'model_name{}.predict_day{}.lr{}.time{}'.format(model_name, shift, lr, it)
            # set parameter
            fname_param = os.path.join('Result/model/', '4k_data_smooth{}.best.h5'.format(hyperparams_name))
            early_stopping = EarlyStopping(monitor='val_root_mean_square_error', patience=10, mode='min')
            model_checkpoint = ModelCheckpoint(fname_param, monitor='val_root_mean_square_error', verbose=0, save_best_only=True, mode='min')
            print('=' * 50)

            #model.load_weights('./MODEL/model_nameResNet50.predict_day30.lr0.0003.best.h5')
            if os.path.exists(fname_param):
               model.load_weights(fname_param)
            print("Training model...")
            if key == 0:
               history = model.fit(train_X,  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(test_X,test_Y), callbacks=[model_checkpoint])
            else:
               history = model.fit([train_X,extra_train_X],  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=([test_X, extra_test_X],test_Y), callbacks=[model_checkpoint])

            # set parameter
            #model.save_weights(os.path.join('Result/model/', '4k_data_smooth{}.{}.final.best.h5'.format(hyperparams_name, model_name)), overwrite=True)

            print('=' * 10)
            if key == 0:
               score = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)
            else:
               score = model.evaluate([test_X,extra_test_X], test_Y, batch_size=batch_size, verbose=1)
            print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (scalar._max - scalar._min) / 2.))
            time_end=time.time()
            print('time cost',time_end-time_start,'s')
            if key == 0:
               y_hat_norm = model.predict(test_X, batch_size=batch_size, verbose=0)
            else:
               y_hat_norm = model.predict([test_X,extra_test_X], batch_size=batch_size, verbose=0)
            y_hat = scalar.inverse_transform(y_hat_norm)
            y_label = scalar.inverse_transform(test_Y)

            # set parameter
            sio.savemat(os.path.join('Result/prediction/model_test/{}'.format(varname[i]), '4k{}.{}days_smooth_{}.mat'.format(model_name, shift, it)),{'y_lab':y_label, 'y_pre':y_hat} )
                                                    