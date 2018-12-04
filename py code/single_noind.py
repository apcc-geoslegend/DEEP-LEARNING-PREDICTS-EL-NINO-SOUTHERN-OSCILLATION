Last login: Tue Nov 20 14:02:45 on ttys005
jiangyiningdeMacBook-Pro:~ jiangyining$ ssh -p 3333 hxm_stu@166.111.5.236
The authenticity of host '[166.111.5.236]:3333 ([166.111.5.236]:3333)' can't be established.
RSA key fingerprint is SHA256:om2vRxCA10L08u2yuOa6cP8vhZKBozzyLBad2YocXds.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added '[166.111.5.236]:3333' (RSA) to the list of known hosts.
hxm_stu@166.111.5.236's password: 
Connection closed by 166.111.5.236 port 3333
jiangyiningdeMacBook-Pro:~ jiangyining$ ll
-bash: ll: command not found
jiangyiningdeMacBook-Pro:~ jiangyining$ ssh -p 3333 hxm_stu@166.111.5.236
hxm_stu@166.111.5.236's password: 
Last login: Tue Nov 20 14:47:26 2018 from 153.35.171.100

=========================================================

Pls. do not run jobs on manager node.
Pls. run NCL and MATLAB on b1n41-43, thanks!

job script samples:

QUEUE hpca: each node has 20 cores, apply 120 core, use
#PBS -l nodes=6:ppn=20

QUEUE hpcb: each node has 28 cores, apply 140 cores, use
#PBS -l nodes=5:ppn=28

=========================================================

[hxm_stu@manager ~]$ ll
total 6076880
-rw-r--r--  1 hxm_stu hxm       60000 Nov 19 21:55 2007_test.txt
-rw-r--r--  1 hxm_stu hxm      118833 Nov 19 21:55 2007_train.txt
-rw-r--r--  1 hxm_stu hxm       60000 Nov 19 21:55 2007_val.txt
-rw-r--r--  1 hxm_stu hxm        3431 May  5  2018 a.c
-rwxr-x---  1 hxm_stu hxm    18972051 May  8  2015 acs.tar.gz
-rw-r--r--  1 hxm_stu hxm    14954536 Apr 23  2018 all.csv.zip
-rw-r--r--  1 hxm_stu hxm    35440391 Apr 18  2018 annotations.tar.gz
-rwxr-x---  1 hxm_stu hxm        7260 Jun 18  2015 a.out
-rw-r--r--  1 hxm_stu hxm        4844 Aug 30 18:14 ap01c.c
-rw-r--r--  1 hxm_stu hxm        1093 Aug 30 18:14 ap01.res
-rw-r--r--  1 hxm_stu hxm     4166553 Feb 26  2016 astrill-setup-linux64.rpm
-rw-r--r--  1 hxm_stu hxm         394 Jul 21  2016 authorized_keys
-rw-r--r--  1 hxm_stu hxm         143 Apr 18  2018 buildings_AOI_4_Shanghai_img5669.geojson.prj
lrwxrwxrwx  1 root    root         34 Jul 23 22:19 ccmp -> /home/xfh/WORK4/atop_ccmp/R05/out/
drwxr-x---  4 hxm_stu hxm        4096 Jun 24  2015 cesm
-rw-r--r--  1 hxm_stu hxm          60 Jun  6 03:22 cnt.sh
-rwxr-x---  1 hxm_stu hxm       50495 Dec 14  2015 conLab.png
drwxr-xr-x  9 hxm_stu hxm        4096 Nov 17 19:59 darknet
-rw-r--r--  1 hxm_stu hxm   162482580 Nov 20 11:39 darknet53.conv.74
-rw-r--r--  1 hxm_stu hxm    16793600 Sep 24  2016 DCIM.tar
-rw-r--r--  1 hxm_stu hxm        5598 Feb 20  2016 desktop
drwxr-x---  3 hxm_stu hxm        4096 Jul 15  2016 Desktop
-rw-rw-r--  1 hxm_stu hxm       63544 Mar 22  2016 dxdy0.txt
-rw-rw-r--  1 hxm_stu hxm       63544 Mar 22  2016 dxdy1.txt
-rw-rw-r--  1 hxm_stu hxm       63544 Mar 22  2016 dxdy2.txt
-rw-rw-r--  1 hxm_stu hxm       63544 Mar 22  2016 dxdy3.txt
drwxr-xr-x  3 hxm_stu hxm        4096 Apr 25  2018 ecmwf-api-client-python
-rw-r--r--  1 root    root  233410504 Jul 23 22:19 ETOPO2v2g_f4.nc
drwxr-x---  3 hxm_stu hxm        4096 Jun  3  2015 exe
-rw-rw-r--  1 hxm_stu hxm    85024960 Mar 23  2016 exp001_fortran.zip
-rw-rw-r--  1 hxm_stu hxm      948521 Mar 28  2016 exp00x_cuda_GMD_submit.tgz
drwxr-xr-x  3 hxm_stu hxm        4096 Apr  8  2016 ez_setup-0.9
-rw-rw-r--  1 hxm_stu hxm        6577 Jan  8  2016 ez_setup-0.9.tar.gz
-rw-r--r--  1 hxm_stu hxm    24145996 Apr 25  2018 fangjr@muradin.cs.ucdavis.edu
drwxr-xr-x  2 hxm_stu hxm        4096 Sep  3 20:07 fjr
-rw-r--r--  1 hxm_stu hxm     5734230 Apr 24  2018 fjrparis.csv
-rw-r--r--  1 hxm_stu hxm    24145996 Apr 25  2018 fjrshanghia.csv
-rw-r--r--  1 root    root     227697 Jul 23 17:52 grid.f
-rw-r--r--  1 root    root  353271544 Jul 23 17:53 grid.nc
drwxr-xr-x  2 hxm_stu hxm        4096 Nov 17 19:55 home
drwxrwxr-x  4 hxm_stu hxm        4096 Oct  9  2016 hruan
drwxr-x---  9 hxm_stu hxm        4096 Oct  6  2016 hy
-rwxr-x---  1 hxm_stu hxm        1679 Jul 16  2015 id_rsa
-rw-r--r--  1 hxm_stu hxm         390 Apr 24  2018 id_rsa.pub
-rwxr-xr-x  1 hxm_stu hxm  1479949950 May 15  2018 IMG_3242.mov
drwxr-x---  7 hxm_stu hxm        4096 Jul  5 13:56 intel
drwxr-x---  3 hxm_stu hxm        4096 Apr 22  2015 Library
-rwxr-x---  1 hxm_stu hxm      143148 May 13  2015 log
-rw-rw-r--  1 hxm_stu hxm         637 Mar 16  2016 log_print_hostname.txt
-rw-r--r--  1 hxm_stu hxm           0 Dec  1  2017 log.txt
drwxr-xr-x  9 hxm_stu hxm        4096 Oct 25 19:30 lwj
drwxr-xr-x  4 hxm_stu hxm        4096 Nov 11 23:49 maocai
-rw-rw-r--  1 hxm_stu hxm       13801 May 17  2016 matlab_crash_dump.11541-1
-rw-rw-r--  1 hxm_stu hxm           0 May 16  2016 matlab_crash_dump.23466-1
-rw-rw-r--  1 hxm_stu hxm       12441 May 18  2016 matlab_crash_dump.2992-1
drwxr-x---  2 hxm_stu hxm        4096 May 18  2016 netcdf-test
-rw-r--r--  1 hxm_stu hxm     7157389 Dec  1  2017 nh_pom.tar.gz
-rw-------  1 hxm_stu hxm     4407514 May  3  2018 nohup.out
lrwxrwxrwx  1 hxm_stu hxm          10 Mar 17  2015 nyf -> WORK1/nyf/
drwxr-xr-x  2 hxm_stu hxm        4096 Dec  1  2017 output
-rwxr-x---  1 hxm_stu hxm         869 Dec  1  2017 pac10exp369.2008-09-01.sh
-rw-r--r--  1 hxm_stu hxm   314720189 Apr 17  2018 paris.imposm-shapefiles.zip
-rw-r--r--  1 hxm_stu hxm  1972770095 May  5  2018 pgilinux-2018-184-x86-64.tar.gz
drwxrwxr-x  2 hxm_stu hxm        4096 May  5  2018 pgitest
drwxr-xr-x  2 root    root       4096 Jul 23 22:19 pom
lrwxrwxrwx  1 hxm_stu hxm          18 May 11  2016 pom.nml -> pom.nml.2012-01-17
-rwxr-x---  1 hxm_stu hxm     1303544 Apr 12  2015 pop_src.tar.gz
-rwxr-x---  1 hxm_stu hxm       42674 May 21  2015 pp-1.6.4.zip
-rwxr-x---  1 hxm_stu hxm       15726 Aug  3  2015 prep.F
-rw-r--r--  1 hxm_stu hxm     6662844 May 30 22:46 protobuf-all-3.5.1.tar.gz
-rw-r--r--  1 hxm_stu hxm     4424543 Apr 23  2016 pyproj-1.9.5.1.tar.gz
drwxr-xr-x  6 hxm_stu hxm        4096 Apr  8  2016 python-2.7.11
drwxr-xr-x 18 hxm_stu hxm        4096 Apr  8  2016 Python-2.7.11
-rw-rw-r--  1 hxm_stu hxm    16856409 Dec  6  2015 Python-2.7.11.tgz
-rw-r--r--  1 hxm_stu hxm           0 Dec  1  2017 RANS-PROP.inp
-rw-r--r--  1 hxm_stu hxm    82291074 Apr 19  2018 rice@gpu
-rw-rw-r--  1 hxm_stu hxm         218 May 20  2016 run.sh
-rwxr-x---  1 hxm_stu hxm         895 Apr 25  2018 setup.py
lrwxrwxrwx  1 root    root         43 Jul 23 22:18 ssts -> /home/xfh/WORK4/grid/run/R05-3/outfile/ssts
lrwxrwxrwx  1 hxm_stu hxm          21 May 11  2016 switch.nml -> switch.nml.2012-01-17
drwxr-xr-x  3 hxm_stu hxm        4096 Dec 26  2016 sxzhang
-rwxr-x---  1 hxm_stu hxm         178 Apr 24  2016 tansuo.sh
-rwxr-x---  1 hxm_stu hxm         281 Jun 18  2015 test.cpp
-rw-rw-r--  1 hxm_stu hxm           0 Apr 12  2016 test_scp.txt
-rw-r--r--  1 hxm_stu hxm      150000 Sep 21 16:57 tmp
drwxrwxr-x  2 hxm_stu hxm        4096 Feb 13  2017 try
lrwxrwxrwx  1 root    root         45 Jul 23 22:19 tsclim -> /home/xfh/WORK4/grid/run/R05-3/outfile/tsclim
-rw-r--r--  1 hxm_stu hxm        3201 Feb 15  2016 tunet-cli-master.zip
drwxrwxr-x  3 hxm_stu hxm        4096 Nov 19 21:40 VOCdevkit
-rw-rw-r--  1 hxm_stu hxm  1103133764 Apr 15  2016 vorticity_850_1998-2010daily.nc
-rw-rw-r--  1 hxm_stu hxm         263 Dec 24  2015 vote.py
drwxr-xr-x  2 hxm_stu hxm        4096 Aug  9 13:09 wangmq
lrwxrwxrwx  1 hxm_stu hxm          25 Mar  6  2015 WORK1 -> /home/share/hxm_stu/work1
lrwxrwxrwx  1 hxm_stu hxm          26 Jan  7  2016 WORK2 -> /home/share2/hxm_stu/work2
lrwxrwxrwx  1 hxm_stu hxm          26 Jan  7  2016 WORK3 -> /home/share3/hxm_stu/work3
lrwxrwxrwx  1 hxm_stu hxm          26 May 15  2017 WORK4 -> /home/share4/hxm_stu/work4
drwxr-xr-x 11 hxm_stu hxm        4096 Nov  8 09:36 wwf
drwxr-x---  4 hxm_stu hxm        4096 Jun 24  2015 Yin-Yang_grid
-rw-r--r--  1 hxm_stu hxm   248007048 Nov 17 20:27 yolov3.weights
lrwxrwxrwx  1 hxm_stu hxm           8 Mar 13  2015 zc -> WORK1/zc
drwxr-xr-x  4 hxm_stu hxm       98304 Nov 20 13:10 zjp
[hxm_stu@manager ~]$ ssh cheny@10.0.0.188
cheny@10.0.0.188's password: 
Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-119-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

142 packages can be updated.
0 updates are security updates.

New release '18.04.1 LTS' available.
Run 'do-release-upgrade' to upgrade to it.

*** System restart required ***
Last login: Tue Nov 20 14:36:53 2018 from 10.0.0.10
lcheny@gpu:~$ ll
total 92
drwxr-xr-x 12 cheny cheny  4096 11月 20 14:21 ./
drwxr-xr-x 26 root  root   4096 11月 20 14:34 ../
-rw-------  1 cheny cheny 21646 11月 20 13:54 .bash_history
-rw-r--r--  1 cheny cheny   220 3月  30  2018 .bash_logout
-rw-r--r--  1 cheny cheny  3771 3月  30  2018 .bashrc
drwx------  5 cheny cheny   101 4月  13  2018 .cache/
drwxrwxr-x  2 cheny cheny    74 4月  13  2018 code/
drwxrwxr-x  3 cheny cheny    32 4月  13  2018 .config/
-rw-r--r--  1 cheny cheny  8980 3月  30  2018 examples.desktop
drwxrwxr-x  3 cheny cheny    27 11月  3 15:36 .java/
drwxrwxr-x  3 cheny cheny    52 5月   5  2018 .keras/
drwxrwxr-x  3 cheny cheny    28 11月  3 15:36 .matlab/
drwx------  3 cheny cheny    34 4月  14  2018 .nv/
-rw-r--r--  1 cheny cheny   675 3月  30  2018 .profile
-rw-------  1 cheny cheny    61 4月  23  2018 .python_history
drwx------  2 cheny cheny    33 4月  10  2018 .ssh/
drwxrwxr-x  6 cheny cheny   149 11月  7 15:23 test/
drwxr-xr-x  2 cheny cheny    32 11月  3 16:06 .vim/
-rw-------  1 cheny cheny 31428 11月 20 14:21 .viminfo
cheny@gpu:~$ cd test/session_test/
cheny@gpu:~/test/session_test$ l
code/
cheny@gpu:~/test/session_test$ ll
total 0
drwxrwxr-x 3 cheny cheny  26 10月 31 11:39 ./
drwxrwxr-x 6 cheny cheny 149 11月  7 15:23 ../
drwxrwxr-x 3 cheny cheny  26 10月 31 11:40 code/
cheny@gpu:~/test/session_test$ cd code/neno/
cheny@gpu:~/test/session_test/code/neno$ ll
total 248
drwxr-xr-x 6 cheny cheny  4096 11月 20 14:42 ./
drwxrwxr-x 3 cheny cheny    26 10月 31 11:40 ../
-rw-rw-r-- 1 cheny cheny  4911 11月 18 14:39 1
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train1.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train_multivar1.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train_multivar_all2.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train_multivar_one2.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train_onevar1.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train_onevar.py
-rw-r--r-- 1 cheny cheny     0 10月 31 12:21 4k_st_train.py
drwxr-xr-x 3 cheny cheny    36 11月  3 11:42 dataset/
drwxrwxr-x 8 cheny cheny   214 10月 31 12:21 .git/
-rw-rw-r-- 1 cheny cheny     0 11月  4 10:12 __init__.py
-rw-r--r-- 1 cheny cheny   282 11月  3 15:34 metrics.py
-rw-r--r-- 1 cheny cheny   594 11月  3 16:27 metrics.pyc
-rw-r--r-- 1 cheny cheny 13454 11月 18 15:45 model1.py
-rw-r--r-- 1 cheny cheny 11545 11月 19 00:14 model1.pyc
-rw-r--r-- 1 cheny cheny 13451 11月  3 16:11 model.py
-rw-r--r-- 1 cheny cheny 11677 11月  3 16:27 model.pyc
-rw-r--r-- 1 root  root  13453 11月 20 14:38 multi_gpu_model.py
-rw-r--r-- 1 root  root  16384 11月 20 14:42 .multi_gpu_model.py.swp
drwxr-xr-x 2 cheny cheny  4096 11月 18 15:47 __pycache__/
drwxrwxr-x 4 cheny cheny    49 10月 31 14:53 Result/
-rw-rw-r-- 1 cheny cheny   587 11月  3 15:33 rmse.py
-rw-rw-r-- 1 cheny cheny  1180 11月  3 16:27 rmse.pyc
-rw-r--r-- 1 cheny cheny  4910 11月 18 22:46 single_noind1.py
-rw-r--r-- 1 cheny cheny  4675 11月  8 13:49 single_noind2.py
-rw-r--r-- 1 cheny cheny  4912 11月 19 00:08 single_noind31.py
-rw-r--r-- 1 cheny cheny  4910 11月 19 00:09 single_noind32.py
-rw-r--r-- 1 cheny cheny  4912 11月 19 00:12 single_noind33.py
-rw-r--r-- 1 cheny cheny  4912 11月 19 00:12 single_noind34.py
-rw-r--r-- 1 cheny cheny  4912 11月 19 00:13 single_noind35.py
-rw-r--r-- 1 cheny cheny  4674 11月  9 13:29 single_noind36.py
-rw-r--r-- 1 cheny cheny  4784 11月  7 14:34 single_noind3.py
-rw-r--r-- 1 cheny cheny  4675 11月  8 13:53 single_noind4.py
-rw-r--r-- 1 cheny cheny  4782 11月  7 09:30 single_noind5.py
-rw-r--r-- 1 cheny cheny  4766 11月 19 15:21 single_noind6.py
-rw-r--r-- 1 cheny cheny  4757 11月 19 15:26 single_noind7.py
-rw-r--r-- 1 cheny cheny  4911 11月 18 14:39 single_noind.py
-rw-rw-r-- 1 cheny cheny  1314 11月 18 14:15 single_SST.py
-rw-r--r-- 1 cheny cheny  3857 11月  6 12:02 utils.py
-rw-r--r-- 1 cheny cheny  5162 11月  6 16:29 utils.pyc
cheny@gpu:~/test/session_test/code/neno$ vi single_noind.py 
cheny@gpu:~/test/session_test/code/neno$ vi model.py
cheny@gpu:~/test/session_test/code/neno$ vi model.py
cheny@gpu:~/test/session_test/code/neno$ vi utils.py
cheny@gpu:~/test/session_test/code/neno$ vi metrics.py
cheny@gpu:~/test/session_test/code/neno$ vi metrics.py
cheny@gpu:~/test/session_test/code/neno$ vi metrics.py
cheny@gpu:~/test/session_test/code/neno$ vi utils.py
cheny@gpu:~/test/session_test/code/neno$ vi metrics.py
cheny@gpu:~/test/session_test/code/neno$ vi single_noind.py 



# coding: utf-8

import tensorflow as tf
import scipy.io as sio
import keras
import numpy as np
import pandas as pd
from utils import *
from model1 import *
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
batch_size = 32
key = 0 #不考虑index因素
p1 = 0
p2 = 1
timestep = 1;

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

root_path = os.getcwd().replace("\\","/")
X_path = root_path + '/dataset/test_for_paper/SST_OISST_time_19820101-20161231.mat'
#X_path = root_path + '/dataset/test_for_paper/SST_OISST_region_19820101-20161231.mat'
#X_path = root_path + '/dataset/test_for_paper/SST_OISST_space_19820101-20161231.mat'
Y_path = root_path + '/dataset/test_for_paper/Nino3.4_OISST_19820101-20161231.mat'

X = load_data(X_path,'sst_daily'）###x第一维以天为单位共10000多，二三维为经纬度 
Y = load_data(Y_path, 'nino34_daily') ###10000多x1
varname = ["SST_daily","SST_pentad","SST_godas","SST_5d"]
#varname = ["SST_30p","SST_10p","SST_daily_05deg"]

for i in range(p1,p2):
    model_name = 'ResNet50_' + varname[i] + '_1deg_noind_model1'
    print(model_name)

    X, Y, scalar = preprocess_data(X, Y, split_ratio=0.8)
    X = np.asarray(X)
    print(X.shape)
    
    day = [90,180]
    for shift in day:
        lead = int(shift/timestep)
        train_X, extra_train_X, train_Y, test_X, extra_test_X, test_Y = create_st_dataset(X, Y, predict_day=lead, split_ratio=0.8)

        print("train X shape: {}".format(train_X.shape))
        print("extra train X shape: {}".format(extra_train_X.shape))
        print("train Y shape: {}".format(train_Y.shape))

        for it in range(1,11):
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
            if key == 0:
               history = model.fit(train_X,  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(test_X,test_Y), callbacks=[model_checkpoint])
            else:
               history = model.fit([train_X,extra_train_X],  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=([test_X, extra_test_X],test_Y), callbacks=[model_checkpoint])

            # set parameter
            model.save_weights(os.path.join('Result/model/', '4k_data_smooth{}.{}.final.best.h5'.format(hyperparams_name, model_name)), overwrite=True)

            print('=' * 10)
            if key == 0:
               score = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)
            else:
               score = model.evaluate([test_X,extra_test_X], test_Y, batch_size=batch_size, verbose=1)
            print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (scalar._max - scalar._min) / 2.))

            if key == 0:
               y_hat_norm = model.predict(test_X, batch_size=batch_size, verbose=0)
            else:
               y_hat_norm = model.predict([test_X,extra_test_X], batch_size=batch_size, verbose=0)
            y_hat = scalar.inverse_transform(y_hat_norm)
            y_label = scalar.inverse_transform(test_Y)

            # set parameter
            sio.savemat(os.path.join('Result/prediction/model_test/{}'.format(varname[i]), '4k{}.{}days_smooth_{}.mat'.format(model_name, shift, it)),{'y_lab':y_label, 'y_pre':y_hat} )
