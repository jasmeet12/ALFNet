from __future__ import division
import os
from keras_alfnet import config
from keras_alfnet.model.model_2step import Model_2step
import cv2

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# define paths for weight files and detection results
w_path = 'data/models/city_res50_2step.hdf5'
data_path = 'data/examples/demo'
# C.random_crop = (1024, 2048)
C.random_crop = (512,1024) #for crowd human
C.network = 'resnet50'
# define the ALFNet network
model = Model_2step()
model.initialize(C)

# foldername = ['snow', 'Fog', 'Rain', 'snowLand', 'images']
#foldername = ['crowdHuman']
foldername = ['demo']
for f in foldername:
    path = 'data/examples/{}'.format(f)
    val_data = os.listdir(path)

    #resize images for crowdhuman
    for filename in val_data:
        img = cv2.imread(os.path.join(path,filename))
        img = cv2.resize(img, (C.random_crop[1], C.random_crop[0]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(path,filename),img)

    out_path = os.path.join(data_path,'detections/{}'.format(f))

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for name in os.listdir(out_path):
        os.remove(out_path+name)

    model.creat_model(C, val_data, phase='inference')
    model.demo(C, val_data, w_path, out_path,f)