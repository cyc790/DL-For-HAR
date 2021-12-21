import os
from keras.backend import argmax
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

'''
'''

def load_data(type='train', prefix=r'/Users/cyc/Desktop/'):
    motions = ['行走','跑','跳跃','上楼梯','下楼梯']
    labels = {
        '行走': [1, 0, 0, 0, 0],
        '跑':[0, 1, 0, 0, 0],
        '跳跃': [0, 0, 1, 0, 0],
        '上楼梯': [0, 0, 0, 1, 0],
        '下楼梯': [0, 0, 0, 0, 1]
    }

    cols = ['total_acc_x', 'total_acc_y','total_acc_z','gyr_x', 'gyr_y', 'gyr_z']
    
    trainFile, trainX, trainY = [], [], []

    first_level = prefix + r'new {0} set/segment128/ADL'.format(type)
    second_level_paths = [ dir for dir in os.listdir(first_level) if dir in motions and os.path.isdir(os.path.join(first_level, dir)) ]
    for second_level_path in second_level_paths:
        second_level = os.path.join(first_level, second_level_path)
        third_level_paths = [ dir for dir in os.listdir(second_level) if os.path.isdir(os.path.join(second_level, dir)) ]
        for third_level_path in third_level_paths:
            third_level = os.path.join(second_level, third_level_path)
            files = [ file for file in os.listdir(third_level) if os.path.isfile(os.path.join(third_level, file)) and not file.startswith('.') ]
            for file in files:
                file_path = os.path.join(third_level, file)

                data = pd.read_csv(file_path)[cols]
                data['total_acc_x'] = data['total_acc_x'] / 9.81
                data['total_acc_y'] = data['total_acc_y'] / 9.81
                data['total_acc_z'] = data['total_acc_z'] / 9.81
                data['gyr_x'] = data['gyr_x'] / 57.30
                data['gyr_y'] = data['gyr_y'] / 57.30
                data['gyr_z'] = data['gyr_z'] / 57.30

                trainFile.append(third_level_path+'/'+file)
                trainX.append(data)
                trainY.append(labels[second_level_path])

    return trainFile, np.array(trainX), np.array(trainY)

# trainFile, trainX, trainY = load_data(type ='train')
# testFile, testX, testY = load_data(type ='test')
# print(len(trainX), len(trainY) )
# _trainx,  _validx, _trainy, _validy = train_test_split(trainX, trainY, test_size= 0.2, random_state=0)



 

if __name__ == '__main__':

    model = load_model(r'/Users/cyc/Desktop/Deep ML HAR/TCNN/')
    print(model.summary())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('TCNNModel.tflite', 'wb') as f:
        f.write(tflite_model)
    
    
    # print('train_acc:', train_acc)
# run the experiment