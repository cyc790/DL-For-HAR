import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, LSTM, Reshape
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Flatten

def generate_label(path):
    labels = {
        '坐着':   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '站着':   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '躺着':   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '蹲着':   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        '弯着腰':  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        '行走':    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        '跑':      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        '跳跃':    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        '上楼梯':   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        '下楼梯':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        '跌倒':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        '过渡性动作': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    if path in ['坐着','站着','躺着','蹲着','弯着腰','行走','跑', '跳跃','上楼梯','下楼梯']:
        return labels[path]
    elif path in ['站起','坐下' ,'蹲下','躺下','弯腰捡拾']:
        return labels['过渡性动作']
    elif path in ['前跌', '后跌', '左跌', '右跌']:
        return labels['跌倒']

def load_data(type='train', prefix=r'/Users/cyc/Desktop/'):
    types = ['坐着','站着','躺着','蹲着','弯着腰','行走','跑', '跳跃','上楼梯','下楼梯', '站起','坐下' ,'蹲下','躺下','弯腰捡拾', '前跌', '后跌', '左跌', '右跌']
    cols = ['total_acc_x', 'total_acc_y','total_acc_z','gyr_x', 'gyr_y', 'gyr_z']
    
    trainFile, trainX, trainY = [], [], []

    raw_path = prefix + r'new {0} set/segment128'.format(type)
    first_level_paths = [ dir for dir in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, dir)) ]
    for first_level_path in first_level_paths:
        first_level = os.path.join(raw_path, first_level_path)
        second_level_paths = [ dir for dir in os.listdir(first_level) if dir in types and os.path.isdir(os.path.join(first_level, dir)) ]
        for second_level_path in second_level_paths:
            second_level = os.path.join(first_level, second_level_path)
            third_level_paths = [ dir for dir in os.listdir(second_level) if os.path.isdir(os.path.join(second_level, dir)) ]
            for third_level_path in third_level_paths:
                third_level = os.path.join(second_level, third_level_path)
                files = [ file for file in os.listdir(third_level) if os.path.isfile(os.path.join(third_level, file)) and not file.startswith('.') ]
                for file in files:
                    file_path = os.path.join(third_level, file)

                    data = pd.read_csv(file_path)[cols]
                    data['total_acc_x'] = data['total_acc_x'] / 9.80
                    data['total_acc_y'] = data['total_acc_y'] / 9.80
                    data['total_acc_z'] = data['total_acc_z'] / 9.80
                    data['gyr_x'] = data['gyr_x'] / 57.30
                    data['gyr_y'] = data['gyr_y'] / 57.30
                    data['gyr_z'] = data['gyr_z'] / 57.30

                    trainFile.append(third_level_path+'/'+file)
                    trainX.append(data.values)
                    trainY.append(generate_label(second_level_path))

    # print('trainX dim:',trainX[0].shape)
    # print('trainY dim:', trainY.shape)
    # for i in trainY:
    #     if len(i) != 12:
    #         print(i)
        

    return trainFile,np.asarray(trainX,dtype='float32'), np.asarray(trainY, dtype='float32')

def create_model(input_shape, output_shape, lr=1e-3):
    """[summary]

    Args:
        input_shape ([type]): (128, 6)
        output_shape ([type]): class nums
    """
    DeepConvLSTM = Sequential()
    DeepConvLSTM.add(Conv1D(64, kernel_size=(5,), padding='same', activation='relu', input_shape=input_shape))
    DeepConvLSTM.add(Conv1D(64, kernel_size=(5,), padding='same', activation='relu'))
    DeepConvLSTM.add(Conv1D(1, kernel_size=(5,), padding='same', activation='relu'))
    DeepConvLSTM.add(LSTM(128,return_sequences=True))
    DeepConvLSTM.add(Dropout(0.5))
    DeepConvLSTM.add(LSTM(128))
    DeepConvLSTM.add(Dropout(0.5))
    DeepConvLSTM.add(Flatten())
    DeepConvLSTM.add(Dense(output_shape, activation='softmax'))
    DeepConvLSTM.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

    return DeepConvLSTM

if __name__ == '__main__':
    trainFile, trainX, trainY = load_data(type ='train')
    print(len(trainX), len(trainY) )
    _trainx,  _validx, _trainy, _validy = train_test_split(trainX, trainY, test_size= 0.2, random_state=0)

    model = load_model('./12-DeepConvLSTM')
    # model = create_model((128,6), 12)
    model.fit(_trainx, _trainy, batch_size=32,epochs=50, verbose=1)

    train_loss, train_acc = model.evaluate(_trainx, _trainy, batch_size = 32, verbose = 0)
    print('test_loss, test_acc:', train_loss, train_acc)
    valid_loss, valid_acc = model.evaluate(_validx, _validy, batch_size = 32, verbose = 0)
    print('valid_loss, valid_acc:', valid_loss, valid_acc)

    model.save('./12-DeepConvLSTM')









