import os
from random import sample, shuffle
from keras.backend import argmax
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical


def load_data(type='train', prefix=r'/Users/cyc/Desktop/'):
    motions = ['行走','跑','跳跃','上楼梯','下楼梯','坐下','站起','蹲下','躺下','弯腰拾取', '前跌', '后跌', '左跌', '右跌']
    p_motions = ['行走','跑','跳跃','上楼梯','下楼梯']
    t_motions = ['坐下','站起','蹲下','躺下','弯腰拾取', '前跌', '后跌', '左跌', '右跌']
    labels = {
        'transition': [1,0],
        'period': [0,1]
    }

    cols = ['total_acc_x', 'total_acc_y','total_acc_z','gyr_x', 'gyr_y', 'gyr_z']
    
    trainFile, trainX, trainY = [], [], []

    transition = []
    period = []

    raw_path = prefix + r'new {0} set/segment128'.format(type)
    first_level_paths = [ dir for dir in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, dir)) ]
    for first_level_path in first_level_paths:
        first_level = os.path.join(raw_path, first_level_path)
        second_level_paths = [ dir for dir in os.listdir(first_level) if dir in motions and os.path.isdir(os.path.join(first_level, dir)) ]
        for second_level_path in second_level_paths:
            second_level = os.path.join(first_level, second_level_path)
            third_level_paths = [ dir for dir in os.listdir(second_level) if os.path.isdir(os.path.join(second_level, dir)) ]
            for third_level_path in third_level_paths:
                third_level = os.path.join(second_level, third_level_path)
                files = [ file for file in os.listdir(third_level) if os.path.isfile(os.path.join(third_level, file)) and not file.startswith('.') ]
                for file in files:
                    file_path = os.path.join(third_level, file)

                    if second_level_path in p_motions:
                        period.append(file_path)
                    elif second_level_path in t_motions:
                        transition.append(file_path)

                # data = pd.read_csv(file_path)[cols]
                # data['total_acc_x'] = data['total_acc_x'] / 9.80
                # data['total_acc_y'] = data['total_acc_y'] / 9.80
                # data['total_acc_z'] = data['total_acc_z'] / 9.80
                # data['gyr_x'] = data['gyr_x'] / 57.30
                # data['gyr_y'] = data['gyr_y'] / 57.30
                # data['gyr_z'] = data['gyr_z'] / 57.30
 
                # trainFile.append(third_level_path+'/'+file)
                # trainX.append(data)
                # trainY.append([1,0] if second_level_path in t_motions else [0,1])
    print(len(transition))
    print(len(period))
    period = sample(period, len(transition))

    files = transition + period
    shuffle(files)

    for file in files:
        data = pd.read_csv(file)[cols]
        data['total_acc_x'] = data['total_acc_x'] / 9.80
        data['total_acc_y'] = data['total_acc_y'] / 9.80
        data['total_acc_z'] = data['total_acc_z'] / 9.80
        data['gyr_x'] = data['gyr_x'] / 57.30
        data['gyr_y'] = data['gyr_y'] / 57.30
        data['gyr_z'] = data['gyr_z'] / 57.30

        if file in transition:
            # trainFile.append(third_level_path+'/'+file)
            trainX.append(data)
            trainY.append([1,0])
        elif file in period:
            trainX.append(data)
            trainY.append([0,1])

    return trainFile, np.array(trainX), np.array(trainY)

trainFile, trainX, trainY = load_data(type ='train')
# testFile, testX, testY = load_data(type ='test')
print(len(trainX), len(trainY))
_trainx,  _validx, _trainy, _validy = train_test_split(trainX, trainY, test_size= 0.2, random_state=0)

print('trainX num:', len(_trainx))
print('trainY num:', len(_trainy))


# fit and evaluate a model
def evaluate_model():
    verbose, epochs, batch_size = 1, 50, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    print(len(_trainx))
    print(len(_trainy))
    model.fit(_trainx, _trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate model
    # loss, accuracy = model.evaluate(_validx, _validy, batch_size=batch_size, verbose=0)
    return model #, loss, accuracy


if __name__ == '__main__':

    model = evaluate_model()
    train_loss, train_acc = model.evaluate(_trainx, _trainy, batch_size = 32, verbose = 0)
    print('test_loss, test_acc:', train_loss, train_acc)
    valid_loss, valid_acc = model.evaluate(_validx, _validy, batch_size = 32, verbose = 0)
    print('valid_loss, valid_acc:', valid_loss, valid_acc)

    model.save('./PTCNN')

    # train_out = model.predict(_trainx)
    # train_res = [ list(i).index(max( list(i))) for i in  list(train_out) ]
    # train_lable = [ list(i).index(max(list(i))) for i in list(_trainy) ]

    # train_acc = sum([ 1 for i in range(len(train_res)) if train_res[i] == train_lable[i] ]) / len(train_res)
    # labels = ['walk',  'run',  'jump',  'upstairs',  'downstairs']
    # results = []