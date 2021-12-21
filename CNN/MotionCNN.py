import os
from keras.backend import argmax
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical


'''
错误集中出现在某个人的某个动作上，考虑是个体差异性的原因。怎样消除？
------------------------------------------------------------------------------
           walk  run  jump  upstairs  downstairs
walk         236    0     0         3          16
run            0  223     7         0          13
jump           0   32   186         0          31
upstairs       0    0     0       238           2
downstairs     3    1     0         4         253
------------------------------------------------------------------------------
118
黄/51_178_0_1637751695054跳跃数据采集.csv ---> result:run
黄/170_297_0_1637751615720跳跃数据采集.csv ---> result:downstairs
任/378_505_1_1637667116482跳跃数据采集.csv ---> result:run
任/51_178_0_1637667076189跳跃数据采集.csv ---> result:run
任/351_478_0_1637667019382跳跃数据采集.csv ---> result:run
任/51_178_0_1637667043250跳跃数据采集.csv ---> result:run
任/348_475_0_1637667092155跳跃数据采集.csv ---> result:run
任/51_178_0_1637667008559跳跃数据采集.csv ---> result:run
任/51_178_0_1637667138150跳跃数据采集.csv ---> result:run
任/363_490_0_1637667030467跳跃数据采集.csv ---> result:run
任/51_178_0_1637667127470跳跃数据采集.csv ---> result:run
任/51_178_0_1637667030467跳跃数据采集.csv ---> result:run
任/359_486_0_1637667159751跳跃数据采集.csv ---> result:run
任/332_459_0_1637667138150跳跃数据采集.csv ---> result:run
任/201_328_0_1637667019382跳跃数据采集.csv ---> result:run
任/51_178_0_1637667065371跳跃数据采集.csv ---> result:run
任/343_470_0_1637667008559跳跃数据采集.csv ---> result:run
任/199_326_0_1637667092155跳跃数据采集.csv ---> result:run
任/207_334_0_1637667030467跳跃数据采集.csv ---> result:run
任/192_319_0_1637667148729跳跃数据采集.csv ---> result:run
任/188_315_0_1637667127470跳跃数据采集.csv ---> result:run
任/325_452_0_1637667127470跳跃数据采集.csv ---> result:run
任/346_473_0_1637667053972跳跃数据采集.csv ---> result:run
任/51_178_0_1637667148729跳跃数据采集.csv ---> result:run
任/199_326_0_1637667076189跳跃数据采集.csv ---> result:run
任/51_178_0_1637667184214跳跃数据采集.csv ---> result:run
任/380_507_0_1637667065371跳跃数据采集.csv ---> result:run
任/51_178_0_1637667170144跳跃数据采集.csv ---> result:run
任/51_178_0_1637667019382跳跃数据采集.csv ---> result:run
任/205_332_0_1637667159751跳跃数据采集.csv ---> result:run
任/249_376_0_1637667105130跳跃数据采集.csv ---> result:run
任/51_178_0_1637667159751跳跃数据采集.csv ---> result:run
任/191_318_0_1637667138150跳跃数据采集.csv ---> result:run
任/192_319_0_1637667043250跳跃数据采集.csv ---> result:run
任/348_475_0_1637667076189跳跃数据采集.csv ---> result:run
任/351_478_0_1637666986611跳跃数据采集.csv ---> result:run
任/51_178_0_1637667092155跳跃数据采集.csv ---> result:run
邹/228_355_0_1637750075964跳跃数据采集.csv ---> result:downstairs
邹/207_334_0_1637750099342跳跃数据采集.csv ---> result:downstairs
邹/401_528_0_1637750088010跳跃数据采集.csv ---> result:downstairs
邹/242_369_0_1637750176809跳跃数据采集.csv ---> result:downstairs
邹/216_343_0_1637750188480跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750132393跳跃数据采集.csv ---> result:downstairs
邹/360_487_0_1637750119708跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750155924跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750188480跳跃数据采集.csv ---> result:downstairs
邹/438_565_0_1637750049662跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750088010跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750099342跳跃数据采集.csv ---> result:downstairs
邹/228_355_0_1637750155924跳跃数据采集.csv ---> result:downstairs
邹/304_431_0_1637750203814跳跃数据采集.csv ---> result:downstairs
邹/434_561_0_1637750176809跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750119708跳跃数据采集.csv ---> result:downstairs
邹/244_371_0_1637750049662跳跃数据采集.csv ---> result:downstairs
邹/226_353_0_1637750088010跳跃数据采集.csv ---> result:downstairs
邹/364_491_0_1637750099342跳跃数据采集.csv ---> result:downstairs
邹/405_532_0_1637750155924跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750176809跳跃数据采集.csv ---> result:downstairs
邹/405_532_0_1637750075964跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750143744跳跃数据采集.csv ---> result:downstairs
邹/557_684_0_1637750203814跳跃数据采集.csv ---> result:downstairs
邹/381_508_0_1637750188480跳跃数据采集.csv ---> result:downstairs
邹/51_178_0_1637750203814跳跃数据采集.csv ---> result:downstairs
邹/365_492_0_1637750143744跳跃数据采集.csv ---> result:downstairs
邹/670_797_0_1637750119708跳跃数据采集.csv ---> result:downstairs
黄/265_392_0_1637750922039下楼梯数据采集.csv ---> result:upstairs
黄/277_404_0_1637750989102下楼梯数据采集.csv ---> result:walk
黄/247_374_0_1637751054602下楼梯数据采集.csv ---> result:walk
黄/153_280_0_1637751001546下楼梯数据采集.csv ---> result:walk
王/237_364_0_1637827660955下楼梯数据采集.csv ---> result:run
常/51_178_0_1637666185303下楼梯数据采集.csv ---> result:upstairs
常/240_367_0_1637666253422下楼梯数据采集.csv ---> result:upstairs
常/144_271_0_1637666220169下楼梯数据采集.csv ---> result:upstairs
任/136_263_0_1637666702001下楼梯数据采集.csv ---> result:run
任/51_178_0_1637666591859下楼梯数据采集.csv ---> result:walk
任/120_247_0_1637666618821下楼梯数据采集.csv ---> result:run
王/199_326_0_1637825987811跑数据采集.csv ---> result:downstairs
王/51_178_0_1637826101205跑数据采集.csv ---> result:downstairs
王/345_472_0_1637826101205跑数据采集.csv ---> result:downstairs
王/224_351_0_1637825935408跑数据采集.csv ---> result:downstairs
王/341_468_0_1637825947235跑数据采集.csv ---> result:downstairs
王/341_468_0_1637826013440跑数据采集.csv ---> result:downstairs
王/210_337_0_1637826157052跑数据采集.csv ---> result:downstairs
王/398_525_0_1637825935408跑数据采集.csv ---> result:downstairs
王/355_482_0_1637825999791跑数据采集.csv ---> result:downstairs
王/51_178_0_1637826142638跑数据采集.csv ---> result:downstairs
王/196_323_0_1637826013440跑数据采集.csv ---> result:downstairs
王/51_178_0_1637826237410跑数据采集.csv ---> result:downstairs
王/203_330_0_1637826083744跑数据采集.csv ---> result:downstairs
王/51_178_0_1637825987811跑数据采集.csv ---> result:downstairs
王/51_178_0_1637826039649跑数据采集.csv ---> result:downstairs
任/51_178_0_1637665473330跑数据采集.csv ---> result:jump
任/202_329_0_1637665387422跑数据采集.csv ---> result:jump
任/51_178_0_1637665563452跑数据采集.csv ---> result:jump
任/337_464_0_1637665494666跑数据采集.csv ---> result:jump
任/333_460_0_1637665552488跑数据采集.csv ---> result:jump
任/345_472_0_1637665473330跑数据采集.csv ---> result:jump
任/201_328_0_1637665450605跑数据采集.csv ---> result:jump
任/51_178_0_1637665365421跑数据采集.csv ---> result:jump
任/345_472_0_1637665398205跑数据采集.csv ---> result:jump
任/392_519_0_1637665428925跑数据采集.csv ---> result:jump
任/342_469_0_1637665578233跑数据采集.csv ---> result:jump
任/51_178_0_1637665387422跑数据采集.csv ---> result:jump
王/51_178_0_1637825921184行走数据采集.csv ---> result:downstairs
王/195_322_0_1637825855467行走数据采集.csv ---> result:downstairs
王/51_178_0_1637825842146行走数据采集.csv ---> result:downstairs
王/201_328_0_1637825882951行走数据采集.csv ---> result:downstairs
王/51_178_0_1637825907893行走数据采集.csv ---> result:downstairs
王/51_178_0_1637825807776行走数据采集.csv ---> result:upstairs
王/197_324_0_1637825308055行走数据采集.csv ---> result:upstairs
王/51_178_0_1637825796128行走数据采集.csv ---> result:upstairs
王/200_327_0_1637825907893行走数据采集.csv ---> result:downstairs
王/51_178_0_1637825896513行走数据采集.csv ---> result:downstairs
王/211_338_0_1637825831181行走数据采集.csv ---> result:downstairs
任/186_313_0_1637663612490行走0数据采集.csv ---> result:downstairs
任/322_449_0_1637663444613行走1数据采集.csv ---> result:downstairs
邹/358_485_0_1637749961146行走数据采集.csv ---> result:upstairs
邹/51_178_0_1637750575369上楼梯数据采集.csv ---> result:downstairs
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

trainFile, trainX, trainY = load_data(type ='train')
testFile, testX, testY = load_data(type ='test')
print(len(trainX), len(trainY) )
_trainx,  _validx, _trainy, _validy = train_test_split(trainX, trainY, test_size= 0.2, random_state=0)

# fit and evaluate a model
def evaluate_model():
    verbose, epochs, batch_size = 0, 100, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(_trainx, _trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate model
    # loss, accuracy = model.evaluate(_validx, _validy, batch_size=batch_size, verbose=0)
    return model #, loss, accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_data()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 

if __name__ == '__main__':

    model = evaluate_model()
    print(model.summary())
    model.save('./CNN.h5')
    # print('valid_loss, valid_acc:', valid_loss, valid_acc)
    train_loss, train_acc = model.evaluate(_trainx, _trainy, batch_size = 32, verbose = 0)
    print('test_loss, test_acc:', train_loss, train_acc)
    valid_loss, valid_acc = model.evaluate(_validx, _validy, batch_size = 32, verbose = 0)
    print('valid_loss, valid_acc:', valid_loss, valid_acc)
    test_loss, test_acc = model.evaluate(testX, testY, batch_size = 32, verbose = 0)
    print('test_loss, test_acc:', test_loss, test_acc)

    train_out = model.predict(_trainx)
    test_out = model.predict(testX)
    train_res = [ list(i).index(max( list(i))) for i in  list(train_out) ]
    train_lable = [ list(i).index(max(list(i))) for i in list(_trainy) ]
    test_res = [ list(i).index(max(list(i))) for i in list(test_out) ]
    test_lable = [ list(i).index(max(list(i))) for i in list(testY) ]

    # print(train_res)
    # print(train_lable)

    train_acc = sum([ 1 for i in range(len(train_res)) if train_res[i] == train_lable[i] ]) / len(train_res)
    # test_acc = sum([ 1 for i in range(len(test_res)) if test_res[i] == test_lable[i] ]) / len(test_res)
    labels = ['walk',  'run',  'jump',  'upstairs',  'downstairs']
    results = []
    # test_res = print(test_res)
    # # testY = print(test_label)
    # for i in range(len(test_res)):
    #     if test_res[i] != test_lable[i]:
    #         results.append(testFile[i] + ' ---> result:' + labels[ test_res[i]])
    # print(len(results))
    # for i in results:
    #     print(i)

    # cm = confusion_matrix(test_lable, test_res)
    # cm = pd.DataFrame(cm, index=labels, columns=labels)

    # print(cm)



    
    
    # print('train_acc:', train_acc)
# run the experiment