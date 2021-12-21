import os
from keras.backend import argmax
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical


def load_data(type='train', prefix=r'/Users/cyc/Desktop/'):
    stillness = ['站着', '坐着', '躺着', '蹲着', '弯着腰']
    labels = {
        '站着': [1,0,0,0,0],
        '坐着': [0,1,0,0,0],
        '躺着': [0,0,1,0,0],
        '蹲着': [0,0,0,1,0],
        '弯着腰': [0,0,0,0,1],
    }

    cols = ['total_acc_x', 'total_acc_y','total_acc_z','gyr_x', 'gyr_y', 'gyr_z']
    
    trainFile, trainX, trainY = [], [], []

    # transition = []
    # transition_label = []
    # period = []
    # period_label = []

    first_level = prefix + r'new {0} set/segment128/ADL'.format(type)
    second_level_paths = [ dir for dir in os.listdir(first_level) if dir in stillness and os.path.isdir(os.path.join(first_level, dir)) ]
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
                trainX.append(data)
                trainY.append(labels[second_level_path])

    return trainFile, np.array(trainX), np.array(trainY)

trainFile, trainX, trainY = load_data(type ='train')
# testFile, testX, testY = load_data(type ='test')
print(len(trainX), len(trainY) )
_trainx,  _validx, _trainy, _validy = train_test_split(trainX, trainY, test_size= 0.2, random_state=0)



# fit and evaluate a model
def evaluate_model():
    verbose, epochs, batch_size = 1, 100, 32
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
    train_loss, train_acc = model.evaluate(_trainx, _trainy, batch_size = 32, verbose = 0)
    print('test_loss, test_acc:', train_loss, train_acc)
    valid_loss, valid_acc = model.evaluate(_validx, _validy, batch_size = 32, verbose = 0)
    print('valid_loss, valid_acc:', valid_loss, valid_acc)

    model.save('./SCNN')

    print('_validx num', len(_validx))
    train_out = model.predict(_validx)
    train_res = [ list(i).index(max( list(i))) for i in  list(train_out) ]
    train_lable = [ list(i).index(max(list(i))) for i in list(_validy) ]

    

    train_acc = sum([ 1 for i in range(len(train_res)) if train_res[i] == train_lable[i] ]) / len(train_res)
    # test_acc = sum([ 1 for i in range(len(test_res)) if test_res[i] == test_lable[i] ]) / len(test_res)
    labels = ['stand',  'sit',  'lie',  'squat',  'bend']
    results = []
    # test_res = print(test_res)
    # # testY = print(test_label)
    # for i in range(len(test_res)):
    #     if test_res[i] != test_lable[i]:
    #         results.append(testFile[i] + ' ---> result:' + labels[ test_res[i]])
    # print(len(results))
    # for i in results:
    #     print(i)

    cm = confusion_matrix(train_lable, train_res)
    cm = pd.DataFrame(cm, index=labels, columns=labels)

    print(cm)