import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader

from Dataset import HARDataset

def load_data(type='train', prefix=r'/Users/cyc/Desktop/'):
    motions = ['行走','跑','跳跃','上楼梯','下楼梯']
    labels = {
        '行走': [1.0, 0.0, 0.0, 0.0, 0.0],
        '跑':[0.0, 1.0, 0.0, 0.0, 0.0],
        '跳跃': [0.0, 0.0, 1.0, 0.0, 0.0],
        '上楼梯': [0.0, 0.0, 0.0, 1.0, 0.0],
        '下楼梯': [0.0, 0.0, 0.0, 0.0, 1.0]
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

                # data = pd.read_csv(file_path)[cols]
                # data['total_acc_x'] = data['total_acc_x'] / 9.81
                # data['total_acc_y'] = data['total_acc_y'] / 9.81
                # data['total_acc_z'] = data['total_acc_z'] / 9.81
                # data['gyr_x'] = data['gyr_x'] / 57.30
                # data['gyr_y'] = data['gyr_y'] / 57.30
                # data['gyr_z'] = data['gyr_z'] / 57.30

                data = pd.read_csv(file_path)
                total_acc_x = data['total_acc_x'] / 9.81
                total_acc_y = data['total_acc_y'] / 9.81
                total_acc_z = data['total_acc_z'] / 9.81
                gyr_x = data['gyr_x'] / 57.30
                gyr_y = data['gyr_y'] / 57.30
                gyr_z = data['gyr_z'] / 57.30

                trainFile.append(third_level_path+'/'+file)
                trainX.append([total_acc_x, total_acc_y, total_acc_z, gyr_x, gyr_y, gyr_z])
                trainY.append(labels[second_level_path])

    return trainFile, np.array(trainX,dtype='float32'), np.array(trainY,dtype='float32')

def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def accuracy(y_pred, y):
    '''计算准确率'''
    if len(y_pred.shape) > 1 and y_pred.shape[0] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred == y.argmax(axis=1)
    return float(cmp.type(y.dtype).sum()) / len(y)

model = nn.Sequential(
    nn.Conv1d(6, 64, 3),
    nn.ReLU(),
    nn.Conv1d(64, 64, 3), 
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.MaxPool1d(2), 
    nn.Flatten(), 
    nn.Linear(3968, 100), 
    nn.ReLU(), 
    nn.Linear(100,5), 
    nn.Softmax(dim=1)
)
#parameters initialization
model.apply(init_weight)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

if __name__ == '__main__':

    trainFile, trainX, trainY = load_data('train')
    testFile, testX, testY = load_data('train')

    epoches, batch_size = 20, 32

    dataset = HARDataset(trainX, trainY)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True, num_workers=5)

    # print(model.state_dict())
    for epoch in range(epoches):
        for i, (X, Y) in enumerate(data_loader):
            pred = model(X)
            l = loss( pred, Y )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  

        # if epoch % 5 == 0:
        with torch.no_grad():
            pred = model(torch.tensor(trainX))
            l_ = loss(pred, torch.tensor(trainY))
            accuracy_ = accuracy(pred, torch.tensor(trainY))
            print('epoch={0}: loss={1}, accuracy={2}'.format(epoch, l_, accuracy_))

    torch.save(model, './torchCNN')



