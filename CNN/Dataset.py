from torch.utils.data import Dataset,DataLoader

class HARDataset(Dataset):
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def __getitem__(self, index) :
        return self.trainX[index], self.trainY[index]

    def __len__(self):
        return len(self.trainX)