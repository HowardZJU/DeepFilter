import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import random


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

def mkdir(path):

    if os.path.exists(path) == False:
        os.makedirs(path)

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=True, delta=0, path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model, os.path.join(self.path, 'finish_model.pkl'))                 # 这里会存储迄今最优的模型
        self.best_score = score

class Data_utility(object):
    
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=1):
        self.cuda = cuda;
        self.window = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.loadtxt(fin,delimiter=',');
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.featureNum = self.dat.shape;
        self.normalize = normalize
        self.scale = np.ones(self.featureNum);
        self._normalized();
        self._split(int(train * self.n), int((train+valid) * self.n), self.n);
        
        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.featureNum);
            
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);
        
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));
    
    def _normalized(self):
        #normalized by the maximum value of entire matrix.
       
        if (self.normalize == 0):
            self.dat = self.rawdat
            
        if (self.normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);
            
        #normlized by the maximum value of each row(sensor).
        if (self.normalize == 2):
            for i in range(self.featureNum):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]));
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]));
            
        
    def _split(self, train, valid, test):
        
        train_set = range(self.window+self.h-1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);
        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set);
        X = torch.zeros((n,self.window,self.featureNum));
        Y = torch.zeros((n,self.featureNum));
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.window;
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]);
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]);

        return [X, Y];

    def get_batches(self, inputs, targets, batchSize, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batchSize)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();  
            yield Variable(X), Variable(Y);
            start_idx += batchSize


class DataDb(Dataset):

    def __init__(self, args, mode, scaler=None):

        self.mode = mode
        self.data = pd.read_csv('./data/{}.txt'.format(args.dataName), header=None, index_col=None).values
        self.data = np.concatenate((self.data[4:, :5], self.data[3:-1, [4]], self.data[2:-2, [4]], self.data[1:-3, [4]], (self.data[4:, [5]] + self.data[4:, [6]])/2, self.data[3:-1, [7]], self.data[2:-2, [7]], self.data[1:-3, [7]], self.data[0:-4, [7]], self.data[4:, [-1]]), axis=1)
        self.data = self._split(int(args.trainProb * len(self.data)), int((args.trainProb+args.validProb) * len(self.data)))
        self._normalized(scaler)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))

    def _split(self, trainNum, validNum):

        if self.mode == 'train':
            return self.data[:trainNum, :]
        if self.mode == 'valid':
            return self.data[trainNum:validNum, :]
        if self.mode == 'test':
            return self.data[validNum:, :]

    def _normalized(self, scaler):

        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):

        return self.data[index, :-1], self.data[index, -1]


    def __len__(self):
        return len(self.data)



class DataS(Dataset):

    def __init__(self, args, mode, scaler=None):

        self.mode = mode
        self.window = args.window
        self.h = args.horizon
        self.data = pd.read_csv('./data/{}.txt'.format(args.dataName), header=None, index_col=None).values
        self.data = np.concatenate((self.data[2:, :-1], self.data[1:-1, [-1]], self.data[:-2, [-1]], self.data[2:, [-1]]), axis=1)
        self.data = self._split(int(args.trainProb * len(self.data)), int((args.trainProb+args.validProb) * len(self.data)))
        self.n, self.featureNum = self.data.shape
        self.featureNum = self.featureNum - 1 # 让m是输入特征维数
        self._normalized(scaler)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))

    def _split(self, trainNum, validNum):

        if self.mode == 'train':
            return self.data[:trainNum, :]
        if self.mode == 'valid':
            return self.data[trainNum:validNum, :]
        if self.mode == 'test':
            return self.data[validNum:, :]

    def _normalized(self, scaler):

        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):


        start = index
        end = index + self.window
        return self.data[start:end, :-1], self.data[end - 1 + self.h, -1]

    def __len__(self):
        return len(self.data) - self.window - self.h


class DataM(Dataset):
    
    def __init__(self, args, mode, scaler=None):

        self.mode = mode
        self.window = args.window
        self.h = args.horizon
        self.data = pd.read_csv('./data/{}.txt'.format(args.dataName), header=None, index_col=None).values
        self.data = self._split(int(args.trainProb * len(self.data)), int((args.trainProb+args.validProb) * len(self.data)))
        self.n, self.featureNum = self.data.shape
        self._normalized(scaler);
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))
        
    def _split(self, train_num, valid_num):
        
        if self.mode == 'train':
            return self.data[:train_num, :]
        if self.mode == 'valid':
            return self.data[train_num:valid_num, :]
        if self.mode == 'test':
            return self.data[valid_num:, :]
    
    
    def _normalized(self, scaler):
    
        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

        
    def __getitem__(self, index):

        start = index
        end = index + self.window

        return self.data[start:end], self.data[end]

    def __len__(self):
        return len(self.data) - self.window - self.h


class DataR(Dataset):

    def __init__(self, args, mode, scaler=None):

        self.mode = mode
        self.device = args.device
        self.data = pd.read_csv('./data/{}.txt'.format(args.dataName), header=None, index_col=None).values
        self.data = np.concatenate((self.data[4:, :-1], self.data[3:-1, [-1]], self.data[2:-2, [-1]], self.data[1:-3, [-1]], self.data[:-4, [-1]], self.data[4:, [-1]]), axis=1)
        self.data = self._split(int(args.trainProb * len(self.data)), int((args.trainProb+args.validProb) * len(self.data)))
        self.n, self.featureNum = self.data.shape
        self._normalized(scaler)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=torch.device('cuda:{}'.format(args.device)))

    def _split(self, trainNum, validNum):

        if self.mode == 'train':
            return self.data[:trainNum, :]
        if self.mode == 'valid':
            return self.data[trainNum:validNum, :]
        if self.mode == 'test':
            return self.data[validNum:, :]

    def _normalized(self, scaler):

        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):

        return self.data[index, :-1], self.data[index, -1]


    def __len__(self):
        return len(self.data)
