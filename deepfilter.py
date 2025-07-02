import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from Optim import Optim
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, median_absolute_error, r2_score
from model import DeepFilter
from utils import DataS, DataM, mkdir, EarlyStopping, seed_all
import os.path as osp
import logging


class DeepFilterModel(BaseEstimator, RegressorMixin):

    def __init__(self, args):
        
        super().__init__()
        self.epoch = 0
        self.epochs = args.epochs
        self.batchSize = args.batchSize
        self.lr = args.lr
        self.labda = args.labda

        if args.taskName == 'S':

            self.trainSet = DataS(args, mode='train')
            self.validSet = DataS(args, mode='valid', scaler=self.trainSet.scaler)
            self.testSet = DataS(args, mode='test', scaler=self.trainSet.scaler)
            self.dimI = self.trainSet.data.shape[1] - 1
            self.dimO = 1

        else:

            self.trainSet = DataM(args, mode='train')
            self.validSet = DataM(args, mode='valid', scaler=self.trainSet.scaler)
            self.testSet = DataM(args, mode='test', scaler=self.trainSet.scaler)
            self.dimI = self.trainSet.data.shape[1]
            self.dimO = self.dimI

        self.criterionReg = nn.L1Loss()
        self.criterionRec = nn.MSELoss()
        self.model = DeepFilter(args, featureNum=self.dimI, outputNum=self.dimO).to(torch.device('cuda:{}'.format(args.device)))

        self.optimizer = Optim(self.model.parameters(), args.optim, args.lr, args.clip, args.labda)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.8)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        
        self.trainDict = dict()
        self.evalDict = dict()
        self.trainDict['loss'] = np.array([])
        self.evalDict['medianAbsoluteError'] = np.array([])
        self.evalDict['explainedVarianceScore'] = np.array([])
        self.evalDict['meanSquaredError'] = np.array([])
        self.evalDict['meanAbsoluteError'] = np.array([])
        self.evalDict['r2'] = np.array([])
        self.testDict = self.evalDict.copy()
        self.evalDict['r2Best'] = 0
        
        self.savePath = osp.join('./', args.rootPath, args.dataName, args.modelName+args.taskName+str(args.horizon), str(args.labda), str(args.seed))
        mkdir(self.savePath)
        self.early_stop = EarlyStopping(path=self.savePath)


    def fit(self):

        trainLoader = DataLoader(dataset=self.trainSet, batch_size=self.batchSize, shuffle=True)

        for i in range(self.epochs):
            
            self.model.train()
            self.epoch = i

            for batchX, batchY in trainLoader:
                self.model.zero_grad()
                output = self.model(batchX)
                output = output.reshape(batchY.shape)
                loss = self.criterionRec(output, batchY)  # shape
                loss.backward()
                self.optimizer.step()
                self.trainDict['loss'] = np.append(self.trainDict['loss'], loss.data.cpu().numpy())
            logging.info('Epoch: {}, Loss: {}'.format(self.epoch, loss.data.cpu().numpy()))
                                                                                                
            self.valid()    

            if self.evalDict['r2'][-1] > self.evalDict['r2Best']:
                self.evalDict['r2Best'] = self.evalDict['r2'][-1]
            self.early_stop(self.evalDict['r2'][-1], self.model)

            if self.early_stop.early_stop:
                break

        self.test()
        pd.DataFrame(self.trainDict).to_csv(self.savePath+'/trainLog.txt', header=0, index=0)
        pd.DataFrame(self.evalDict).to_csv(self.savePath+'/validLog.txt', header=0, index=0)
        pd.DataFrame(self.testDict).to_csv(self.savePath+'/testLog.txt', header=0, index=0)

    def valid(self):

        with torch.no_grad():

            self.model.eval()
            validLoader = DataLoader(dataset=self.validSet, batch_size=self.batchSize, shuffle=True)
            results = {'output':np.array([]), 'label':np.array([])}

            for batchX, batchY in validLoader:

                output = self.model(batchX)
                output = output.reshape(batchY.shape).data.cpu().numpy()
                batchY = batchY.data.cpu().numpy()
                results['label'] = np.append(results['label'], batchY)
                results['output'] = np.append(results['output'], output)

            r2 = r2_score(results['label'], results['output'])
            explainedVarianceScore = explained_variance_score(results['label'], results['output'])
            meanSquaredError = mean_squared_error(results['label'], results['output'])
            meanAbsoluteError = mean_absolute_error(results['label'], results['output'])
            medianAbsoluteError = median_absolute_error(results['label'], results['output'])
            
            self.evalDict['r2'] = np.append(self.evalDict['r2'], r2)
            self.evalDict['explainedVarianceScore'] = np.append(self.evalDict['explainedVarianceScore'], explainedVarianceScore)
            self.evalDict['meanSquaredError'] = np.append(self.evalDict['meanSquaredError'], meanSquaredError)
            self.evalDict['meanAbsoluteError'] = np.append(self.evalDict['meanAbsoluteError'], meanAbsoluteError)
            self.evalDict['medianAbsoluteError'] = np.append(self.evalDict['medianAbsoluteError'], medianAbsoluteError)

        logging.info('Epoch: {}, Valid Loss: {}, R2: {}'.format(self.epoch, meanSquaredError, r2))

    def test(self):

        with torch.no_grad():

            self.model.eval()
            validLoader = DataLoader(dataset=self.testSet, batch_size=self.batchSize, shuffle=True)
            results = {'output':np.array([]), 'label':np.array([])}

            for batchX, batchY in validLoader:

                output = self.model(batchX)
                output = output.reshape(batchY.shape).data.cpu().numpy()
                batchY = batchY.data.cpu().numpy()
                results['label'] = np.append(results['label'], batchY)
                results['output'] = np.append(results['output'], output)

            r2 = r2_score(results['label'], results['output'])
            explainedVarianceScore = explained_variance_score(results['label'], results['output'])
            meanSquaredError = mean_squared_error(results['label'], results['output'])
            meanAbsoluteError = mean_absolute_error(results['label'], results['output'])
            medianAbsoluteError = median_absolute_error(results['label'], results['output'])
            
            self.testDict['r2'] = np.append(self.testDict['r2'], r2)
            self.testDict['explainedVarianceScore'] = np.append(self.testDict['explainedVarianceScore'], explainedVarianceScore)
            self.testDict['meanSquaredError'] = np.append(self.testDict['meanSquaredError'], meanSquaredError)
            self.testDict['meanAbsoluteError'] = np.append(self.testDict['meanAbsoluteError'], meanAbsoluteError)
            self.testDict['medianAbsoluteError'] = np.append(self.testDict['medianAbsoluteError'], medianAbsoluteError)

        logging.info('Epoch: {}, Test Loss: {}, R2: {}'.format(self.epoch, meanSquaredError, r2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    # parser.add_argument('--dataName', type=str, default='0201A13-白城工业园区站')
    parser.add_argument('--dataName', type=str, default='0106A15')

    parser.add_argument('--taskName', type=str, default='S')
    parser.add_argument('--rootPath', type=str, default='baseline1226')
    parser.add_argument('--modelName', type=str, default='DeepFilter')

    parser.add_argument('--trainProb', type=float, default=0.7)
    parser.add_argument('--validProb', type=float, default=0.15)

    parser.add_argument('--hidR', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--layerR', type=int, default=1, help='number of RNN hidden layers')
    parser.add_argument('--window', type=int, default=16, help='window size')
    parser.add_argument('--init_ratio', type=int, default=0.02)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--kernelCNN', type=int, default=6)
    parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')

    parser.add_argument('--d_k', type=int, default=32, help='default=64')
    parser.add_argument('--labda', type=float, default=0)
    parser.add_argument('--decayS', type=float, default=16)
    parser.add_argument('--decayT', type=float, default=1)
    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--labda', type=float, default=0.00005)

    
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=64, metavar='N')
    parser.add_argument('--dropout', type=float, default=0, help='(0 = no dropout)')
    parser.add_argument('--device', default=0)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1024)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    seed_all(args.seed)
    model = DeepFilterModel(args)
    model.fit()
    model.valid()
