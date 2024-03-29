import numpy as np
from scipy import stats
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
from inflation_all import *
from torch.utils.data import random_split, DataLoader, TensorDataset
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split

# Training functions
# earlystopping = EarlyStopping(patience=10,
#                               verbose=0,
#                               restore_best_weights=True)

#Adding our optimizer

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch


class Net(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.d = 1
        self.p = 0.05
        self.output = output
        self.linear1 = nn.Linear(self.d, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, self.d)
        self.dropout1 = nn.Dropout(self.p)
        self.dropout3 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)        
    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        if self.output == 'relu':
            x = nn.ReLU()(x)
            return x
        if self.output == 'linear':
            return x

# Pytorch early stopping but not restoring best weights from last epoch
earlystopping = EarlyStopping(monitor='val_loss', patience=10,
                              verbose=0, mode='min')
class create_model(pl.LightningModule):
    def __init__(self, loss_fun, output, optimizer, learning_rate, eta, F0, nu):
        super(create_model, self).__init__()
        self.d = 1
        self.p = 0.05
        self.optimizer = optimizer
        self.linear1 = nn.Linear(self.d, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, self.d)
        self.dropout1 = nn.Dropout(self.p)
        self.dropout3 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)
        self.learning_rate = learning_rate
        self.F0 = F0
        self.eta = eta
        self.nu = nu
        self.output = output
        self.loss_fun = loss_fun
        self.automatic_optimization = False
    
    def forward(self, x):
        p = 0.05
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        if self.output == 'relu':
            x = nn.ReLU()(x)
            return x
        if self.output == 'linear':
            return x

    
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer == 'ECD':
            optimizer = ECD_q1(self.parameters(), lr = self.learning_rate, eta = self.eta, F0=self.F0,  nu = self.nu)
        return optimizer    

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()
        # compute loss
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = torch.mean(self.loss_fun(y, y_pred))
        # print(loss)
        loss.backward()
        def closure():
            return loss
        optimizer.step(closure)
        optimizer.zero_grad()
        # self.log_dict({'train_loss': loss}, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def validation_step(self, test_batch, batch_idx):
        X, y = test_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()
        # compute metrics
        loss = torch.mean(self.loss_fun(y, y_pred))
        # self.log_dict({'val_loss': loss}, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('val_loss', loss)
        return loss


class reweightingDataModule(pl.LightningDataModule):
    def __init__(self, bkgd, sgnl, N):
        super().__init__()
        self.bkgd = bkgd
        self.sgnl = sgnl
        self.N = N
    def prepare_data(self):
        X_train, X_test, y_train, y_test = make_data(self.bkgd, self.sgnl, self.N)
        X_train = X_train.reshape(-1,1).astype(np.float32)
        X_test = X_test.reshape(-1,1).astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        self.X_train = torch.from_numpy(X_train)
        self.X_test = torch.from_numpy(X_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)

        self.train_data = TensorDataset(self.X_train, self.y_train)
        self.test_data = TensorDataset(self.X_test, self.y_test)
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=int(0.1*self.N))
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=int(0.1*self.N))




def train(data, 
          loss,
          d = 1,
          hidden = 'relu', 
          output = 'sigmoid', 
          dropout = True, 
          optimizer = 'adam', 
          metrics = ['accuracy'], 
          verbose = 0):
    print(d)
    X_train, X_test, y_train, y_test = data
    
    N = len(X_train) + len(X_test)
    
    model = create_model(loss, d, hidden, output, dropout, optimizer, metrics, verbose)      
    
    model.compile(loss = loss,
                  optimizer = optimizer, 
                  metrics = metrics)
    
    trace = model.fit(X_train, 
                      y_train,
                      epochs = 100, 
                      batch_size = int(0.1*N), 
                      validation_data = (X_test, y_test),
                      callbacks = [earlystopping], 
                      verbose = verbose)
    print(trace.history['val_loss'][-1], '\t', len(trace.history['val_loss']), end = '\t')
    
    return model, trace



def make_data(bkgd, sgnl, N_trn=10**7, N_tst=10**5):
    y_trn = stats.bernoulli.rvs(0.5, size = N_trn)
    
    X_bkgd = bkgd.rvs(size = N_trn)
    X_sgnl = sgnl.rvs(size = N_trn)
    
    X_trn = np.zeros_like(X_bkgd)
    X_trn[y_trn == 0] = X_bkgd[y_trn == 0]
    X_trn[y_trn == 1] = X_sgnl[y_trn == 1]
    
    y_tst = stats.bernoulli.rvs(0.5, size = N_tst)
    
    X_bkgd = bkgd.rvs(size = N_tst)
    X_sgnl = sgnl.rvs(size = N_tst)
    
    X_tst = np.zeros_like(X_bkgd)
    X_tst[y_tst == 0] = X_bkgd[y_tst == 0]
    X_tst[y_tst == 1] = X_sgnl[y_tst == 1]
    
    return X_trn, X_tst, y_trn, y_tst

def split_data(X, y):
    # Split into train and validation sets.
    X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, random_state = 666)
    
    # Standardize both the train and validation set.
    m = np.mean(X_trn, axis = 0)
    s = np.std(X_trn, axis = 0)
    X_trn = (X_trn - m) / s
    X_vld = (X_vld - m) / s
    
    return (X_trn, X_vld, y_trn, y_vld), m, s

def make_lr(bkgd, sgnl):
    return lambda x: sgnl.pdf(x) / bkgd.pdf(x)

def make_mae(bkgd, sgnl, dir_name):
    X_tst = np.load(dir_name + 'X_tst.npy')
    lr = make_lr(bkgd, sgnl)
    lr_tst = np.squeeze(lr(X_tst))
    
    def mae(model_lr):
        return np.abs(model_lr(X_tst) - lr_tst).mean()
    return mae

def make_mpe(bkgd, sgnl, dir_name):
    X_tst = np.load(dir_name + 'X_tst.npy')
    
    lr = make_lr(bkgd, sgnl)
    lr_tst = np.squeeze(lr(X_tst))
    def mpe(model_lr):
        return np.abs((model_lr(X_tst) - lr_tst) / lr_tst).mean() * 100
    return mpe

def make_mr(bkgd, sgnl, dir_name):
    X_tst = np.load(dir_name + 'X_tst.npy')
    
    lr = make_lr(bkgd, sgnl)
    lr_tst = np.squeeze(lr(X_tst))
    def mr(model_lr):
        return np.mean(model_lr(X_tst) / lr_tst)
    return mr

def make_null_statistic(bkgd, sgnl, dir_name):
    X_tst = np.load(dir_name + 'X_tst.npy')
    y_tst = np.load(dir_name + 'y_tst.npy')
    X_null = X_tst[y_tst == 1]
    
    lr = make_lr(bkgd, sgnl)
    null_lr = np.mean(lr(X_null))
    def null_statistic(model_lr):
        return abs(np.mean(model_lr(X_null)) - null_lr)
    return null_statistic
    
def odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(f / (1. - f))
    return model_lr

def square_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(f**2 / (1. - f**2))
    return model_lr

def exp_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(np.exp(f) / (1. - np.exp(f)))
    return model_lr

def pure_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(f)
    return model_lr

def square_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(f**2)
    return model_lr

def exp_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(np.exp(f))
    return model_lr

def pow_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(f**p)
    return model_lr

def pow_exp_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(np.exp(f)**p)
    return model_lr

def pow_odds_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze( (f / (1. - f))**(p - 1))
    return model_lr

def t_tanh(x):
    return 0.5 * (np.tanh(x) + 1)

def tanh_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(t_tanh(f) / (1. - t_tanh(f)))
    return model_lr

def t_arctan(x):
    return 0.5 + (np.arctan(x) / np.pi)

def arctan_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(t_arctan(f) / (1. - t_arctan(f)))
    return model_lr

def probit(x):
    normal = tfp.distributions.Normal(loc=0., scale=1.)
    return normal.cdf(x)

def probit_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return np.squeeze(probit(f) / (1. - probit(f)))
    return model_lr

def tree_lr(model, m = 0, s = 1):
    def model_lr(x):
        x = x.reshape(x.shape[0], -1)
        f = model((x - m) / s)[:, 1]
        return np.squeeze(f / (1. - f))
    return model_lr

'''
#bkgd sgnl maybe an array
# distribution primarily only needed for plotting
def compare(params, bkgd, sgnl, reps=100, N = 10**6, num, filestrs):
    #assert len(params) == len(filestrs)
    data = make_data(bkgd, sgnl, N) + [N]
    for i in range(reps):
        for j in len(params):
            model, trace = train(data, **params[i])
            model.save_weights(filestrs[i])

match = {bce: [bce, square_bce, exp_bce],
         mse: [mse, square_mse, exp_mse],
         mlc: [mlc, square_mlc, exp_mlc],
         sqr: [sqr, square_sqr, exp_sqr]
        }
def c_param(loss, bkgd, sgnl):
    losses = match[loss]
    linear_param = {losses[0], ...}   # the remaining information will have to be put into match as well
    square_param = {losses[1], ...}   # such as whether the activation is relu, sigmoid, or linear
    exponl_param = {losses[2], ...}
    
    params = [linear_param, square_param, exponl_param]
    
    linear_filestr = 'models/demo/...' # need some way to figure out num and other values, like the loss
    square_filestr = 'models/demo/...' # possibly also included in match
    exponl_filestr = 'models/demo/...'
    
    filestrs = [linear_filestr, square_filestr, exponl_filestr]
    
    compare(params, filestrs, bkgd, sgnl, num = ..., filestrs)
    #print statement like "you can find your models in [path to dir]
'''
        