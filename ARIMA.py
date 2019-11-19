import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score 

def cost(parameters, *arg): 
    #p,d,q,P,D,Q,s,a1,a2,a3,a4 = 1,1,1,1,1,1,3,1,1,1,1
    p,d,q,P,D,Q,s,a1,a2,a3,a4= parameters
    param = (int(p),int(d),int(q))
    param_seasonal = (int(P),int(D),int(Q),int(s))
    TREND = [int(a1>0.5),int(a2>0.5),int(a3>0.5),int(a4>0.5)]
    data = np.reshape(np.array(arg).astype(float),-1,1)
    batch = NestedCrossVal(data,5,15,10)
    Cost = [] 
    k=5
    for i in range(k):
        dataset = batch[i]
        train = dataset[:len(dataset)-15]
        val = dataset[len(dataset)-15:]
        model = sm.tsa.statespace.SARIMAX(train,order=param,seasonal_order=param_seasonal, enforce_stationarity=False,enforce_invertibility=False)
        FitedModel = model.fit()
        pred = FitedModel.get_prediction(start= (len(dataset)-14),end = len(dataset), dynamic=False) 
        y_pred = np.array(pred.predicted_mean)
        mf, smape, m = metrics(val,y_pred) 
        Cost.append(float(m))
    #plt.plot(val,'r')
    #plt.plot(y_pred,'b')
    result = np.array(Cost).mean()*100     
    return result

from scipy.optimize import differential_evolution

def optimize(data):
    #initial guess for variation of parameters
    #            p       d       q      P      D      Q       s       a1    a2     a3     a4            
    bounds = [(0, 3), (0, 2), (0,1), (0, 1), (0,2), (0,1), (2,6), (0,2), (0,2), (0,2), (0,2) ]
        
    #producing "experimental" data 
    #packing "experimental" data into args
    args = data

    result = differential_evolution(cost, bounds, args=args,maxiter = 10, popsize = 10  ,recombination = 0.25)
    result.x
    return result.x


def evaluate(DATA,parameters,serie):
    RMSE = []
    MF = []
    SMAPE = []
    data = DATA[:,serie]
    y = []
    RESID=[]
    colors = np.linspace(start=100, stop=255, num=90)
    for i in range(0,90,1):
        p,d,q,P,D,Q,s,a1,a2,a3,a4 = parameters[i,:]
        (int(p),int(d),int(q))
        param = (int(p),int(d),int(q))
        param_seasonal = (int(P),int(D),int(Q),int(s))
        TREND = [int(a1>0.5),int(a2>0.5),int(a3>0.5),int(a4>0.5)]
        train = data
        t=np.reshape(np.array(train).astype(float),-1,1)
        mod = sm.tsa.statespace.SARIMAX(t,order=param,seasonal_order=param_seasonal, enforce_stationarity=True,enforce_invertibility=True)
        results = mod.fit()
        pred0 = results.get_prediction(start= 130,end = 159, dynamic=False)
        pred0_ci = pred0.conf_int()
        y_pred = pred0.predicted_mean
        plt.plot(pred0.predicted_mean, color=plt.cm.Reds(int(colors[i])), alpha=.9)
        mf, smape, rmse = metrics(data[130:],y_pred)
        RMSE.append(rmse)
        SMAPE.append(smape)
        MF.append(mf)
        test = np.array(data[130:]).reshape(-1,1)
        y_pred=y_pred.reshape(-1,1)
        RESID.append(np.subtract(test, y_pred))
        
        y.append(y_pred)
    plt.grid(linestyle='dashed')
    plt.plot(np.array(data[130:]),label="Original dataset",linewidth=3)
    plt.xlim((0, 30))
    plt.ylim((-0.1, 1.1))
    
    y=np.array(y)
    y_max = []
    y_min = []  
    for i in range(y.shape[1]):
        Y_max,Y_min = confIntMean(np.array(y[:,i]),0.68)
        y_max.append(Y_max)
        y_min.append(Y_min) 
    return np.array(MF),np.array(SMAPE),np.array(RMSE),y_max,y_min #,RESID

import numpy as np, scipy.stats as st
def confIntMean(a, conf=0.95):
    mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
    return mean - m*sem, mean + m*sem


def NestedCrossVal(data,k,Val_Size,Test_Size):
    data = np.array(data)
    train,val = [],[]
    batchsize = int((len(data)-Test_Size)/k)
    for i in range(1,1+k):
        train.append(data[:i*batchsize])
        val.append(data[(i*batchsize-Val_Size):i*batchsize])
    return np.array(train) #,np.array(val)

def metrics(testY,y_pred):
    
    testY=testY.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    resid = np.subtract(testY, y_pred)

    RMSE = abs(resid).mean()
    mean = RMSE  
    r = r2_score(testY, y_pred[:len(testY)])
    smape=0
    for h in range(len(resid)):
        smape=smape+abs(resid[h])/((y_pred[h]+testY[h])/2)
    smape = (smape/len(resid))*100 
    MF = sum(np.power(resid,2))/sum(np.power(testY,2))*100
 
    return  MF, smape,np.array(mean)
