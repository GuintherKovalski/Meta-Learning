import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from sklearn.svm import SVR
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score 

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

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
    return  MF, smape,np.array(mean),np.array(r) 

def cost(parameters, *arg):
    d = arg
    d = np.reshape(np.array(d).astype(float),-1,1)
    #parameters = 3.9,1e10,3,6,0.001,0.5,2,6.9
    K_index,  C,  coef0,degree,tol,epsilon,gamma,look_back = parameters
    look_back = int(look_back)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid'] #0 a 4
    Kernel = kernel[int(K_index)]  
    k=5 #k folds in nested cross validation 
    Cost1,Cost2,Cost3 = [],[],[] 
    batch = NestedCrossVal(d,5,15,10)
    for i in range(k):
        dataset = batch[i]
        train = dataset[:len(dataset)-15]
        trainX, trainY = create_dataset(train, look_back)
        trainX = trainX.reshape(trainX.shape[0],look_back)
        trainY = trainY.reshape(-1, 1) 
        svr = SVR(kernel= Kernel, C=int(C), coef0 = coef0, 
                  degree=int(degree),tol=tol, gamma=gamma,
                  max_iter = 200000,epsilon= 0.01,cache_size = 5)       
        model = svr.fit(trainX,trainY.ravel())     
        val = dataset[len(dataset)-15-look_back:]
        valX, valY = create_dataset(val, look_back)
        valY = valY.reshape(-1, 1)
        valX = valX.reshape(valX.shape[0],valX.shape[1])
        y_hat = model.predict(valX)
        mf, smape, rmse = metrics(valY,y_hat) 
        Cost1.append(float(rmse))
        #Cost2.append(smape)
        #Cost3.append(mf)
    #np.array(Cost1).mean()
    #np.array(Cost2).mean()
    #np.array(Cost3).mean() 
    result = np.array(Cost1).mean()*100
    return result

def optimize(Data):
    #initial guess for variation of parameters                 
    #        |Kernel_index|     C,     |   coef0   |   degree  |   tol      |   epsilon  |    gamma   |  look_back            
    bounds = [  (1, 3.9),   (1e4,1e10),    (0,3),     (1,6),  (0.0001, 0.001),(0.01,0.5),  (0.05,2),  (3,6.9)]

    args = np.array(Data) #  trainX, trainY, TestX, TestY
    parameters = differential_evolution(cost, bounds, args=args,strategy='rand2exp',maxiter = 10, popsize = 10  ,recombination = 0.20)
    return parameters.x

def evaluate(DATA,parameters,serie):
    RMSE = []
    MF = []
    SMAPE = []
    data = DATA[:,serie]
    #data = DATA[serie]
    y=[]
    colors = np.linspace(start=100, stop=255, num=90) 
    for i in range(serie,90,3): 
        Kernel_index,C,coef0,degree,tol,epsilon,gamma,look_back = parameters[i]
        look_back = int(look_back)
        train,test = data[:130],data[130-look_back:] 
        TestX, TestY = create_dataset(test,look_back)
        TestY = TestY.reshape(len(TestY), 1)
        trainX, trainY = create_dataset(train,look_back)
        trainY = trainY.reshape(len(trainY), 1)
        trainX = trainX.reshape(trainX.shape[0],look_back)
        kernel = ['linear', 'poly', 'rbf', 'sigmoid'] #1 a 5
        Kernel = kernel[int(Kernel_index)]
        svr = SVR(kernel= Kernel, C=int(C)/(2e8), coef0 = coef0, 
                  degree=int(degree),tol=tol, gamma=gamma,
                  max_iter = 2000000,epsilon= 0.1,cache_size = 5)               
        model = svr.fit(trainX,trainY.ravel())
        TestX = TestX.reshape(TestX.shape[0],TestX.shape[1])
        y_pred = model.predict(TestX)
        plt.plot(y_pred, color=plt.cm.Reds(int(colors[i])), alpha=.9)
        mf, smape, rmse = metrics(TestY,y_pred)
        RMSE.append(rmse)
        MF.append(mf)
        SMAPE.append(smape) 
        y.append(y_pred)
    plt.grid(linestyle='dashed')
    plt.plot(TestY,label="Original dataset",linewidth=3)
    
    y=np.array(y)
    y_max = []
    y_min = []  
    for i in range(y.shape[0]):
        Y_max,Y_min = confIntMean(np.array(y[:,i]),0.975)
        y_max.append(Y_max)
        y_min.append(Y_min) 
    return np.array(MF),np.array(SMAPE),np.array(RMSE),y_max,y_min

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
    return np.array(train) 

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





