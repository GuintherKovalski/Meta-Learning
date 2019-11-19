import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score 
from scipy.optimize import differential_evolution
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

#neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr = args
#parameters = 200,200,200,15,0.1,0.98,1e-6,0.2
#model = CreateModel(args)

def CreateModel(args):
    look_back = 8
    neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr = args
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = int(neurons_l1) , return_sequences = True, input_shape = (look_back,1)))
    regressor.add(Dropout(0.1))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = int(neurons_l2), return_sequences = True))
    regressor.add(Dropout(0.1))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = int(neurons_l3)))
    regressor.add(Dropout(0.1))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Compiling the RNN
    optimizer = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    #optimizer = keras.optimizers.SGD(lr=0.05, decay=5e-4, momentum=0.4, nesterov=True)
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')       
    return regressor
    
def cost(parameters, *arg):
    #preparing data
    DATA = arg
    #parameters = 60,59,60,15,0.1,0.7, 1e-3,0.8  
    #creating model
    neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr = parameters
    neurons_l1 ,neurons_l2, neurons_l3,batch= int(neurons_l1) ,int(neurons_l2), int(neurons_l3),int(batch)
    parameters = neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr
    model = CreateModel(parameters)  
    #preparing data
    #d = DATA[2]
    DATA  = np.reshape(np.array(DATA).astype(float),-1,1)
    k = 3
    look_back = 8
    Cost= []
    batchs = NestedCrossVal(DATA,k,15,30)
    #fit model
    
    for i in range(k):
        dataset = batchs[i]
        train = dataset[:len(dataset)-15 ]
        trainX, trainY = create_dataset(train, look_back)
        trainX = trainX.reshape(trainX.shape[0],look_back,1)
        trainY = trainY.reshape(-1, 1)
        model.fit(trainX, trainY, epochs = 30 , batch_size = 15) #100 32
        val = dataset[len(dataset)-15-look_back:]
        valX, valY = create_dataset(val, look_back)
        valY = valY.reshape(-1, 1)
        valX = valX.reshape(valX.shape[0],valX.shape[1],1)
        y_hat = model.predict(valX)
        rmse,smape,mf = metrics(valY,y_hat) 
        #plt.plot(valY,'blue')
        #plt.plot(y_hat,'red')
        Cost.append(float(rmse))
    result = np.array(Cost).mean()*100      
    return result

def metrics(testY,y_pred):
    testY=testY.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    resid = np.subtract(testY, y_pred[:len(testY)])
    RMSE = abs(resid).mean()
    mean = RMSE  
    r = abs(r2_score(testY, y_pred[:len(testY)]))
    smape=0
    for h in range(len(resid)):
        smape=smape+abs(resid[h])/((y_pred[:len(testY)]+testY[h])/2)   
    smape = (smape/len(resid))*100 
    smape = smape.mean()
    MF = sum(np.power(resid,2))/sum(np.power(testY,2))*100    
    return  np.array(mean),smape,MF # MF, smape,np.array(r) 

def NestedCrossVal(data,k,Val_Size,Test_Size):
    data = np.array(data)
    train,val = [],[]
    batchsize = int((len(data)-Test_Size)/k)
    for i in range(1,1+k):
        train.append(data[:i*batchsize])
        val.append(data[(i*batchsize-Val_Size):i*batchsize])
    return np.array(train) #,np.array(val)

def optimize(Data):
    #initial guess for variation of parameters
    #              nl1         ,nl2,         nl3,      batch,      Dp,         momentum,      decay,       lr          
    bounds = [  (10, 140),    (10, 140),    (10, 140),    (2,15),  (0.05, 0.15),  (0.7,0.98),  (5e-4,5e-5),  (0.05,0.6)]
    # reshape input to be [samples, time steps, features
    args = np.array(Data) #  trainX, trainY, TestX, TestY
    parameters = differential_evolution(cost, bounds, args=args,strategy='best1bin',maxiter = 1, popsize = 10  ,recombination = 0.20,tol=0.01,mutation = 0.5)
    return parameters.x


def evaluate(DATA,parameters,serie):
    RMSE = []
    MF = []
    SMAPE = []
    data = DATA[:,serie]
    look_back = 8
    train,test = data[:130],data[130-look_back:] 
    colors = np.linspace(start=100, stop=255, num=90) 
    y=[]
    for i in range(serie,90,3): 
        neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr = parameters[i]
        neurons_l1 ,neurons_l2, neurons_l3,batch= int(neurons_l1) ,int(neurons_l2), int(neurons_l3),int(batch)
        parameter = neurons_l1 ,neurons_l2, neurons_l3,batch,Dp,momentum,decay,lr
        model = CreateModel(parameter)  
        
        trainX, trainY = create_dataset(train, look_back)
        trainX = trainX.reshape(trainX.shape[0],look_back,1)
        trainY = trainY.reshape(len(trainY),1) 
        
        TestX, TestY = create_dataset(test,look_back)
        TestX = TestX.reshape(TestX.shape[0],TestX.shape[1],1)
        TestY = TestY.reshape(len(TestY),1)   
       
        model.fit(trainX, trainY, epochs = 200 , batch_size = 15) #100 32
        
        y_pred = model.predict(TestX)
        plt.plot(y_pred, color=plt.cm.Reds(int(colors[i])), alpha=.9)
        rmse,smape , mf = metrics(TestY,y_pred)
        RMSE.append(rmse)
        MF.append(mf)
        SMAPE.append(smape)  
        y.append(y_pred)
    
    plt.plot(y_pred, color=plt.cm.Reds(int(colors[i])), alpha=.9)
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


