import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score 
from matplotlib.pyplot import figure
import seaborn as sns
from scipy.optimize import differential_evolution
import time

def cost(parameters, *args):  
    #we have 3 parameters which will be passed as parameters and
    #"experimental" x,y which will be passed as data
    #alpha,beta,gamma, periodo = 
    #parameters = 0.1, 0.1,0.0001,4 
    #parameters = a
    alpha,beta,gamma, periodo = parameters
    d = args
    #d = data
    d = np.reshape(np.array(d).astype(float),-1,1)
    k = 5
    Cost = []
    batch = NestedCrossVal(d,5,15,10)
    for i in range(k):
        dataset = batch[i]
        train=np.reshape(np.array(dataset).astype(float),-1,1)
        val = dataset[len(dataset)-15:]
        
        ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods= int( periodo))
        ets_fit = ets_model.fit(smoothing_level= alpha, smoothing_slope= beta, smoothing_seasonal= gamma)
        if len(dataset)>145:
            y_hat_train = ets_fit.predict(start=0, end=len(dataset)+15)
            plt.plot(dataset)
            plt.plot(y_hat_train)
            plt.savefig('TESTE/'+str(time.time())+'.jpg')
            plt.cla()
            plt.clf()
        
        y_pred = ets_fit.predict(start=len(dataset)+1, end=len(dataset)+15)
        
        #for i in range(len(x)):
        #result += (a*x[i]**2 + b*x[i]+ c - y[i])**2  
        #y_pred = ets_fit.predict(start=131, end= 130+len(test))
        #t=np.reshape(np.array(train).astype(float),-1,1)
        #plt.plot(val)
        #plt.plot(y_pred)
        mf, smape, m = metrics(val,y_pred) 
        Cost.append(float(m))
    
    result = np.array(Cost).mean()*100     
    return result


#initial guess for variation of parameters



#producing "experimental" data 
def optimize(data):
    #             alpha     beta        gamma    periodo
    bounds = [(0.0001, 0.5), (0.0001, 0.3), (0, 0.01),(3, 6.9)]
    args = data
    result = differential_evolution(cost, bounds, args=args,strategy='rand2exp',maxiter = 5, popsize =10  ,recombination = 0.25)
    return result.x

def evaluate(DATA,se,serie):
    sns.set(rc={'figure.figsize':(7,5)})
    sns.set_style("whitegrid")
    RMSE = []
    MF = []
    SMAPE = [] 
    data = DATA[:,serie]
    colors = np.linspace(start=100, stop=255, num=90)
    y=[]
    RESID = []
    #fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    for i in range(serie,90,3):
        alpha,beta,gamma, periodo = se[i]
        train = data
        test = data[130:] 
        t =np.reshape(np.array(train).astype(float),-1,1)
        ets_model = ExponentialSmoothing(t, trend='add', seasonal='add', seasonal_periods= int( periodo))
        ets_fit = ets_model.fit(smoothing_level= alpha, smoothing_slope= beta, smoothing_seasonal= gamma)
        y_pred = ets_fit.predict(start=len(train), end= len(train)+len(test))    
        y_pred = ets_fit.predict(start=131, end= 130+len(test)) 
        #y_pred = ets_fit.predict(start=101, end= 130+len(test))
        plt.plot(y_pred, color=plt.cm.Reds(int(colors[i])), alpha=.9)
        mf, smape, m = metrics(test,y_pred)
        test=test.reshape(-1,1)
        
        y_pred=y_pred.reshape(-1,1)
        RESID.append(np.subtract(test, y_pred))
        RMSE.append(m)
        SMAPE.append(smape)
        MF.append(mf)
        y.append(y_pred)
        #ACC.append(ConfusionMatrix(testY,y_pred,LimI[i],LimS[i])) 

    plt.grid(linestyle='dashed')
    plt.plot(data[130:],label="Original dataset",linewidth=3)
    
    y=np.array(y)
    y_max = []
    y_min = []  
    for i in range(y.shape[0]):
        Y_max,Y_min = confIntMean(np.array(y[:,i]),0.975)
        y_max.append(Y_max)
        y_min.append(Y_min) 
    return np.array(MF),np.array(SMAPE),np.array(RMSE),y_max,y_min#,RESID

import numpy as np, scipy.stats as st
def confIntMean(a, conf=0.95):
    mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
    return mean - m*sem, mean + m*sem


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
 
    return  MF, smape , np.array(mean)

def NestedCrossVal(data,k,Val_Size,Test_Size):
    data = np.array(data)
    train,val = [],[]
    batchsize = int((len(data)-Test_Size)/k)
    for i in range(1,1+k):
        train.append(data[:i*batchsize])
        val.append(data[(i*batchsize-Val_Size):i*batchsize])
        
    #ex: k=5,Val_Size = 15,Test_Size = 10 
    #train1,val1 = data[:30],data[15:30]
    #train2,val2 = data[:60],data[45:60]
    #train3,val3 = data[:90],data[75:90]
    #train4,val4 = data[:120],data[105:120]
    #train5,val5 = data[:150],data[135:150]
    
    return np.array(train) #,np.array(val)

def save(Smooth_Exp_parameter,name):
    SE = np.array(Smooth_Exp_parameter)
    lista1 = []
    for i in range(4*3):
        lista1.append(SE[i][0])
    lista2 = []
    for i in range(12):
        lista2.append(SE[i][1])
    a=np.array(lista1)
    b=np.array(lista2)
    SEhead= ['alpha','beta','gamma','periodo','best']
    np.c_[a,b]
    SEdf = pd.DataFrame(np.c_[a,b], columns= SEhead)
    SEdf.to_csv(name,encoding='utf-8', index=False)
    
def findbest(Smooth_Exp_parameter):
    SE = np.array(Smooth_Exp_parameter)
    lista1 = []
    for i in range(4*3):
        lista1.append(SE[i][0])
    lista2 = []
    for i in range(12):
        lista2.append(SE[i][1])
    a=np.array(lista1)
    b=np.array(lista2)
    SEhead= ['alpha','beta','gamma','periodo','best']
    np.c_[a,b]
    SEdf = pd.DataFrame(np.c_[a,b], columns= SEhead)
    serie1 =[]
    for i in range(0,12,3):
        serie1.append(SEdf.iloc[i,4])
    best = np.median(np.array(serie1))
    index1 = int(np.argwhere(SEdf.iloc[:,4]==best))
    serie2 =[]
    for i in range(1,12,3):
        serie2.append(SEdf.iloc[i,4])
    best = np.median(np.array(serie2))
    index2 = int(np.argwhere(SEdf.iloc[:,4]==best))
    serie3 =[]
    for i in range(2,12,3):
        serie3.append(SEdf.iloc[i,4])
    best = np.median(np.array(serie3))
    index3 = int(np.argwhere(SEdf.iloc[:,4]==best))
    best =[]
    best.append(np.array(SEdf.iloc[index1,:3]))
    best.append(np.array(SEdf.iloc[index2,:3]))
    best.append(np.array(SEdf.iloc[index3,:3]))
    return np.array(best)
        
    
    