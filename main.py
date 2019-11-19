import numpy as np
import pandas as pd
import ARIMA
import RBF 
import SVR as SVM_SVR
import SE as Smooth_Exp
import RNN
import time

data = pd.read_csv('series.csv') 
DATA = np.array(data)
lims = pd.read_csv('lims.csv')

#otimização
RBF30,Smooth_Exp30,Arima30,SVR30,RNN30  = [],[],[],[],[]
RBF_parameters, Smooth_Exp_parameter, Arima_parameters, SVR_parameters, RNN_weights =[],[],[],[],[] 
RBF_time,Smooth_Exp_time,Arima_time,SVR_time,RNN_time = [],[],[],[],[]

for j in range(30): #30 iterações para validação estatistica da otimização
    for i in range(3): #3 séries temporais
        data = DATA[:,i]  
        start = time.time()
        Smooth_Exp_parameter.append(Smooth_Exp.optimize(data))
        end = time.time()
        Smooth_Exp_time.append(end - start)
        
for j in range(30):
    for i in range(3):
        data = DATA[:,i]         
        start = time.time()
        Arima_parameters.append(ARIMA.optimize(data))  
        end = time.time()
        Arima_time.append(end-start) 
        
for j in range(30):
    for i in range(3):
        data = DATA[:,i]         
        start = time.time()
        RBF_parameters.append(RBF.optimize(data))
        end = time.time()
        RBF_time.append(end-start)
        
for j in range(30):
    for i in range(3):
        data = DATA[:,i]         
        start = time.time()
        SVR_parameters.append(SVM_SVR.optimize(data))
        end = time.time()
        SVR_time.append(end-start)
        
for j in range(30):
    for i in range(3):
        data = DATA[:,i]         
        start = time.time()
        RNN_weights.append(RNN.optimize(data))  
        end = time.time()
        RNN_time.append(end-start)

#salvando os dados
rbf = np.array(RBF_parameters).reshape(90,3) 
rbfhead = ['hidden_shape','sigma','look_back']
rbfdf = pd.DataFrame(rbf, columns= rbfhead)

se = np.array(Smooth_Exp_parameter).reshape(90, 4)
sehead = ['alpha','beta','gama','lookback']
sefdf = pd.DataFrame(se, columns= sehead)

arima = np.array(Arima_parameters).reshape(90,11)
arimahead = ['p','d','q','P','D','Q','s','a1','a2','a3','a4']
arimadf = pd.DataFrame(arima, columns= arimahead)

svm = np.array(SVR_parameters).reshape(90,8)
svmhead = ['Kernel_index','C','coef0','degree','tol','epsilon','gamma','look_back']    
svmdf = pd.DataFrame(svm, columns= svmhead)

RNN = np.array(RNN_wights).reshape(1,8)
RNNhead = ['nl1','nl2','nl3','momentum','decay','lr']
RNNdf = pd.DataFrame(RNN, columns= RNNhead)

sefdf.to_csv('SE.csv',encoding='utf-8', index=False)
rbfdf.to_csv('RBF.csv',encoding='utf-8', index=False)
arimadf.to_csv('ARIMA.csv',encoding='utf-8', index=False)
svmdf.to_csv('SVR_SVM.csv',encoding='utf-8', index=False)
RNN.to_csv('RNN.csv',encoding='utf-8', index=False)

#carregando arquivos
se = pd.read_csv('SE.csv')
se = np.array(np.array(se)).reshape(90,4) 
rbf = pd.read_csv('RBF.csv')
rbf = np.array(rbf).reshape(90,3) 
arima = pd.read_csv('ARIMA.csv')
arima = np.array(arima).astype(float).reshape(90,11)
svr = pd.read_csv('SVR_SVM.csv')
svr = np.array(svr).astype(float).reshape(90,8)
rnn = pd.read_csv('RNN.csv')
rnn = np.array(rnn).astype(float).reshape(90,8)


#avaliando resultados
import SE as Smooth_Exp
import ARIMA
import RBF
import SVR as SVM_SVR
import RNN

MF_mean,SMAPE_mean,RMSE_mean,MF_var, SMAPE_var,RMSE_var, MF_std,SMAPE_std,RMSE_std, = [],[],[],[],[],[],[],[],[]
y_interval = []
for i in range(3):
    MF1, SMAPE1, RMSE1, y_min1, y_max1  =  Smooth_Exp.evaluate(DATA,se,i) # o ultimo argumento é o indicie da serie. 0=serie1, 1 = serie 2 ...
    MF2, SMAPE2, RMSE2, y_min2, y_max2  =  ARIMA.evaluate(DATA,arima,i)
    MF3, SMAPE3, RMSE3, y_min3, y_max3  =  RBF.evaluate(DATA,rbf,i) 
    MF4, SMAPE4, RMSE4, y_min4, y_max4  =  SVM_SVR.evaluate(DATA,svr,i)
    MF5, SMAPE5, RMSE5, y_min5, y_max5  =  RNN.evaluate(DATA,rnn,i)
    
    y_interval.append(y_min1)
    y_interval.append(y_max1)
    y_interval.append(y_min2)
    y_interval.append(y_max2)
    y_interval.append(y_min3)
    y_interval.append(y_max3)
    y_interval.append(y_min4)
    y_interval.append(y_max4)
    y_interval.append(y_min5)
    y_interval.append(y_max5)
    
    MF_mean.append([MF1.mean(),MF2.mean(),MF3.mean(),MF4.mean(),MF5.mean()])
    SMAPE_mean.append([SMAPE1.mean(),SMAPE2.mean(),SMAPE3.mean(),SMAPE4.mean(),SMAPE5.mean()])
    RMSE_mean.append([RMSE1.mean(),RMSE2.mean(),RMSE3.mean(),RMSE4.mean(),RMSE5.mean()])
    
    MF_var.append([MF1.var(),MF2.var(),MF3.var(),MF4.var(),MF5.var()])
    SMAPE_var.append([SMAPE1.var(),SMAPE2.var(),SMAPE3.var(),SMAPE4.var(),SMAPE5.var()])
    RMSE_var.append([RMSE1.var(),RMSE2.var(),RMSE3.var(),RMSE4.var(),RMSE5.var()])
   
    MF_std.append([MF1.std(),MF2.std(),MF3.std(),MF4.std(),MF5.std()])
    SMAPE_std.append([SMAPE1.std(),SMAPE2.std(),SMAPE3.std(),SMAPE4.std(),SMAPE5.std()])
    RMSE_std.append([RMSE1.std(),RMSE2.std(),RMSE3.std(),RMSE4.std(),RMSE5.std()])
     
   
import seaborn as sns 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

for i in range(int(len(y_interval)/2)): 
    y_min,y_max = np.array(y_interval[i]).reshape(-1),np.array(y_interval[i+1]).reshape(-1)
    sns.set(rc={'figure.figsize':(6,4)})
    sns.set_style("whitegrid")
    plt.fill_between(np.array(range(len(y_min))), np.array(y_max)*1.05, (np.array(y_min)-np.array(y_min)*0.05),facecolor='grey', interpolate=True)
    plt.plot(DATA[130:,2],linewidth=3)
    plt.rc('xtick',labelsize=2)
    plt.rc('ytick',labelsize=2)


#salvando os resultados em um dataframe    
head = ['model','SMAPE1','SMAPE2','SMAPE','RMSE1','RMSE2','RMSE','MF1','MF2','MF']
models = ['SE','ARIMA','RBF','SVR','LSTM']

result_mean = pd.DataFrame(models, columns= [head[0]]) #Cria o Data frame
for i in range(3):
    result_mean[head[i+1]] = MF_mean[i]
for i in range(3):
    result_mean[head[i+4]] = RMSE_mean[i]
for i in range(3):
    result_mean[head[i+7]] = SMAPE_mean[i]

result_std = pd.DataFrame(models, columns= [head[0]]) #Cria o Data frame
for i in range(3):
    result_std[head[i+1]] = MF_std[i]
for i in range(3):
    result_std[head[i+4]] = RMSE_std[i]
for i in range(3):
    result_std[head[i+7]] = SMAPE_std[i]

result_var = pd.DataFrame(models, columns= [head[0]]) #Cria o Data frame
for i in range(3):
    result_var[head[i+1]] = MF_var[i]
for i in range(3):
    result_var[head[i+4]] = RMSE_var[i]
for i in range(3):
    result_var[head[i+7]] = SMAPE_var[i]

#salvando os data frames em .CSV
result_mean.to_csv('RESULT.csv.csv',encoding='utf-8', index=False) 
result_std.to_csv('result_std.csv',encoding='utf-8', index=False)   
result_var.to_csv('result_var.csv',encoding='utf-8', index=False) 
  

#plotando os resultados   
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


Result = pd.read_csv('RESULT.csv')
Result = Result.iloc[:,1:]
#Result= pd.read_csv('result_mean.csv' ) 

plt.figure(1)
sns.set(rc={'figure.figsize':(7,5)})
sns.set_style("whitegrid")
ylen = 0.5
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.regplot(data=Result, x="SMAPE1", y="MF1", fit_reg=False, marker="+", color="skyblue")
p1=sns.regplot(data=Result, x="SMAPE1", y="MF1", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':200}) 
for i in range(5):
    p1.text(Result.iloc[i,1], Result.iloc[i,7]+ylen, Result.iloc[i,0], horizontalalignment='center', size='larger', color='black', weight='semibold')

sns.regplot(data=Result, x="SMAPE2", y="MF2", fit_reg=False, marker="+", color="red")
p1=sns.regplot(data=Result, x="SMAPE2", y="MF2", fit_reg=False, marker="o", color="red", scatter_kws={'s':200}) 
for i in range(5):
    p1.text(Result.iloc[i,2], Result.iloc[i,8]+ylen, Result.iloc[i,0], horizontalalignment='center', size='larger', color='black', weight='semibold')

sns.regplot(data=Result, x="SMAPE(%)", y="MF(%)", fit_reg=False, marker="+", color="yellow")
p1=sns.regplot(data=Result, x="SMAPE(%)", y="MF(%)", fit_reg=False, marker="o", color="yellow", scatter_kws={'s':200}) 
for i in range(5):
    p1.text(Result.iloc[i,3], Result.iloc[i,9]+ylen, Result.iloc[i,0], horizontalalignment='center', size='larger', color='black', weight='semibold')


########################################################################################
plt.figure(1)
sns.regplot(data=Result, x="RMSE1", y="MF1", fit_reg=False, marker="+", color="skyblue")
p1=sns.regplot(data=Result, x="RMSE1", y="MF1", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,4], Result.iloc[i,7]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')

sns.regplot(data=Result, x="RMSE2", y="MF2", fit_reg=False, marker="+", color="red")
p1=sns.regplot(data=Result, x="RMSE2", y="MF2", fit_reg=False, marker="o", color="red", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,5], Result.iloc[i,8]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')

sns.regplot(data=Result, x="RMSE", y="MF(%)", fit_reg=False, marker="+", color="yellow")
p1=sns.regplot(data=Result, x="RMSE", y="MF(%)", fit_reg=False, marker="o", color="yellow", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,6], Result.iloc[i,9]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')


########################################################################################
plt.figure(1)
sns.regplot(data=Result, x="RMSE1", y="SMAPE1", fit_reg=False, marker="+", color="skyblue")
p1=sns.regplot(data=Result, x="RMSE1", y="SMAPE1", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,4], Result.iloc[i,1]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')

sns.regplot(data=Result, x="RMSE2", y="SMAPE2", fit_reg=False, marker="+", color="red")
p1=sns.regplot(data=Result, x="RMSE2", y="SMAPE2", fit_reg=False, marker="o", color="red", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,5], Result.iloc[i,2]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')

sns.regplot(data=Result, x="RMSE", y="SMAPE(%)", fit_reg=False, marker="+", color="yellow")
p1=sns.regplot(data=Result, x="RMSE", y="SMAPE(%)", fit_reg=False, marker="o", color="yellow", scatter_kws={'s':100}) 
for i in range(5):
    p1.text(Result.iloc[i,6], Result.iloc[i,3]+ylen, Result.iloc[i,0], horizontalalignment='center', size='medium', color='black', weight='semibold')

from scipy import stats
stats.f_oneway(RMSE1,RMSE2,RMSE3)
stats.ttest_ind(RMSE1, RMSE3, axis=0, equal_var=True)

collectn_1 = RMSE1
collectn_2 = RMSE2
collectn_3 = RMSE3
collectn_4 = RMSE4 
collectn_5 = RMSE5 

## combine these different collections into a list    
data_to_plot = [collectn_1, collectn_2, collectn_3,collectn_4, collectn_5]

fig = plt.figure(1, figsize=(5, 5))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)
ax.set_xticklabels(['SE', 'ARIMA', 'RBF', 'SVR','LSTM'], fontsize = 13)
plt.ylabel('RMSE',fontsize = 13)
plt.ylim(0.075,0.25)
plt.rc('xtick',labelsize=0.5)
plt.rc('ytick',labelsize=0.5)

############################################

from scipy import stats
stats.f_oneway(RMSE1,RMSE2,RMSE3)
stats.ttest_ind(RMSE1, RMSE3, axis=0, equal_var=True)
RESID = RESID1
data_to_plot = []
for i in range(len(RESID)):
    plt.plot(RESID[i])

#SE Serie1
#ARIMA Serie 2
#RBF Serie 3


plt.plot(RESID[i])
      
data_to_plot = RESID
collectn_1 = RMSE1
collectn_2 = RMSE2
collectn_3 = RMSE3
collectn_4 = RMSE4 
collectn_5 = RMSE5 
data_to_plot = [collectn_1, collectn_2, collectn_3,collectn_4, collectn_5]
fig = plt.figure(1, figsize=(5, 5))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)
ax.set_xticklabels(['SE', 'ARIMA', 'RBF', 'SVR','LSTM'], fontsize = 13)
plt.ylabel('RMSE',fontsize = 13)


plt.ylim(0.075,0.25)
plt.rc('xtick',labelsize=0.5)
plt.rc('ytick',labelsize=0.5)

#RESID = RESID1
import seaborn as sns

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
 
x,y =  create_dataset(DATA[:,1],4)
sns.distplot(RESID[k])
plt.xlabel('Residual size',fontsize = 13)
plt.ylabel('Residual counts',fontsize = 13)
    
plt.plot(RESID[k])
plt.ylabel('Residual amplitude',fontsize = 13)
plt.xlabel('Week',fontsize = 13)

import statsmodels.api as sm
sm.graphics.tsa.plot_acf(RESID[k],lags=25)
np.array(abs(RESID[k])).sum()/(len(RESID[k]))

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white

df_resid = pd.DataFrame(RESID[k],columns = ['resid'])
X= np.concatenate((np.ones(30).reshape(-1,1),x[-30:,:]),axis = 1)
white = het_white(list(df_resid.resid), X )
breuschpagan = het_breuschpagan(list(df_resid.resid), X )

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

statecrime_df = sm.datasets.statecrime.load_pandas().data
f ='violent~hs_grad+poverty+single+urban'
statecrime_model = ols(formula=f, data=statecrime_df).fit()
white_test = het_white(np.array(statecrime_model.resid)[-30:], x[-30:,:] )
np.array(statecrime_model.model.exog)[-30:,:]




