import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


U79_TCT_113 = pd.read_excel('FAROL  U79 TCT 113Cotas  (TAG3).xlsm', 'evo_wX' )
#Extrai o cabeçalho 
week = U79_TCT_113.iloc[38,:]
Head = U79_TCT_113.iloc[4:,0:5]
Head.columns = ['Zone','Name','Nominal','LI','LS']
Head.index = range(Head.shape[0])
#Formata os dados e descarta os valores invaldos
U79_TCT_113 = U79_TCT_113.iloc[5:,1:]
U79_TCT_113 = U79_TCT_113.dropna(axis=1, how='all')
U79_TCT_113 = U79_TCT_113.T
U79_TCT_113.columns=U79_TCT_113.iloc[0,:]
U79_TCT_113 = U79_TCT_113.iloc[1:,:]
U79_TCT_113 = U79_TCT_113[::-1]
U79_TCT_113_W = U79_TCT_113.iloc[range(0,U79_TCT_113.shape[0],2),:]
U79_TCT_113_W.index = range(U79_TCT_113_W.shape[0])
U79_TCT_113_GnF = U79_TCT_113.iloc[range(1,U79_TCT_113.shape[0],2),:]
U79_TCT_113_GnF.index = range(U79_TCT_113_GnF.shape[0])
U79_TCT_113_GnF = U79_TCT_113_GnF.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False) #Remove as variaveis com np.nam
df  = U79_TCT_113_GnF
#divide o dataset em treino teste e validação

y = np.array(U79_TCT_113_GnF)[:,38].astype(float)


data = pd.read_csv('series.csv') 
DATA = np.array(data)
lims = pd.read_csv('lims.csv')

y = DATA[:,1]
dfy = pd.DataFrame(y, columns = ['value'])

import seaborn as sns
sns.set()

"""
Estatisticas basicas
"""
print(pd.DataFrame(y).describe())


"""
Plota a decomposição em series de fourier,
com o periodo invés de a frequência
"""

#y = y-y.mean()
signal = y
fourier = np.fft.rfft(signal)
n = signal.size
sample_rate = 1
freq = np.fft.fftfreq(n, d=1./sample_rate)
module= (np.real(y)**2+np.imag(y)**2)**0.5
angle= np.angle(y)

plt.figure(figsize=(16,4))   
periodo=1/freq[:int(len(module)/2)]
plt.plot(module[:int(len(module)/2)])
#plt.plot(angle[:int(len(angle)/2)])
plt.grid(linestyle='dashed')
plt.ylim(min(module[:int(len(module)/2)]),max(module[:int(len(module)/2)]))
plt.xlim(1,82)
plt.xticks(range(0,len(periodo),5), periodo[range(0,len(periodo),5)].round(decimals = 2)) 
periodo=1/freq[:int(len(module)/2)]
plt.xlabel("Period (1/F)")
plt.ylabel("Amplitude")
index = np.argwhere(max(module[:int(len(module)/2)])==module[:int(len(module)/2)])
print('period of max amp:',periodo[int(index)],'max amp:', module[index])
plt.savefig('fourier_spectrum.eps', format='eps', dpi=1000)

"""
Plota a autocorrelação da série
"""
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(16,4))  
autocorrelation_plot(y)
plt.grid(linestyle='dashed')
plt.ylim(-0.6,0.6)
plt.xlim(1,len(y))
#plt.xticks(range(0,len(periodo),20), periodo[range(0,len(periodo),20)]) 
#periodo=1/freq[:int(len(module)/2)]
plt.xlabel("Período")
plt.ylabel("Amplitude")
plt.savefig('autocorr.eps', format='eps', dpi=1000)


"""
Plota a série original, sazonalidade e tendencia
"""
decomposition = seasonal_decompose(y,freq=16)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,4))  
plt.grid(linestyle='dashed')
plt.plot(y, label='Original',color='blue')
plt.title("Original")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Original.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed') 
plt.plot(trend, label='Trend',color='blue')
plt.title("Trend")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('trend.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed')
plt.plot(seasonal,label='Seasonality',color='blue')
plt.title("Seasonality")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Seasonal.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed')
plt.plot(residual,label='Seasonality',color='blue')
plt.title("Residuals")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Seasonal.eps', format='eps', dpi=1000)

"""
Teste de Estacionariedade
"""
from statsmodels.tsa.stattools import adfuller
dfytest = adfuller(y, autolag='AIC')
dfyoutput = pd.Series(dfytest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfyoutput)

x = seasonal
x = x[np.logical_not(np.isnan(x))]
dfytest1 = adfuller(x, autolag='AIC')
dfyoutput1 = pd.Series(dfytest1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print('Dickey Fuller Test:\n',dfyoutput1)


"""
Filtro de Outlaiers
"""
index_of_non_outlaiers=[]
for i in range(y.shape[0]):
    delete = []
    std=y.std()
    m=y.mean()
    first_list = list(np.reshape(np.argwhere(y<m+2*std),-1,1))
    second_list = list(np.reshape(np.argwhere(y>m-2*std),-1,1))
    index_of_non_outlaiers.append(list(set(first_list).intersection(second_list)))
    
"""
Plota a série com filtros de lowess
"""
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

dfy_loess_5 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.05)[:, 1], index=dfy.index, columns=['value'])
dfy_loess_15 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.15)[:, 1], index=dfy.index, columns=['value'])
dfy_loess_30 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.3)[:, 1], index=dfy.index, columns=['value'])
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
dfy['value'].plot(ax=axes[0], color='k', title='Original Series')
dfy_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
dfy_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
dfy_loess_30['value'].plot(ax=axes[3], title='Loess Smoothed 30%')


"""
Plota a série com filtros de lowess
"""
plt.figure(figsize=(16,4)) 
dfy_ma = dfy.rolling(50, center=True)
plt.plot(dfy_ma.mean())
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()

"""
Autocorrelation by lags
"""
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Import
#ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
#a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(dfy.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))
fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)    
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(dfy.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))
fig.suptitle('Lag Plots of Drug Sales', y=1.05)    
plt.show()


"""
Partial Autocorrelation 
"""

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#dfy = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(dfy.value, nlags=50)
# pacf_50 = pacf(dfy.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(dfy.value.tolist(), lags=50, ax=axes[0])
plot_pacf(dfy.value.tolist(), lags=50, ax=axes[1])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

dfy_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
x = dfy_scaled
#dfy_scaled.to_csv('mim_max.csv')

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x[:-2].T)
plt.scatter(principalComponents[:,0],principalComponents[:,1])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(principalComponents)
cluster1 = np.argwhere(kmeans.labels_==1)
cluster2 = np.argwhere(kmeans.labels_==0)

c1 = principalComponents[cluster1,:]
c2 = principalComponents[cluster2,:]
C1 = pca.inverse_transform(c1) 
C2 = pca.inverse_transform(c2) 

m = principalComponents[cluster1,:].mean()
std = principalComponents[cluster1,:].std()
anomalies = principalComponents[cluster1,:] > m + 3*std

plt.figure(figsize=(8,8)) 
plt.grid(linestyle='dashed')
plt.scatter(principalComponents[:,0],principalComponents[:,1])
plt.scatter(principalComponents[38,0],principalComponents[38,1])
plt.scatter(principalComponents[47,0],principalComponents[47,1])
plt.scatter(principalComponents[0,0],principalComponents[47,1])
plt.legend(['pca','series 1','series 2','series 3'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig('PCA1.eps', format='eps', dpi=1000)


plt.figure(figsize=(8,8)) 
plt.grid(linestyle='dashed')
plt.scatter(principalComponents[cluster1,0],principalComponents[cluster1,1])
plt.scatter(principalComponents[cluster2,0],principalComponents[cluster2,1])
#plt.scatter(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1])
#plt.scatter(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1])
plt.legend(['K-means1','K-means2'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig('PCA2.eps', format='eps', dpi=1000)










