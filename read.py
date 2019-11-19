# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:47:11 2019

@author: guint
"""

import read
import pandas as pd
from pandas.plotting import scatter_matrix
from string import ascii_letters
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.interpolate import spline



def scale(x,a,b,Li,Ls,Nom):
    x_min=x.min()
    x_max=x.max()
    if x_min<0:
        x   =  x   - x_min
        Li  =  Li  - x_min
        Ls  =  Ls  - x_min
        Nom =  Nom - x_min    
        x_min=x.min()
        x_max=x.max()
    Li = (Li-((x_min)**2)**0.5)/(x_max-x_min)
    Ls = (Ls-((x_min)**2)**0.5)/(x_max-x_min)
    x  = (x-((x_min)**2)**0.5)/(x_max-x_min)
    Nom = (Nom-((x_min)**2)**0.5)/(x_max-x_min)
    return x*(b-a)+a,Li,Ls,Nom


def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)



def Import():
    U79_TCT_113 = pd.read_excel('FAROL  U79 TCT 113Cotas  (TAG3).xlsm', 'evo_wX' )
    week = U79_TCT_113.iloc[2,:]
    Head = U79_TCT_113.iloc[4:,0:5]
    Head.columns = ['Zone','Name','Nominal','LI','LS']
    Head.index = range(Head.shape[0])
    
    #Formata os dados e descarta os valores invaldos
    U79_TCT_113 = U79_TCT_113.iloc[4:,1:]
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
    
    #divide o dataset em treino teste e validação
    U79_TCT_113_GnF_train =   U79_TCT_113_GnF.iloc[:int(U79_TCT_113_GnF.shape[0]*0.7),:] 
    U79_TCT_113_GnF_test =  U79_TCT_113_GnF.iloc[int(U79_TCT_113_GnF.shape[0]*0.7):int(U79_TCT_113_GnF.shape[0]*0.9),:]
    U79_TCT_113_GnF_validation =  U79_TCT_113_GnF.iloc[int(U79_TCT_113_GnF.shape[0]*0.9):,:]
    
    return U79_TCT_113_GnF, Head
    
def FindCorr(index,U79_TCT_113_GnF,Head): 
    k=index # index of the selected variable
    min_corr = np.empty(len(U79_TCT_113_GnF.iloc[1,:]))
    max_corr = np.empty(len(U79_TCT_113_GnF.iloc[1,:]))
    max_corr.fill(0.2)
    min_corr.fill(-0.1)
    correlation = []
    for i in range(U79_TCT_113_GnF.iloc[:,:].shape[1]):
        correlation.append(U79_TCT_113_GnF.iloc[:,k].astype(float).corr(U79_TCT_113_GnF.iloc[:,i].astype(float))) #done again to maintem all elements
    above = np.argwhere(correlation > max_corr)
    bellow = np.argwhere(correlation < min_corr)
    corr = np.array(correlation)
    above_val = corr[above]
    bellow_val = corr[bellow]
    maxvalues = sorted(above_val)[::-1]
    minvalues = sorted(bellow_val)
    minvalues = minvalues[0:3]
    maxvalues = maxvalues[1:3]
    
    index = []
    for i in range(3):
        index.append(np.argwhere( corr == minvalues[i])) 
    for i in range(2):
        index.append(np.argwhere( corr == maxvalues[i]))     
    index.append(k)
    ind = np.reshape(np.array(index),-1,1)
    U79_TCT_113_5var = U79_TCT_113_GnF.iloc[:,ind] # estas são as 3 variaveis com maior correlação negativa e duas com a maior correlação direta
    
    veicular_zone = []
    for i in range(np.array(U79_TCT_113_5var.columns).shape[0]):
        Index = int(np.argwhere(Head.iloc[:,1]  == np.array(U79_TCT_113_5var.columns)[i])[0])
        veicular_zone.append(Head.iloc[Index,0]) #zonas veicular das cotas
    

    return U79_TCT_113_5var

def OutlaierFilter(var_6):
    Var = np.array(var_6) 
    index_of_non_outlaiers=[]
    
    for i in range(Var.shape[1]):
        std=Var[:,i].std()
        m=Var[:,i].mean()
        first_list = list(np.reshape(np.argwhere(Var[:,i]<m+2*std),-1,1))
        second_list = list(np.reshape(np.argwhere(Var[:,i]>m-2*std),-1,1))
        index_of_non_outlaiers.append(list(set(first_list[:]).intersection(second_list[:])))
    return np.array(index_of_non_outlaiers)   

def normalize(FullDataframe,var_6,Head,index_of_non_outlaiers):
    U79_TCT_113_5var = var_6
    Var = FullDataframe
    LimI=[]
    LimS=[]
    Nominal=[]
    for i in range(var_6.shape[1]):
        var = np.array(Var.iloc[index_of_non_outlaiers[i],i])
        Index = int(np.argwhere(Head.iloc[:,1]  == np.array(U79_TCT_113_5var.columns)[i])[0])
        LS = (Head.iloc[Index,4])
        LI = (Head.iloc[Index,3])
        nominal = (Head.iloc[Index,2])
        var_6.iloc[index_of_non_outlaiers[i],i] , li, ls, NOMINAL = scale(var,0,1,LI,LS,nominal)     
                
        LimI.append(li)
        LimS.append(ls)
        Nominal.append(NOMINAL)
    return var_6,Nominal,LimI,LimS
       
def plotvar(dataframe,Head,index_of_non_outlaiers):
    
    for i in range(Var.shape[1]):
        Index = int(np.argwhere(Head.iloc[:,1]  == np.array(U79_TCT_113_5var.columns)[i])[0])
        LS = (Head.iloc[Index,4])
        LI = (Head.iloc[Index,3])
        nominal = (Head.iloc[Index,2])
        Var[index_of_non_outlaiers[i],i] , li, ls, NOMINAL = scale(Var[index_of_non_outlaiers[i],i],0,1,LI,LS,nominal)     
        nom = np.empty(Var[index_of_non_outlaiers[i],i].shape) 
        Ls = np.empty(Var[index_of_non_outlaiers[i],i].shape)
        Li = np.empty(Var[index_of_non_outlaiers[i],i].shape) 
        
        LimI.append(li)
        LimS.append(ls)
        Nominal.append(NOMINAL)
        
        Ls.fill(ls)
        Li.fill(li)
        nom.fill(NOMINAL)
        sample=np.array(Var[index_of_non_outlaiers[i],i])        
        ax1=plt.subplot(6,2,2*i+1)
        #ax1=plt.ylim(-2,2)
        ax1.plot(sample)
        ax1.plot(Ls,color = 'black')
        ax1.plot(Li,color = 'black')
        ax1.plot(nom,color = 'red') 
        ax1.set_ylim([-0.5, 1.5])
        ax1.set_title(str(np.array(U79_TCT_113_5var.columns)[i]))
        
        vert_hist=np.histogram(sample,bins=30)
        # Compute height of plot.
        height = math.ceil(max(vert_hist[1])) - math.floor(min(vert_hist[1]))
        # Compute height of each horizontal bar.
        height = height/len(vert_hist[0])
        ax2=plt.subplot(6,2,2*i+2)
        ax2.barh(vert_hist[1][:-1],vert_hist[0], height=height)
        ax2.set_ylim([-0.5, 1.5])
        ax2.set_title(str(np.array(U79_TCT_113_5var.columns)[i]+' Distribuition'))
 
def PreProcessing(FullDataframe,Head,var_6): 
    index_of_non_outlaiers = OutlaierFilter(var_6)
    dataset,Nom,Li,Ls = normalize(FullDataframe,var_6,Head,index_of_non_outlaiers)
    DATA = []
    for i in range(6):
        DATA.append(dataset.iloc[index_of_non_outlaiers[i],i])
    DATA = np.array(DATA)
    #D = []
    #for i in range(1,6,2):
        #D.append(DATA[i]) #selecionando apenas 3 series  
    return DATA,Nom,Li,Ls
       
