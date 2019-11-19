import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 
from sklearn.cluster import KMeans
import numpy as np, scipy.stats as st
from scipy.optimize import differential_evolution


def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        return G

    def _select_centers(self, X,Y):
        """ Random choose"""
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        """ linear equal distribuited """
        #centers = np.linspace(0, len(X), self.hidden_shape)
        """ K_means to choose centers """
        #print(X.shape)
        #time.sleep(5)
        kmeans = KMeans(n_clusters=self.hidden_shape, random_state=0).fit(X)
        kmeans.labels_
        centers = kmeans.cluster_centers_
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X,Y)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

def cost(parameters, *arg):
    #parameters = np.float64(100) ,np.float64(0.1),np.float64(7)
    hidden_shape, sigma, look_back = parameters
    look_back,hidden_shape = look_back.astype(int), hidden_shape.astype(int)
    k=5 #k folds in nested cross validation 
    data = arg
    data = np.reshape(np.array(data).astype(float),-1,1) 
    batch = NestedCrossVal(data,5,15,10)
    Cost = [] 
    for i in range(k):
        dataset = batch[i]
        train = dataset[:len(dataset)-15 ]
        trainX, trainY = create_dataset(train, look_back)
        trainX = trainX.reshape(trainX.shape[0],look_back)
        trainY = trainY.reshape(-1, 1)
        if hidden_shape>trainX.shape[0]:
            hidden_shape = int(trainX.shape[0]*0.8)         
        model = RBFN(hidden_shape=int(hidden_shape), sigma=sigma)
        model.fit(trainX, trainY)
        val = dataset[len(dataset)-15-look_back:]
        valX, valY = create_dataset(val, look_back)
        valY = valY.reshape(-1, 1)
        valX = valX.reshape(valX.shape[0],valX.shape[1])
        y_pred = model.predict(valX)
        mf, smape, m = metrics(valY,y_pred) 
        Cost.append(float(m))
    result = np.array(Cost).mean()*100  
    return result 

def optimize(data):
        #        hidden_shape    sigma  look_back
    bounds = [(15, 150), (0.01, 3), (3, 7)]
    args = data
    result = differential_evolution(cost, bounds, args=args,strategy='best1bin',maxiter = 10, popsize = 10  ,recombination = 0.1)
    return result.x,result.fun

def evaluate(DATA,param,serie):
    RMSE = []
    MF = []
    SMAPE = []
    data = DATA[:,serie]
    y=[]
    colors = np.linspace(start=100, stop=255, num=90) 
    for i in range(serie,90,3):
        hidden_shape,sigma,look_back = param[i]
        train = data[:130]
        train = np.array(train).astype(float)
        trainX, trainY = create_dataset(train, int(look_back))  
        model = RBFN(hidden_shape=int(hidden_shape), sigma=sigma)
        model.fit(trainX, trainY)   
        test = data[130-int(look_back):]
        test = np.array(test).astype(float)
        testX, testY = create_dataset(test, int(look_back)) 
        y_pred = model.predict(testX)
        plt.plot(y_pred, color=plt.cm.Reds(int(colors[i])), alpha=.9)
        mf, smape, rmse = metrics(testY,y_pred)
        RMSE.append(rmse)
        SMAPE.append(smape)
        MF.append(mf) 
        y.append(y_pred)
    plt.grid(linestyle='dashed')
    plt.plot(testY,label="Original dataset",linewidth=3)
    y=np.array(y)
    y_max = []
    y_min = []  
    for i in range(y.shape[0]):
        Y_max,Y_min = confIntMean(np.array(y[:,i]),0.975)
        y_max.append(Y_max)
        y_min.append(Y_min) 
    return np.array(MF),np.array(SMAPE),np.array(RMSE),y_max,y_min

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

    
