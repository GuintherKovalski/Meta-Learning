# Meta-Learning
This is the repository from my undeargrad thesis.
We used an evolutionary algorithm (Differential Evolution) for hyper-parameter optimization of five time series forecasting models, as ilustrated in the image:

![alt text](https://github.com/GuintherKovalski/Meta-Learning/blob/master/IMAGES/FRAMEWK.PNG)

Each model was optimized 30 times, resulting in 30 different models:

<img src="https://github.com/GuintherKovalski/Meta-Learning/blob/master/IMAGES/interval.png" width="550" height="350">

From the 95% confidence interval for the mean we compute this plots:

<img src="https://github.com/GuintherKovalski/Meta-Learning/blob/master/IMAGES/optimization.png" width="550" height="350">

More details available at:

https://1drv.ms/u/s!ApdlapclXdxHtUwjAVYbPFle9fGe?e=Ba7eY0

For usage: 

1 - pip install -r requirements

2 - let a file named series.csv in the directory. Note that we used 3 time series, so small modifications will be necessary for single time series. 

3 - run

WARNING, for a 160 sample time series, using DE for an LSTM archtecture optimizaiton tooks rough one week. 





