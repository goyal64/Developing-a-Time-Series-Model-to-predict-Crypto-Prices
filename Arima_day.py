#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:32:37 2019

@author: shubhamsharma
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import re

#importing packages for the prediction of time-series data
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

myfile = open('dat_coin.csv',"r")
data = myfile.readlines()


rows = []
for line in data:
    data_line = line.rstrip().split('\n')
    data_line = data_line[0].split(',')
    rows.append(data_line)


for i in range(1,len(rows)):
    rows[i][0] = str(datetime.utcfromtimestamp(int(rows[i][0])).strftime('%Y-%m-%d %H:%M:%S'))
    
    
for i in range(2,len(rows)):
    
    prerow = rows[i-1]
    currow = rows[i]
    
    if currow[1] == 'NaN':
        currow[1] = prerow[1]
        currow[2] = prerow[2]
        currow[3] = prerow[3]
        currow[4] = prerow[4]
        currow[5] = prerow[5]
        currow[6] = prerow[6]
        currow[7] = prerow[7]
        

hourlyrows = []
hourlyrows.append(rows[0])
for i in range(1,len(rows)):
    if str(rows[i][0]).endswith('00:00'):
        hourlyrows.append(rows[i])
        

output_scrap = open('datatimed.csv','w')
output_scrap.write(hourlyrows[0][0] +','+hourlyrows[0][1]+','+hourlyrows[0][2]+','+hourlyrows[0][3]+','+hourlyrows[0][4]+','+hourlyrows[0][5]+','+hourlyrows[0][6]+','+hourlyrows[0][7]+ '\n')

for i in range(1,len(hourlyrows)):
    
    output_scrap.write(hourlyrows[i][0] +','+hourlyrows[i][1]+','+hourlyrows[i][2]+','+hourlyrows[i][3]+','+hourlyrows[i][4]+','+hourlyrows[i][5]+','+hourlyrows[i][6]+','+hourlyrows[i][7]+ '\n')
output_scrap.close()

print("done witing the file as datatimed in the current working directory")


datatimed = pd.read_csv('datatimed.csv', index_col = 'Timestamp', parse_dates = True)

datatimed[['Open','Close']].plot.line()

datatimed.tail(100)

#ARIMA Model



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')



# Resampling to daily frequency
df_day = datatimed.resample('D').mean()

# Resampling to monthly frequency
df_month = datatimed.resample('M').mean()

# Resampling to annual frequency
df_year = datatimed.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_Q = datatimed.resample('Q-DEC').mean()

datatimed["Date"]=datatimed.index
data = datatimed['Weighted_Price']
Date1 = datatimed.index
train1 = datatimed[['Date','Weighted_Price']]
# Setting the Date as Index

train2 = train1.set_index('Date')
train2.sort_index(inplace=True)
print (type(train2))
print (train2.head())




plot.plot(train2)
plot.xlabel('Date', fontsize=12)
plot.ylabel('Price in USD', fontsize=12)
plot.title("Closing price distribution of bitcoin", fontsize=15)
plot.show()



from statsmodels.tsa.stattools import adfuller

def test_stationarity(x):


    #Determing rolling statistics
    rolmean = x.rolling(window=22,center=False).mean()

    rolstd = x.rolling(window=12,center=False).std()
    
    #Plot rolling statistics:
    orig = plot.plot(x, color='blue',label='Original')
    mean = plot.plot(rolmean, color='red', label='Rolling Mean')
    std = plot.plot(rolstd, color='black', label = 'Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show(block=False)
    
    #Perform Dickey Fuller test    
    result=adfuller(x)
    print('ADF Stastistic: %f'%result[0])
    print('p-value: %f'%result[1])
    pvalue=result[1]
    for key,value in result[4].items():
         if result[0]>value:
            print("The graph is non stationery")
            break
         else:
            print("The graph is stationery")
            break;
    print('Critical values:')
    for key,value in result[4].items():
        print('\t%s: %.3f ' % (key, value))
        
ts = train2['Weighted_Price']      
test_stationarity(ts)

ts_log = np.log(ts)
plot.plot(ts_log,color="green")
plot.show()

test_stationarity(ts_log)

ts_log_diff = ts_log - ts_log.shift()
plot.plot(ts_log_diff)
plot.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2,1,0))  
results_ARIMA = model.fit(disp=-1)  
plot.plot(ts_log_diff)
plot.plot(results_ARIMA.fittedvalues, color='red')
plot.title('RSS: %.7f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plot.show()



size = int(len(ts_log)-100)
# Divide into train and test

train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]

history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...')
print('\n')

# We go over each value in the test set and then apply ARIMA model and calculate the predicted value. We have the expected value in the test set therefore we calculate the error between predicted and expected value 
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)
    
    output = model_fit.forecast()
    
    pred_value = output[0]
    
        
    original_value = test_arima[t]
    history.append(original_value)
    
    pred_value = np.exp(pred_value)
    
    
    original_value = np.exp(original_value)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')
    
    predictions.append(float(pred_value))
    originals.append(float(original_value))
  
 
    
# After iterating over whole test set the overall mean error is calculated.   
print('\n Mean Error in Predicting Test Case Articles : %f ' % (sum(error_list)/float(len(error_list))), '%')
plot.figure(figsize=(8, 6))
test_day = [t
           for t in range(len(test_arima))]
labels={'Orginal','Predicted'}
plot.plot(test_day, predictions, color= 'green')
plot.plot(test_day, originals, color = 'orange')
plot.title('Expected Vs Predicted Views Forecasting')
plot.xlabel('Per Hour')
plot.ylabel('Weighted Price')
plot.legend(labels)
plot.show()



