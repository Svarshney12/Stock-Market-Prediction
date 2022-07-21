import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import plotly.express as px
# %matplotlib notebook
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from math import sqrt

import streamlit as st

start= '2011-02-02'
end= '2022-01-31'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)
# df.head()

## describing the data
st.subheader('Data from 2011-02-02 to 2022-01-31')
st.write(df.describe())

##Visualization 
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(14,7))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Avg')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(14,7))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart withm 100 Moving Avg & 200 Moving Avg')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'b',label='MA100')
plt.plot(ma200,'r', label='MA200')
plt.plot(df.Close,'g',label='Close')
plt.legend()
st.pyplot(fig)


##Spliting of data
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

## Transform data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


## we dont use training part because we already train our model and now we use it

## Load LSTM model
model=load_model('keras_model.h5')

## for predicting 1938th row closing price we need its upper 100 rows
## so we append it from training data
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)

## scale down testing data
input_data=scaler.fit_transform(final_df)


##Split testing data
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)


## Now predict the output
y_predicted=model.predict(x_test)

##again scale up predicted and test data
## find scale 
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Predicted vs original Value
st.subheader('Predicted Vs Original Values')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Orginal Close Price')
plt.plot(y_predicted,'r',label='Predicted Preice')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



##Performance Matrics
mean_abs_error=mae(y_test,y_predicted)
mean_squ_error=mse(y_test,y_predicted)
root_mean_squ_error=sqrt(mean_squ_error)
r2_score=r2s(y_test,y_predicted)

st.write("Mean Abs Error")
st.write(mean_abs_error)
st.write("Mean Square Error")
st.write(mean_squ_error)
st.write("Root Mean Square Error")
st.write(root_mean_squ_error)
st.write("R Sqaure Score")
st.write(r2_score)

