#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[2]:


data =pd.read_csv(r'C:\Users\gunja\Downloads\archive\TITAN.csv')
data.info()


# In[3]:


data["Close"] =pd.to_numeric(data.Close,errors='coerce')
data= data.dropna()
trainData = data.iloc[:,4:5].values
data.info()


# In[4]:


sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape


# In[5]:


x_train = []
y_train = []

for i in range (60,2456):
    x_train.append(trainData[i-60:i,0])
    y_train.append(trainData[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)


# In[6]:


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[7]:


model = Sequential()

model.add(LSTM(units=100,return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences = False, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer ='adam',loss="mean_squared_error")


# In[8]:


hist = model.fit(x_train, y_train, epochs =20, batch_size = 32, verbose=2)


# In[9]:


plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[10]:


testData = pd.read_csv(r'C:\Users\gunja\Downloads\archive\TITAN.csv')
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values

#input array for the model
inputClosing = testData.iloc[:,0:].values
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape

x_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length): 
    x_test.append(inputClosing_scaled[i-timestep:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[11]:


y_pred = model.predict(x_test)
y_pred


# In[12]:


predicted_price = sc.inverse_transform(y_pred)


# In[13]:


plt.plot(y_test, color = 'red', label = 'Actual stock price')
plt.plot(predicted_price, color = 'green', label = 'predicted stock price')
plt.title('google stock price predictions')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

