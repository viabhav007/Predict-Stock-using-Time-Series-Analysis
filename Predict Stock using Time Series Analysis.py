#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Vaibhav\Desktop\BAJFINANCE.csv")
df


# In[3]:


df.set_index('Date', inplace=True)


# In[4]:


df['VWAP'].plot()
plt.show()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


#computing the mean of NaN values
nan_col = ['Trades', 'Deliverable Volume', '%Deliverble']
mean = df[nan_col].mean()
df.loc[:, nan_col] = df[nan_col].fillna(mean)


# In[8]:


df.head(5)


# In[9]:


data = df.copy()


# In[10]:


lag_features = ['High', 'Low', 'Volume', 'Turnover', 'Trades']
window1 = 3
window2 = 7


# In[11]:


for feature in lag_features:
    data[feature + 'rolling_mean_3'] = data[feature].rolling(window=window1).mean()
    data[feature + 'rolling_mean_7'] = data[feature].rolling(window=window2).mean()


# In[12]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[13]:


data.head(3)


# In[14]:


data.shape


# In[15]:


data.isna().sum()


# In[16]:


data.dropna(inplace=True)


# In[17]:


data.isna().sum()


# In[18]:


data.columns


# In[19]:


ind_features = ['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[20]:


training_data=data[0:4500]
test_data=data[4500:]


# In[21]:


training_data


# In[22]:


get_ipython().system('pip install pmdarima')


# In[23]:


from pmdarima import auto_arima
print(training_data[ind_features].dtypes)


# In[24]:


import warnings
warnings.filterwarnings('ignore')


# In[25]:


model = auto_arima(y=training_data['VWAP'], X=training_data[ind_features], trace=True)


# In[26]:


model.fit(training_data['VWAP'], training_data[ind_features])


# In[27]:


forecast = model.predict(n_periods=len(test_data), X=test_data[ind_features])


# In[28]:


test_data["Forecast_ARIMA"] = forecast


# In[30]:


test_data["Forecast_ARIMA"] = forecast
test_data[['VWAP', 'Forecast_ARIMA']].plot(figsize=(14, 7))
plt.title('VWAP vs. Forecast_ARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(['VWAP', 'Forecast_ARIMA'])
plt.show()


# In[ ]:




