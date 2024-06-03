#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data=pd.read_csv("D:\PyCharm Community Edition 2021.3.1\python projects\Data_Train.csv")


# In[3]:


train_data.head(7)


# In[4]:


train_data.info()


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data.shape


# In[7]:


### getting all the rows where we have missing value
train_data[train_data['Total_Stops'].isnull()]


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.isnull().sum()


# In[10]:


data=train_data.copy()


# In[11]:


data.head(2)


# In[12]:


data.dtypes


# In[13]:


def change_into_datetime(col):
    data[col]=pd.to_datetime(data[col])


# In[14]:


data.columns


# In[15]:


for feature in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(feature)


# In[16]:


data.dtypes


# In[17]:


data['Date_of_Journey'].min()


# In[18]:


data['Date_of_Journey'].max()


# In[19]:


## lets do Feature Engineering of "Date_of_Journey" & fetch day,month,year !
data['journey_day']=data['Date_of_Journey'].dt.day
data['journey_month']=data['Date_of_Journey'].dt.month
data['journey_year']=data['Date_of_Journey'].dt.year


# In[20]:


data.head(2)


# In[21]:


data.drop('Date_of_Journey',axis=1,inplace=True)


# In[22]:


data.head(2)


# In[23]:


## Lets try to clean Dep_Time & Arrival_Time & featurize it..
def extract_hour_min(df,col):
    df[col+'_hour']=df[col].dt.hour
    df[col+'_minute']=df[col].dt.minute
    df.drop(col,axis=1,inplace=True)
    return df.head(2)


# In[24]:


# Departure time is when a plane leaves the gate

extract_hour_min(data,'Dep_Time')


# In[25]:


### lets Featurize 'Arrival_Time' !

extract_hour_min(data,'Arrival_Time')


# In[26]:


## lets analyse when will most of the flights will take-off
### Converting the flight Dep_Time into proper time i.e. mid_night, morning, afternoon and evening.

def flight_dep_time(x):
    '''
    This function takes the flight Departure time 
    and convert into appropriate format.
    '''
    if ( x> 4) and (x<=8 ):
        return 'Early mrng'
    
    elif ( x>8 ) and (x<=12 ):
        return 'Morning'
    
    elif ( x>12 ) and (x<=16 ):
        return 'Noon'
    
    elif ( x>16 ) and (x<=20 ):
        return 'Evening'
    
    elif ( x>20 ) and (x<=24 ):
        return 'Night'
    else:
        return 'Late night'


# In[27]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')


# In[28]:


## lets use Cufflinks & plotly to make your visuals more interactive !
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[29]:


## Lets use Plotly interactive plots directly with Pandas dataframes, but First u need below set-up !

import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[30]:


cf.go_offline()


# In[31]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind='bar')


# In[32]:


def preprocess_duration(x):
    if 'h' not in x:
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x


# In[33]:


data['Duration']=data['Duration'].apply(preprocess_duration)


# In[34]:


data['Duration']


# In[35]:


data['Duration'][0].split(' ')


# In[36]:


data['Duration'][0].split(' ')[0]


# In[37]:


data['Duration'][0].split(' ')[1]


# In[38]:


data['Duration'][0].split(' ')[0][0:-1]


# In[39]:


data['Duration'][0].split(' ')[1][0:-1]


# In[40]:


int(data['Duration'][0].split(' ')[1][0:-1])


# In[41]:


data['Duration_hours']=data['Duration'].apply(lambda x:int(x.split(' ')[0][0:-1]))


# In[42]:


data['Duration_mins']=data['Duration'].apply(lambda x:int(x.split(' ')[1][0:-1]))


# In[43]:


data.head(2)


# In[44]:


data['Duration_tot_mins'] = data['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)


# In[45]:


data.head(2)


# In[46]:


sns.lmplot(x = "Duration_tot_mins", y = 'Price', data = data)


# In[47]:


data['Destination'].unique()


# In[48]:


data['Destination'].value_counts().plot(kind = 'pie')


# In[49]:


data[data['Airline']=='Jet Airways']


# In[50]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending = False)


# In[51]:


plt.figure(figsize = (15, 5))
sns.violinplot(y = 'Price', x = 'Airline', data = data)
plt.xticks(rotation = 'vertical')


# In[52]:


data.drop(columns=["Route","Additional_Info","journey_year","Duration_tot_mins"], axis= 1, inplace=True)
data.head(2)


# In[53]:


cat_col = [col for col in data.columns if data[col].dtype == 'object']


# In[54]:


num_col = [col for col in data.columns if data[col].dtype != 'object']


# In[55]:


data['Source'].unique()


# In[56]:


data['Source'].apply(lambda x : 1 if x == 'Banglore' else 0)


# In[57]:


for category in data['Source'].unique():
    data['Source_'+category] = data['Source'].apply(lambda x : 1 if x == category else 0)


# In[58]:


airlines  = data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[59]:


airlines


# In[60]:


dict1 = {key:index for index, key in enumerate(airlines, 0)}


# In[61]:


dict1


# In[62]:


data['Airline']=data['Airline'].map(dict1)


# In[63]:


data['Destination'].unique()


# In[64]:


data['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[65]:


data['Destination'].unique()


# In[66]:


dest = data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[67]:


dict2 = {key:index for index, key in enumerate(dest, 0)}


# In[68]:


dict2


# In[69]:


data['Destination']=data['Destination'].map(dict2)


# In[70]:


data.head(2)


# In[71]:


data['Total_Stops'].unique()


# In[72]:


stops = {'non-stop':1 ,'2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4 }


# In[73]:


data['Total_Stops'] = data['Total_Stops'].map(stops)
data.head(2)


# In[74]:


def plot(df,col):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    sns.distplot(df[col],ax=ax3,kde=False)


# In[75]:


plot(data, 'Price')


# In[76]:


data['Price'] = np.where(data['Price']>=35000, data['Price'].median(), data['Price'])


# In[77]:


plot(data, 'Price')


# In[78]:


data.drop(columns=['Source','Duration'],axis=1,inplace=True)

data.head(2)
# In[79]:


from sklearn.feature_selection import mutual_info_regression


# In[80]:


X =data.drop(['Price'], axis=1)


# In[81]:


y = data['Price']


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[83]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
model = regressor.fit(X_train, y_train)


# In[84]:


y_pred=model.predict(X_test)


# In[85]:


y_pred


# In[86]:


y_pred.shape


# In[87]:


len(X_test)


# In[88]:


get_ipython().system('pip install pickle')


# In[89]:


import pickle


# In[90]:


file=open(r'D:\PyCharm Community Edition 2021.3.1\python projects/rf_random.pkl','wb')


# In[91]:


pickle.dump(model, file)


# In[92]:


model=open(r'D:\PyCharm Community Edition 2021.3.1\python projects/rf_random.pkl','rb')


# In[93]:


forrest = pickle.load(model)


# In[94]:


forrest.predict(X_test)


# In[95]:


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs(y_true-y_pred)/y_true)*100


# In[96]:


mape(y_test, forrest.predict(X_test))


# In[97]:


def predict(ml_model):
    
    model=ml_model.fit(X_train,y_train)
    print('Training_score: {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('Predictions are : {}'.format(y_prediction))
    print('\n')
    
    from sklearn import metrics
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2_score: {}'.format(r2_score))
    print('MSE : ', metrics.mean_squared_error(y_test,y_prediction))
    print('MAE : ', metrics.mean_absolute_error(y_test,y_prediction))
    print('RMSE : ', np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    print('MAPE : ', mape(y_test,y_prediction))
    sns.distplot(y_test-y_prediction)


# In[98]:


predict(RandomForestRegressor())


# In[99]:


from sklearn.model_selection import RandomizedSearchCV


# In[100]:


reg_rf = RandomForestRegressor()


# In[101]:


np.linspace(start=1000,stop=1200,num=6)


# In[102]:


# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=1000,stop=1200,num=6)]

# Number of features to consider at every split
max_features=["auto", "sqrt"]

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]

# Minimum number of samples required to split a node 
min_samples_split=[5,10,15,100]


# In[103]:


# Create the grid or hyper-parameter space
random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split
    
}


# In[104]:


random_grid


# In[105]:


rf_Random=RandomizedSearchCV(reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[106]:


rf_Random.fit(X_train,y_train)


# In[107]:


### to get your best model..
rf_Random.best_params_


# In[108]:


pred2=rf_Random.predict(X_test)


# In[109]:


from sklearn import metrics
metrics.r2_score(y_test,pred2)


# In[ ]:




