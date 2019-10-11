#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('data.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.isnull().any()


# In[5]:


dataset = dataset.fillna(method='ffill')


# In[7]:


X = dataset[['Action', 'Sports', 'Utilities']].values
y = dataset['Average User Rating'].values


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[11]:


coeff_df = pd.DataFrame(regressor.coef_,['Action', 'Sports', 'Utilities'])  
coeff_df
# Our coeffecients are much better using this data


# In[15]:


y_pred = regressor.predict(X_test)


# In[16]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(30)


# In[17]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# With the Root Mean Squared Error is better as it is just outside the 10 percent of the mean value. It's not the most accurate, but it can make reasonably good predictions.


# In[18]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




