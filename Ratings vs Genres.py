#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


dataset = pd.read_csv('data.csv')


# In[13]:


dataset.shape


# In[14]:


dataset.isnull().any()


# In[15]:


dataset = dataset.fillna(method='ffill')


# In[17]:


X = dataset[['Puzzle', 'Board', 'Entertainment', 'Action', 'Simulation', 'Card', 'Role Playing', 'Sports', 'Adventure', 'Word', 'Family', 'Casino', 'Casual', 'Business', 'Navigation', 'Lifestyle', 'Social Networking', 'Education', 'Reference', 'Utilities', 'Trivia', 'Health & Fitness', 'Books', 'Music', 'Racing']].values
y = dataset['Average User Rating'].values


# In[18]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Average User Rating'])


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[21]:


coeff_df = pd.DataFrame(regressor.coef_,['Puzzle', 'Board', 'Entertainment', 'Action', 'Simulation', 'Card', 'Role Playing', 'Sports', 'Adventure', 'Word', 'Family', 'Casino', 'Casual', 'Business', 'Navigation', 'Lifestyle', 'Social Networking', 'Education', 'Reference', 'Utilities', 'Trivia', 'Health & Fitness', 'Books', 'Music', 'Racing'])  
coeff_df
# It looks like we actually have several coefficients that are usable in this data!
# Action, Utilities, and Sports have the best coefficients to use.


# In[33]:


y_pred = regressor.predict(X_test)


# In[ ]:





# In[34]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(30)


# In[37]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# With the Root Mean Squared Error, we see that it's not particularly useable since the error is expected to be within 22 percent


# In[38]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




