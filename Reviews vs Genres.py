#!/usr/bin/env python
# coding: utf-8

# In[2]:


cd bootcampStuff/theClassRepo/GTATL201908DATA3/07-Project-1/1/Activities/


# In[130]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


dataset1 = pd.read_csv('data.csv')


# In[132]:


dataset2 = dataset1.loc[(
    dataset1["User Rating Count"] < 50000)]
dataset = dataset2.loc[(
    dataset2["User Rating Count"] < 10000)]


# In[133]:


# dataset.head()


# In[ ]:





# In[134]:


dataset.shape


# In[135]:


dataset.isnull().any()


# In[136]:


dataset = dataset.fillna(method='ffill')


# In[137]:


X = dataset[['Puzzle', 'Board', 'Entertainment', 'Action', 'Simulation', 'Card', 'Role Playing', 'Sports', 'Adventure', 'Word', 'Family', 'Casino', 'Casual', 'Business', 'Navigation', 'Lifestyle', 'Social Networking', 'Education', 'Reference', 'Utilities', 'Trivia', 'Health & Fitness', 'Books', 'Music', 'Racing']].values
y = dataset['User Rating Count'].values


# In[138]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['User Rating Count'])


# In[139]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[140]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[141]:


X


# In[142]:


coeff_df = pd.DataFrame(regressor.coef_,['Puzzle', 'Board', 'Entertainment', 'Action', 'Simulation', 'Card', 'Role Playing', 'Sports', 'Adventure', 'Word', 'Family', 'Casino', 'Casual', 'Business', 'Navigation', 'Lifestyle', 'Social Networking', 'Education', 'Reference', 'Utilities', 'Trivia', 'Health & Fitness', 'Books', 'Music', 'Racing'])  
coeff_df
# none of the coefficients are going to be particularly usable or accurate.


# In[143]:


y_pred = regressor.predict(X_test)


# In[144]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(30)


# In[145]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# With the Root Mean Squared Error, we see that it's not particularly useable since the error is expected to be within 22 percent


# In[146]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:





# In[ ]:




