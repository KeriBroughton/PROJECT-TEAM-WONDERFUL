#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


dataset1 = pd.read_csv('appstore_games.csv')


# In[84]:


dataset2 = pd.read_csv('subgenre_dummies.csv')


# In[157]:


dataset = dataset1.loc[(
    dataset1["Price"] < 10)]


# In[158]:


dataset.shape


# In[159]:


dataset.isnull().any()


# In[160]:


dataset = dataset.fillna(method='ffill')
# this fills all the null variables


# In[161]:


dataset.describe()


# In[171]:


dataset.plot(x='Price', y='Average User Rating', style='o')  
plt.title('Price vs Average User Rating')  
plt.xlabel('Price')  
plt.ylabel('Average User Rating')  
plt.show()
# this is a scatter plot of Price and Average User Rating. Since both of the values are fixed, it looks odd.


# In[162]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Average User Rating'])
# this is a bar chart to show that most games were rated a 4.5. Most of the data shows that games were rated between a 4 and 5.


# In[163]:


X = dataset['Price'].values.reshape(-1,1)
y = dataset['Average User Rating'].values.reshape(-1,1)


# In[164]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# we are taking 20 percent of our data to predict how it looks compared to our actual data so that we can see how confident our model is
# we don't need to use all of the data to predict, since we need to have some room for error.
# 20 percent is the standarad around the industry


# In[165]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[166]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[167]:


y_pred = regressor.predict(X_test)


# In[168]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
#this is our dataframe which shows what was predicted and what the actual value of our variable was.


# In[169]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#this is a bar chart that shows our predicted value and actual value


# In[170]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
#This is our linear regression plot.
# It shows that the higher the price, the lower the rating


# In[106]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# this is what we will use to see how trusting our data is.
# Root Mean Squared is probably the best formula to use, and it shows that we expect our expected rating to fall between .77 of the actual rating.
# That shows about a 14 percent margin of error, which means the data we have is not a good predicter of Rating


# In[ ]:




