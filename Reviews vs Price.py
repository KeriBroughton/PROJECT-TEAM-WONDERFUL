#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


dataset1 = pd.read_csv('appstore_games.csv')
dataset1.shape


# In[126]:


dataset2 = dataset1.loc[(
    dataset1["Price"] < 10)]
dataset3 = dataset2.loc[(
    dataset2["User Rating Count"] < 50000)]
dataset = dataset3.loc[(
    dataset3["User Rating Count"] < 10000)]
# we are setting our price to be less than $10.00, and our review count to fall between 1000 and 50,000 to get rid of outliers that would skew the data.


# In[127]:


dataset.shape


# In[128]:


dataset.isnull().any()


# In[129]:


dataset = dataset.fillna(method='ffill')
# this fills all the null variables


# In[130]:


dataset.describe()
# our table


# In[131]:


dataset.plot(x='Price', y='User Rating Count', style='o')  
plt.title('Price vs User Rating Count')  
plt.xlabel('Price')  
plt.ylabel('User Rating Count')  
plt.show()
# This is our scatter plot
# Because all price values are fixed it does look odd


# In[132]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['User Rating Count'])
# We can see that more users left reviews when the price was free


# In[133]:


X = dataset['Price'].values.reshape(-1,1)
y = dataset['User Rating Count'].values.reshape(-1,1)


# In[134]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# we are taking 20 percent of our data to predict how it looks compared to our actual data so that we can see how confident our model is
# we don't need to use all of the data to predict, since we need to have some room for error.
# 20 percent is the standarad around the industry


# In[135]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[136]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[137]:


y_pred = regressor.predict(X_test)


# In[138]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
#this is our dataframe which shows what was predicted and what the actual value of our variable was.


# In[139]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#this is a bar chart that shows our predicted value and actual value


# In[140]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
#This is our linear regression plot.
# It does not show much, if any, correlation of the data collected.


# In[141]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# this is what we will use to see how trusting our data is.
# Root Mean Squared is probably the best formula to use, and it shows that we expect our expected reviews to fall between 1,289 of the actual review count.
# That shows about a 12.89 percent margin of error, which means the data we have is not a good predicter of review count.


# In[ ]:





# In[ ]:




