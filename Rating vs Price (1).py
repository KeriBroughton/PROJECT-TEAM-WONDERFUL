#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


dataset1 = pd.read_csv('appstore_games.csv')


# In[65]:


dataset2 = pd.read_csv('subgenre_dummies.csv')


# In[66]:


dataset = dataset1.loc[(
    dataset1["Price"] < 10)]


# In[67]:


dataset.shape


# In[68]:


dataset.isnull().any()


# In[69]:


dataset = dataset.fillna(method='ffill')
# this fills all the null variables


# In[70]:


dataset.describe()


# In[83]:


Price0 = dataset["Price"] == 0

price_list=print(dataset["Price"].unique())

temp_dat0 = dataset[(dataset["Price"]==0)]
temp_dat1 = dataset[(dataset["Price"]==0.99)]
temp_dat2 = dataset[(dataset["Price"]==1.99)]
temp_dat3 = dataset[(dataset["Price"]==2.99)]
temp_dat4 = dataset[(dataset["Price"]==3.99)]
temp_dat5 = dataset[(dataset["Price"]==4.99)]
temp_dat6 = dataset[(dataset["Price"]==5.99)]
temp_dat7 = dataset[(dataset["Price"]==6.99)]
temp_dat8 = dataset[(dataset["Price"]==7.99)]
temp_dat9 = dataset[(dataset["Price"]==8.99)]
temp_dat10 = dataset[(dataset["Price"]==9.99)]

price0 = temp_dat0["Average User Rating"].mean()
price1 = temp_dat1["Average User Rating"].mean()
price2 = temp_dat2["Average User Rating"].mean()
price3 = temp_dat3["Average User Rating"].mean()
price4 = temp_dat4["Average User Rating"].mean()
price5 = temp_dat5["Average User Rating"].mean()
price6 = temp_dat6["Average User Rating"].mean()
price7 = temp_dat7["Average User Rating"].mean()
price8 = temp_dat8["Average User Rating"].mean()
price9 = temp_dat9["Average User Rating"].mean()
price10 = temp_dat10["Average User Rating"].mean()

prices = [0, 0.99, 1.99, 2.99, 3.99, 4.99, 5.99, 6.99, 7.99, 8.99, 9.99]
rtgs = [price0, price1, price2, price3, price4, price5, price6, price7, price8, price9, price10]

# plt.plot(prices, rtgs, 'o') 
# plt.title('Price vs Average User Rating')  
# plt.xlabel('prices')  
# plt.ylabel('rtgs')  
# plt.show()


# In[72]:


dataset.plot(x='Price', y='Average User Rating', style='o')  
plt.title('Price vs Average User Rating')  
plt.xlabel('Price')  
plt.ylabel('Average User Rating')  
plt.show()
# this is a scatter plot of Price and Average User Rating. Since both of the values are fixed, it looks odd.


# In[73]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Average User Rating'])
# this is a bar chart to show that most games were rated a 4.5. Most of the data shows that games were rated between a 4 and 5.


# In[74]:


X = dataset['Price'].values.reshape(-1,1)
y = dataset['Average User Rating'].values.reshape(-1,1)


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# we are taking 20 percent of our data to predict how it looks compared to our actual data so that we can see how confident our model is
# we don't need to use all of the data to predict, since we need to have some room for error.
# 20 percent is the standarad around the industry


# In[76]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[77]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[78]:


y_pred = regressor.predict(X_test)


# In[79]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
#this is our dataframe which shows what was predicted and what the actual value of our variable was.


# In[80]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#this is a bar chart that shows our predicted value and actual value


# In[84]:


plt.scatter(prices, rtgs,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
#This is our linear regression plot.
# It shows that the higher the price, the lower the rating


# In[82]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# this is what we will use to see how trusting our data is.
# Root Mean Squared is probably the best formula to use, and it shows that we expect our expected rating to fall between .77 of the actual rating.
# That shows about a 14 percent margin of error, which means the data we have is not a good predicter of Rating


# In[ ]:





# In[ ]:




