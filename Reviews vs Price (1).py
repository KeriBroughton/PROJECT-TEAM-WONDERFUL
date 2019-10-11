#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset1 = pd.read_csv('appstore_games.csv')
dataset1.shape


# In[4]:


dataset2 = dataset1.loc[(
    dataset1["Price"] < 10)]
dataset3 = dataset2.loc[(
    dataset2["User Rating Count"] < 50000)]
dataset = dataset3.loc[(
    dataset3["User Rating Count"] < 10000)]
# we are setting our price to be less than $10.00, and our review count to fall between 1000 and 50,000 to get rid of outliers that would skew the data.


# In[5]:


dataset.shape


# In[6]:


dataset.isnull().any()


# In[7]:


dataset = dataset.fillna(method='ffill')
# this fills all the null variables


# In[8]:


dataset.describe()
# our table


# In[9]:


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

price0 = temp_dat0["User Rating Count"].mean()
price1 = temp_dat1["User Rating Count"].mean()
price2 = temp_dat2["User Rating Count"].mean()
price3 = temp_dat3["User Rating Count"].mean()
price4 = temp_dat4["User Rating Count"].mean()
price5 = temp_dat5["User Rating Count"].mean()
price6 = temp_dat6["User Rating Count"].mean()
price7 = temp_dat7["User Rating Count"].mean()
price8 = temp_dat8["User Rating Count"].mean()
price9 = temp_dat9["User Rating Count"].mean()
price10 = temp_dat10["User Rating Count"].mean()

prices = [0, 0.99, 1.99, 2.99, 3.99, 4.99, 5.99, 6.99, 7.99, 8.99, 9.99]
rcount = [price0, price1, price2, price3, price4, price5, price6, price7, price8, price9, price10]


# In[10]:


dataset.plot(x='Price', y='User Rating Count', style='o')  
plt.title('Price vs User Rating Count')  
plt.xlabel('Price')  
plt.ylabel('User Rating Count')  
plt.show()
# This is our scatter plot
# Because all price values are fixed it does look odd


# In[11]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['User Rating Count'])
# We can see that more users left reviews when the price was free


# In[12]:


X = dataset['Price'].values.reshape(-1,1)
y = dataset['User Rating Count'].values.reshape(-1,1)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# we are taking 20 percent of our data to predict how it looks compared to our actual data so that we can see how confident our model is
# we don't need to use all of the data to predict, since we need to have some room for error.
# 20 percent is the standarad around the industry


# In[14]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[15]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[16]:


y_pred = regressor.predict(X_test)


# In[17]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
#this is our dataframe which shows what was predicted and what the actual value of our variable was.


# In[18]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#this is a bar chart that shows our predicted value and actual value


# In[19]:


plt.scatter(prices, rcount,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
#This is our linear regression plot.
# It shows that the higher the price, the more reviews were left.


# In[141]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# this is what we will use to see how trusting our data is.
# Root Mean Squared is probably the best formula to use, and it shows that we expect our expected reviews to fall between 1,289 of the actual review count.
# That shows about a 12.89 percent margin of error, which means the data we have is not a good predicter of review count.


# In[ ]:





# In[ ]:




