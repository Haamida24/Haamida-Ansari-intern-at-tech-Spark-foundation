#!/usr/bin/env python
# coding: utf-8

# #  To Explore Supervised Machine Learning

# # 
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables. 
# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Dataset

# In[2]:


dataset = pd.read_csv( 'http://bit.ly/w-data')


# In[3]:


dataset.shape


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# # Preparing the Data

# In[7]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Training the Algorithm

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


print(regressor.intercept_)


# In[11]:


print(regressor.coef_)


# # Making Predictions

# In[12]:


y_pred = regressor.predict(X_test)


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# # Evaluating the Algorithm

# In[14]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # What will be predicted score if a student study for 9.25 hrs in a day?

# In[15]:


hours=9.25
print("no.of hours the student studied:",hours)
print("Predicted score for the student is :",regressor.predict(np.array(hours).reshape(1,-1))[0])

