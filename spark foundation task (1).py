#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries of python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#uploading and reading scv file
url ="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)


# In[3]:


df.shape
df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


#checking if any value is null
df.isnull().sum()


# In[8]:


#vistualization
df.plot(x='Hours',y='Scores',color='Blue',style='*',markersize=10)
plt.title('Hours vs Percentage')
plt.xlable('Hours studied')
plt.ylable('Percentage Score')
plt.grid()
plt.show() 


# In[9]:


df.corr()


# In[10]:


df.head()


# In[11]:


x=df.iloc[: , :1].values
y=df.iloc[: , 1:].values


# In[12]:


x


# In[13]:


y


# In[14]:


#model building
from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x,y, test_size= 0.2, random_state=50)


# In[15]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[16]:


m= model.coef_
c= model.intercept_
line=m*x+c
plt.scatter(x_train , y_train , color = 'red')
plt.plot(x , line, color='green')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()


# In[17]:


m= model.coef_
c= model.intercept_
line=m*x+c
plt.scatter(x_test , y_test , color = 'red')
plt.plot(x , line, color='green')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()


# In[18]:


print(x_test)
y_pred = model.predict(x_test)


# In[19]:


y_test


# In[20]:


y_pred


# In[21]:


#compairing actual vs predicted
comp = pd.DataFrame({'Actual':[y_test],'Predicted':[y_pred]})
comp


# In[22]:


#testing with own data
Hours = 9.25
own_pred = model.predict([[Hours]])
print('the predicted score if a person studies for', Hours,'hours is', own_pred[0])


# In[23]:


#evaluating the model

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:





# In[ ]:




