#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\AVINESH\Downloads\cardekhopred\cardekho.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().values.any()


# In[6]:


co = df.corr()
c = sns.heatmap(df.corr())


# In[7]:


final_dataset = df.iloc[:,1:]


# In[8]:


final_dataset.head()


# In[9]:


final_dataset['Current_year']=2020


# In[10]:


final_dataset.head()


# In[11]:


final_dataset['No_of_years'] = final_dataset['Current_year']-final_dataset['Year']


# In[12]:


final_dataset.head()


# In[13]:


final_dataset.drop(['Year'],inplace=True,axis=1)


# In[14]:


final_dataset.drop(['Current_year'],inplace=True,axis=1)


# In[15]:


final_dataset.head()


# In[16]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[17]:


final_dataset.head()


# In[18]:


final_dataset.corr()


# In[19]:


cor = final_dataset.corr()
g = sns.heatmap(final_dataset.corr())


# In[20]:


final_dataset.hist()


# In[21]:


X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[30]:


#reg = LinearRegression()

#reg = reg.fit(X,y)
#pred = reg.predict(X_test)
#r2_score= reg.score(X,y)

#print(r2_score)

reg = RandomForestRegressor()
reg.fit(X,y)
reg.score(X_test,y_test)


# In[31]:


import pickle 


# In[32]:


file = open('reg_model.pkl','wb')


# In[33]:


pickle.dump(reg,file)


# In[34]:


pip freeze


# In[ ]:




