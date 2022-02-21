#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[13]:


from sklearn.model_selection import train_test_split
X = heart.drop(columns=['HeartDisease'], axis=1)
y = heart[['HeartDisease']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[15]:


dtree.fit(X_train, y_train)


# In[16]:


predictions = dtree.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(predictions, y_test))
print('Accuracy : ', accuracy_score(y_test, predictions))


# In[18]:


print(confusion_matrix(y_test, predictions))


# In[ ]:




