#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(heart.drop(columns=['HeartDisease'], axis=1))


# In[25]:


trans_heart = scaler.transform(heart.drop(columns=['HeartDisease'], axis=1))
trans_heart = pd.DataFrame(trans_heart, columns=heart.columns[:-1])
trans_heart.head()


# In[26]:


X = trans_heart.copy()
y = heart[['HeartDisease']]
y.head()


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train.values.ravel())


# In[29]:


predictions = neigh.predict(X_test)


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[31]:


print(classification_report(y_test, predictions))
print("Accuracy : ", accuracy_score(y_test, predictions))


# In[32]:


print(confusion_matrix(y_test, predictions))


# #### Finding the optimal 'k' values

# In[33]:


kvalues = []
error = []
for i in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train.values.ravel())
    predictions_1 = neigh.predict(X_test)
    error.append(1-accuracy_score(y_test, predictions_1))
    kvalues.append(i)


# In[34]:


sns.lineplot(x=kvalues, y=error)
plt.title('error_rate vs kvalue')
plt.xlabel('kvalues')
plt.ylabel('error_rate')


# #### So as we can see the best k value using the elbow method can be determined to be 3

# In[35]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train.values.ravel())


# In[36]:


predictions = neigh.predict(X_test)
print(classification_report(y_test, predictions))
print("Accuracy : ", accuracy_score(y_test, predictions))


# In[37]:


print(confusion_matrix(y_test, predictions))

