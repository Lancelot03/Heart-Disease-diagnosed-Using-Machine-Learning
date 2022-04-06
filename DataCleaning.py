#!/usr/bin/env python
# coding: utf-8

# In[38]:
import numpy as np
import pandas as pd

# In[39]:
heart = pd.read_csv('heart.csv')
heart.head()

# In[40]:
heart.info()

# In[28]:
heart.describe()

# In[29]:
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# In[30]:
sns.pairplot(data=heart, hue='HeartDisease')

# In[31]:
sns.boxplot(data=heart)


# #### Here we can see that "Cholestrol" and "RestingBP" have some 0 values in them.
# #### Which make no sense in an alive person
# #### Therefore here we replace these values with the means of the rest of the data for these specific columns

# In[32]:
heart[heart['Cholesterol']!=0].describe()

# In[42]:
heart[heart['RestingBP']!=0].describe()

# #### Replacing every 0 value of "Cholesterol" to 244.635389 & every 0 value of "RestingBP" to 132.540894

# In[34]:
heart.loc[heart['RestingBP'] == 0, 'RestingBP'] = 132.540894
heart.loc[heart['Cholesterol'] == 0, 'Cholesterol'] = 244.635389
sns.boxplot(data=heart)

# In[35]:
sns.pairplot(data=heart, hue="HeartDisease")

# ### Now replacing all the string values into numerical values
# #### We have 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina' and 'ST_Slope' as features with string values
# #### So for converting them into numerical variables we use get_dummies() method

# In[36]:
heart_modified = pd.get_dummies(heart)
heart_modified.head()

# In[37]:
heart_modified.to_csv('heart_modified.csv')

# ### Hence the data cleaning step is completed
