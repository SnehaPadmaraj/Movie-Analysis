#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure


# In[3]:


df = pd.read_csv(r'C:\Users\sneha\Downloads\movie\movies.csv')


# In[5]:


df.head()


# In[19]:


df = df.dropna()


# In[20]:


df.dtypes


# In[21]:


df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[22]:


df


# In[23]:


df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)


# In[24]:


df


# In[27]:


df.sort_values(by=['gross'], inplace = False, ascending = False)
df.head()


# In[36]:


#comparing independent variables like budget and company with dependent variable which is gross revenue 
#scatter plot with budget vs gross 
plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget')


# In[37]:


#positive value means that as budget increases, so does the gross revenue 
correlation = df['budget'].corr(df['gross'])
correlation


# In[38]:


sns.regplot(x="gross", y="budget", data=df)


# In[40]:


df.corr(numeric_only=True)


# In[48]:


correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot = True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")


# In[45]:


get_ipython().system('pip install scipy')
import scipy as scipy


# In[47]:


correlation_matrix = df.corr(numeric_only=True, method = 'kendall')
sns.heatmap(correlation_matrix, annot = True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")


# In[52]:


get_ipython().system('pip install pingouin')


# In[58]:


#looking at the relationship between the other selected independent variable (company) and the dependent variable (gross revenue)
import pingouin as pg 
a1 = pg.anova(data=df, dv='gross', between='company')
a1


# In[59]:


a2 = pg.anova(data=df, dv='gross', between='genre')
a2


# In[60]:


a3 = pg.anova(data=df, dv='gross', between='country')
a3


# In[ ]:


#key takeaways
#gross revenue is the dependent variable and others are the independent variable
#as budget increases, so does the gross revenue 
#strong relationship between company and gross revenue(35.6% of variability in gross revenue can be explained by the company). 
#choice of company significantly impacts gross revenue 
#low p value noticed in the anova table for gross revenue and genre. this means that there is a strong relationship between gross revenue and genre. 



