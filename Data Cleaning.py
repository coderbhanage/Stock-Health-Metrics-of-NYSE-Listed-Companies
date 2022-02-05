#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import csv
import datetime
import dateutil
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("prices-split-adjusted.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by = ['symbol','date'])
df['year'] = pd.DatetimeIndex(df['date']).year
grouped = df.groupby(df.year)


# In[4]:


df_10 =grouped.get_group(2010)
df_11 =grouped.get_group(2011)
df_12 =grouped.get_group(2012)
df_13 =grouped.get_group(2013)
df_14 =grouped.get_group(2014)
df_15 =grouped.get_group(2015)
df_16 =grouped.get_group(2016)


# In[5]:


company_list = list(set(df_10.symbol) & set(df_11.symbol) & set(df_12.symbol)                      & set(df_13.symbol) & set(df_14.symbol) & set(df_15.symbol)
                     & set(df_16.symbol))


# In[6]:


df_10 = df_10[df_10['symbol'].isin(company_list)]
df_11 = df_11[df_11['symbol'].isin(company_list)]
df_12 = df_12[df_12['symbol'].isin(company_list)]
df_13 = df_13[df_13['symbol'].isin(company_list)]
df_14 = df_14[df_14['symbol'].isin(company_list)]
df_15 = df_15[df_15['symbol'].isin(company_list)]
df_16 = df_16[df_16['symbol'].isin(company_list)]


# In[7]:


print(len(df_10.symbol.unique()))
print(len(df_11.symbol.unique()))
print(len(df_12.symbol.unique()))
print(len(df_13.symbol.unique()))
print(len(df_14.symbol.unique()))
print(len(df_15.symbol.unique()))
print(len(df_16.symbol.unique()))


# In[8]:


df_10 = df_10.drop_duplicates(subset=['symbol'], keep='last')
df_11 = df_11.drop_duplicates(subset=['symbol'], keep='last')
df_12 = df_12.drop_duplicates(subset=['symbol'], keep='last')
df_13 = df_13.drop_duplicates(subset=['symbol'], keep='last')
df_14 = df_14.drop_duplicates(subset=['symbol'], keep='last')
df_15 = df_15.drop_duplicates(subset=['symbol'], keep='last')
df_16 = df_16.drop_duplicates(subset=['symbol'], keep='last')


# In[9]:


df_10['stock_val_10'] = df_10['close'] * df_10['volume'] 
df_11['stock_val_11'] = df_11['close'] * df_11['volume'] 
df_12['stock_val_12'] = df_12['close'] * df_12['volume'] 
df_13['stock_val_13'] = df_13['close'] * df_13['volume'] 
df_14['stock_val_14'] = df_14['close'] * df_14['volume'] 
df_15['stock_val_15'] = df_15['close'] * df_15['volume'] 
df_16['stock_val_16'] = df_16['close'] * df_16['volume'] 


# In[10]:


df1 = pd.merge(df_10[['symbol','stock_val_10']],df_11[['symbol','stock_val_11']])
df2 = pd.merge(df_12[['symbol','stock_val_12']],df_13[['symbol','stock_val_13']])
df3 = pd.merge(df_14[['symbol','stock_val_14']],df_15[['symbol','stock_val_15']])
df4 = pd.merge(df1,df2)
df5 = pd.merge(df3,df_16[['symbol','stock_val_16']])
final_df = pd.merge(df4,df5)


# In[11]:


def rateing(row):
    stars = 0
    for i in range (1,6):
        if row[i+1]>row[i]:
            stars+=1
    return stars


# In[12]:


final_df['rating'] = final_df.apply (lambda row: rateing(row), axis=1)


# In[14]:


df = pd.read_csv("fundamentals.csv")


# In[15]:


df = df[["Ticker Symbol",
                      "After Tax ROE",
                      "Capital Expenditures",
                      "Cost of Revenue",
                      "Depreciation",
                      "Earnings Before Tax",
                      "Fixed Assets",
                      "Goodwill",
                      "Gross Margin",
                      "Gross Profit",
                      "Investments",
                      "Liabilities",
                      "Long-Term Debt",
                      "Long-Term Investments",
                      "Net Borrowings",
                      "Net Income",
                      "Total Assets",
                      "Total Liabilities & Equity",
                      "Total Revenue",
                      "For Year"
                      ]]


# In[16]:


df = df.rename(columns={'Ticker Symbol':"symbol"})


# In[17]:


company_list = list(set(final_df.symbol) & set(df.symbol))


# In[18]:


df = df[df['symbol'].isin(company_list)]
final_df = final_df[final_df['symbol'].isin(company_list)]


# In[19]:


df['For Year'] = df['For Year'].fillna(0)
df['For Year'] = df['For Year'].astype(int)
df = df.sort_values(by = ['symbol','For Year'])
df = df.drop_duplicates(subset=['symbol'], keep='last')
final_df = pd.merge(final_df,df)
columns = ['symbol', 'stock_val_10', 'stock_val_11', 'stock_val_12',
       'stock_val_13', 'stock_val_14', 'stock_val_15', 'stock_val_16', 'For Year', 'After Tax ROE', 'Capital Expenditures', 'Cost of Revenue',
       'Depreciation', 'Earnings Before Tax', 'Fixed Assets', 'Goodwill',
       'Gross Margin', 'Gross Profit', 'Investments', 'Liabilities',
       'Long-Term Debt', 'Long-Term Investments', 'Net Borrowings',
       'Net Income', 'Total Assets', 'Total Liabilities & Equity',
       'Total Revenue',
       'rating']
final_df = final_df.reindex(columns=columns)
final_df


# In[20]:


final_df.to_csv('final_data.csv')


# In[21]:


final_df.isnull().sum()


# In[26]:


final_df['rating'].value_counts()
count = final_df['rating'].value_counts()
sns.set(style="darkgrid")
sns.barplot(count.index, count.values, alpha=0.9)
plt.title('Frequency Distribution of Ratings')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Rting', fontsize=12)
plt.show()


# In[ ]:




