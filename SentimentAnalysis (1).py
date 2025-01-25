#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install nltk')


# ## Import the data 
# 
# Link to data - https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis/data

# In[7]:


import pandas as pd 
import numpy as np 


# In[8]:


df = pd.read_csv('sentiment_analysis.csv')


# In[9]:


df


# In[10]:


df['sentiment'].unique()


# In[11]:


df['sentiment'].value_counts()


# In[12]:


df = df[['text','sentiment']]


# In[13]:


df


# ## using Vader and SIA

# In[16]:


import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


# In[15]:


nltk.download('vader_lexicon')


# ### Understanding SIA

# In[17]:


sentence = "This movie was amazing, but the acting was a bit disappointing."
sentiment_scores = sia.polarity_scores(sentence)
print(sentiment_scores) 


# In[18]:


def analyze_sentiment_nltk(review):
    score = sia.polarity_scores(review)
    final_score = score['compound']
    if final_score > 0.4:
        return 'positive'
    elif final_score >= -0.3 and final_score <= 0.4:
        return 'neutral'
    else :
        return 'negative'


# In[19]:


df['predicted_sentiment_nltk'] = df['text'].apply(analyze_sentiment_nltk)


# In[20]:


df


# In[21]:


df['prediction_correctness_nltk'] = np.where(df['sentiment']==df['predicted_sentiment_nltk'],1,0)


# In[22]:


df


# In[23]:


df['prediction_correctness_nltk'].value_counts()


# In[ ]:





# In[ ]:





# ## Using textblob

# In[24]:


get_ipython().system(' pip install textblob')


# In[25]:


from textblob import TextBlob


# In[26]:


def analyze_sentiment_textblob(review):
    blob = TextBlob(review)
    if blob.sentiment.polarity >= 0.4:
        return 'positive'
    elif blob.sentiment.polarity >= -0.3 and blob.sentiment.polarity <= 0.4:
        return 'neutral'
    else : 
        return 'negative'


# In[27]:


analyze_sentiment_textblob('This is a good movie')


# In[28]:


df['predicted_sentiment_textblob'] = df['text'].apply(analyze_sentiment_textblob)


# In[29]:


df['prediction_correctness_textblob'] = np.where(df['sentiment']==df['predicted_sentiment_textblob'],1,0)


# In[30]:


df


# In[31]:


df['prediction_correctness_textblob'].value_counts()


# In[ ]:




