# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import nltk
import numpy as np
import re
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from matplotlib import pyplot
#plt.style.use('fivethirtyeight')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
#!pip install datashader
import datashader as ds
import datashader.transfer_functions as tf

df = pd.read_csv("data.csv",encoding='UTF-8')
print(len(df))
df.head(5)

unique_text = df.tweet.unique()
print(len(unique_text))

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

df['Clean_text'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")

df['Clean_text'] = df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = df['Clean_text'].apply(lambda x: x.split())

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
df['Clean_text'] = tokenized_tweet

df.loc[:,('tweet','Clean_text')]

df[df.tweet.isnull()]

df[df.Clean_text.isnull()]

unique_clean_text = df.Clean_text.unique()
unique_full_text = df.tweet.unique()
print(len(unique_clean_text))
print(len(unique_full_text))
print(len(df))

df.drop_duplicates(subset=['Clean_text'], keep = 'first',inplace= True)

df.reset_index(drop=True,inplace=True)

df['Clean_text_length'] = df['Clean_text'].apply(len)
#df.head()

df[df['Clean_text_length']==0]['Clean_text'] ## Looks like these are tweets with di
# We can simply drop these tweets
list = df[df['Clean_text_length']==0]['Clean_text'].index
#list

df.drop(index = list,inplace=True)

#df.info()

df.reset_index(drop=True,inplace=True)
#df.info()

from textblob import TextBlob

def calculate_sentiment(Clean_text):
    return TextBlob(Clean_text).sentiment

def calculate_sentiment_analyser(Clean_text):
    return analyser.polarity_scores(Clean_text)

df['sentiment']=df.Clean_text.apply(calculate_sentiment)
df['sentiment_analyser']=df.Clean_text.apply(calculate_sentiment_analyser)

s = pd.DataFrame(index = range(0,len(df)),columns= ['compound_score','compound_score_sentiment'])

for i in range(0,len(df)): 
  s['compound_score'][i] = df['sentiment_analyser'][i]['compound']
  
  if (df['sentiment_analyser'][i]['compound'] <= -0.05):
    s['compound_score_sentiment'][i] = 'Negative'    
  if (df['sentiment_analyser'][i]['compound'] >= 0.05):
    s['compound_score_sentiment'][i] = 'Positive'
  if ((df['sentiment_analyser'][i]['compound'] >= -0.05) & (df['sentiment_analyser'][i]['compound'] <= 0.05)):
    s['compound_score_sentiment'][i] = 'Neutral'
    
df['compound_score'] = s['compound_score']
df['compound_score_sentiment'] = s['compound_score_sentiment']
#df.head(4)

df.compound_score_sentiment.value_counts()

df['Clean_text'].head()

df.compound_score_sentiment.value_counts()

df['Clean_text'].head()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Considering 3 grams and mimnimum frq as 0
tf_idf_vect = CountVectorizer(analyzer='word',ngram_range=(1,1),stop_words='english', min_df = 0.0001)
tf_idf_vect.fit(df['Clean_text'])
desc_matrix = tf_idf_vect.transform(df["Clean_text"])

class Kmeans:
    def __init__(self, X, num_clusters):
        self.K = num_clusters # cluster number
        self.max_iterations = 100 # max iteration. don't want to run inf time
        self.num_examples, self.num_features = X.shape # num of examples, num of features
        self.plot_figure = True # plot figure
        
    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero 
        for k in range(self.K): # iterations of 
            centroid = X[np.random.choice(range(self.num_examples))] # random centroids
            centroids[k] = centroid
        return centroids # return random centroids
    def create_cluster(self, X, centroids):
      clusters = [[] for _ in range(self.K)]
      for point_idx, point in enumerate(X):
          closest_centroid = np.argmin(
              np.sqrt(np.sum((point-centroids)**2, axis=1))
          ) # closest centroid using euler distance equation(calculate distance of every point from centroid)
          clusters[closest_centroid].append(point_idx)
      return clusters 
  
    # new centroids
    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids
    def predict_cluster(self, clusters, X):
      y_pred = np.zeros(self.num_examples) # row1 fillup with zero
      for cluster_idx, cluster in enumerate(clusters):
          for sample_idx in cluster:
              y_pred[sample_idx] = cluster_idx
      return y_pred
    
    # plotinng scatter plot
    def plot_fig(self, X, y):
        fig = px.scatter(X[:, 0], X[:, 1], color=y)
        fig.show() # visualize
    def fit(self, X):
      centroids = self.initialize_random_centroids(X) # initialize random centroids
      for _ in range(self.max_iterations):
          clusters = self.create_cluster(X, centroids) # create cluster
          previous_centroids = centroids
          centroids = self.calculate_new_centroids(clusters, X) # calculate new centroids
          diff = centroids - previous_centroids # calculate difference
          if not diff.any():
              break
      y_pred = self.predict_cluster(clusters, X) # predict function
      if self.plot_figure: # if true
          self.plot_fig(X, y_pred) # plot function 
      return y_pred

# num_clusters = 3
# km = KMeans(n_clusters=num_clusters)
# km.fit(desc_matrix)
# clusters = km.labels_.tolist()

# # create DataFrame films from all of the input files.
# tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
# frame = pd.DataFrame(tweets, df['label'])
# frame

