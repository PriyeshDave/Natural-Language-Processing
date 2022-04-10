import pandas as pd
import numpy as np
import seaborn as sns

import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import re
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import fuzzywuzzy
from fuzzywuzzy import fuzz

lm = WordNetLemmatizer()
sw = stopwords.words('English')
cv = joblib.load('../Model/tfidf_vectorizer.pkl')

def cleanData(sentence):
  if sentence.__contains__('[math]'):
    return sentence
  else:
    cleaned_sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    cleaned_sentence = re.sub(' +', " ", cleaned_sentence)
    return cleaned_sentence

def removeStopWords(sentence):
  sentence = cleanData(sentence)
  sentence = sentence.lower()
  words = nltk.word_tokenize(sentence)
  words = [lm.lemmatize(word) for word in words if not word in sw]
  return " ".join(words)

def getUnique(arr):
  return len(set(arr.split(' ')))

def getCommon(arr1, arr2):

  set1 = set(arr1.split(' '))
  set2 = set(arr2.split(' '))
  return len(set1 & set2)

def wordsTotal(arr1, arr2):

  set1 = set(arr1.split(' '))
  set2 = set(arr2.split(' '))
  return len(set1) + len(set2)


def predict(q1, q2):
  q1 = q1.apply(removeStopWords)
  q2 = q2.apply(removeStopWords)

  q1_arr = cv.transform(q1).toarray()
  q2_arr = cv.transform(q2).toarray()

  temp_df1 = pd.DataFrame(q1_arr,  columns=list(range(0,3000)))
  temp_df2 = pd.DataFrame(q2_arr,  columns=list(range(3001,6001)))
  temp_df = pd.concat([temp_df1, temp_df2], axis=1)

  q1_len = [len(sentence) for sentence in q1]
  q2_len = [len(sentence) for sentence in q2]

  # Number of words in each sentences
  q1_words = [len(arr.split(' ')) for arr in q1]
  q2_words = [len(arr.split(' ')) for arr in q2]

  common_words = [getCommon(arr1, arr2) for arr1, arr2 in zip(q1, q2)]

  total_words = [wordsTotal(arr1, arr2) for arr1, arr2 in zip(q1, q2)]

  temp_df['q1_len'] = q1_len
  temp_df['q2_len'] = q2_len

  temp_df['q1_words'] = q1_words
  temp_df['q2_words'] = q2_words

  temp_df['common_words'] = common_words
  temp_df['total_words'] = total_words

  temp_df['words_share'] = round(temp_df['common_words']/temp_df['total_words'],2)

  temp_df['token_set'] = fuzz.token_set_ratio(q1, q2)
  temp_df['token_sort'] = fuzz.token_sort_ratio(q1, q2)
  temp_df['partial_token_set'] = fuzz.partial_token_set_ratio(q1, q2)
  temp_df['partial_token_sort'] = fuzz.partial_token_sort_ratio(q1, q2)

  return temp_df


