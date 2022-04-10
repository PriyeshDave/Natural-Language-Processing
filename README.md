# Natural Language Processing
Natural language processing (NLP) is a widely discussed and studied subject these days. 
NLP, one of the oldest areas of machine learning research, is used in major fields such as machine translation speech recognition and word processing. 
As part of this repository, I have implemented few projects on Natural Language Processing using **Python** and **Machine Learning**.

## 1.) Sentiment Analysis of IMDB Movie Reviews :smile::angry:⌛:camera:
### 🧭 Problem Statement:
In Machine Learning, Sentiment analysis refers to the application of natural language processing, computational linguistics, and text analysis to identify and classify subjective opinions in source documents. In this article, I will introduce you to a machine learning project on sentiment analysis with the Python programming language. Sentiment analysis aims to determine a writer’s attitude towards a topic or the overall contextual polarity of a document. The attitude can be his judgment or assessment, his emotional state or the intended emotional communication. The main task is to identify opinion words, which is very important. Opinion words are dominant indicators of feelings, especially adjectives, adverbs, and verbs, for example: “I love this camera. It’s amazing!”
In this project Ihave built a Machine Learning model on sentiment analysis with Python programming language which predicts the sentiment of the texts.

### 🧾 Dataset:
The dataset is taken from **IMDB movie reviews DB** which contains 40k movie reviews records. There are two prominent columns, one being TEXT which contains the criticism and the other being LABEL which contains the O’s and 1’s, where 0-NEGATIVE and 1-POSITIVE.

### :cloud: Word Cloud:
The word cloud consists of positive and negative words in the texts
* Positive Sentiments 
<img width="350" alt="positive-sentiments" src="https://user-images.githubusercontent.com/81012989/159269353-e922cbd2-b3dd-4e16-9d54-2817bbd10644.png">

* Negative Sentiments
<img width="355" alt="negative-sentiments" src="https://user-images.githubusercontent.com/81012989/159269374-5b99dc3a-0020-40d4-81a5-5f29aca929ac.png">

### Web Application :computer: :earth_americas: : 
Built a web application using Streamlit and deployed on Heroku.

<img width="451" alt="image" src="https://user-images.githubusercontent.com/81012989/159292592-756f3f43-b58c-4d1f-86d7-18faac38ee84.png">

## 2.) Fake News Detection using NLP and Machine Learning

### 🧭 Problem Statement:
The term fake news has become a buzz word these days.There was a time when if anyone needed any news, he or she would wait for the next-day newspaper. However, with the growth of online newspapers who update news almost instantly, people have found a better and faster way to be informed of the matter of his/her interest. Nowadays social-networking systems, online news portals, and other online media have become the main sources of news through which interesting and breaking news are shared at a rapid pace. However, many news portals serve special interest by feeding with distorted, partially correct, and sometimes imaginary news that is likely to attract the attention of a target group of people. Fake news has become a major concern for being destructive sometimes spreading confusion and deliberate disinformation among the people.

The aim of this projects is to use the Natural Language Processing and Machine learning to detect fake news based on the text content of the article.


### 🧾 Description:
This data set is collected from **Kaggle**. The data set has 20,800 news records with collection of real and fake news. It contains 5 columns (id, title, author, text and label) where text is the news that were are predicting upon and label has values as 0 or 1 where
* 0 -> Real news
* 1 -> Fake news.

#### 📊 Exploratory Data Analysis:
* Exploratory Data Analysis is the first step of understanding your data and acquiring domain knowledge.

#### ⌛ Data Preprocessing:
* Text columns is selected as the independent columns while label as the target dependent variable. All rest of the features were dropped.
* The text column was cleaned by removing the special and numerics characters if any in the text. I used python's **re** library to do this.
* After cleaning the text, the **stopwords** were removed from the sentences.
* After removing the stopwords, the sentences were stemmed by converting the words to their root words. I used nltk's **PorterStemmer** to achieve this.
* Post stemming, word vectorization was done using **TFIDF** vectorization.

#### ⚙ Model Training:
* Once the data is preprocessed, I used Logistic Regressor and fit it on the vectorized dataset.
* The model was trained with an accuracy of **98.5%** on training set and **97%** on the test set.


