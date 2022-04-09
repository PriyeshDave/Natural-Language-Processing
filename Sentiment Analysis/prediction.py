import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

sw = stopwords.words('English')
ps = PorterStemmer()
lm = WordNetLemmatizer()

def cleanData(value):
  value = re.sub("[^a-zA-Z]", " ", value)
  value = re.sub(" +", " ", value)
  return value

def stemSentences(sentence):
  sentence = cleanData(sentence)
  wordsList = list()
  sentence = sentence.lower()
  wordsList = sentence.split(" ")
  stemmedWords = [ps.stem(x) for x in wordsList if not x in sw]
  return " ".join(stemmedWords)

def lemamtizeSentence(sentence):
  sentence = cleanData(sentence)
  wordsList = list()
  print("Sentence here",type(sentence))
  sentence = sentence.lower()
  wordsList = sentence.split(" ")
  print('Word List', wordsList)
  stemmedWords = [lm.lemmatize(x) for x in wordsList if not x in sw]
  return " ".join(stemmedWords)

def get_prediction(text, model, tfidf_vectorizer):
  vectorized_data = tfidf_vectorizer.transform([text]).toarray()
  print('Vectorized Text', vectorized_data)
  prediction = model.predict(vectorized_data)
  return prediction[0]





