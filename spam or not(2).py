from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split
#from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np

dataset = pd.read_csv(r"F:\python new\spam or not spam\spamham.csv")


#tot_mails = 5726
dataset.drop_duplicates(subset = 'text' , inplace = True)
#print(dataset.isnull().sum())
#print(dataset['spam'].value_counts())
X = dataset['text']
Y = dataset['spam']
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size = 0.20, random_state = 0)
print(y_train.value_counts())
print(y_test.value_counts())

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

