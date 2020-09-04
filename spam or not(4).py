import pandas as pd 
import numpy as np 
import seaborn as sns 
import sklearn 
from sklearn.model_selection import train_test_split
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle 
def process_text(text):
     nopunc = [char for char in text if char not in string.punctuation]
     nopunc = ''.join(nopunc)
    
     clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
     return clean_words

def load():

    with open('model.pkl', 'rb') as file:
     vectorizer, clf = pickle.load(file) 
    return vectorizer, clf

vectorizer , classifier = load() 
msg = input("Enter email:")
email = [msg]
msg_transformed = vectorizer.transform(email)
prediction = classifier.predict(msg_transformed)

print("email is ", 'SPAM' if prediction else 'NOT SPAM')